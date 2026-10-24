"""Small convolutional patch classifier (pure JAX, runs on CPU).

Input per patch: two channels, both normalized by statistics of the panel
the patch came from, so the network judges a region against its own blot:
  0. intensity relative to the panel background
  1. high pass noise residual divided by the panel's typical residual level
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .bands import darkness_map
from .imgio import normalize_polarity

P = 32          # patch size
STRIDE = 8
WEIGHTS = Path(__file__).resolve().parents[1] / "models" / "splice_cnn.npz"


def panel_channels(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g, _ = normalize_polarity(gray)
    g = g.astype(np.float32)
    bg = float(np.median(g))
    lo = float(np.percentile(g, 1))
    c0 = (g - bg) / max(8.0, bg - lo)
    res = g - cv2.GaussianBlur(g, (0, 0), 0.8)
    c1 = res / (float(np.median(np.abs(res))) * 1.4826 + 0.5)
    c1 = np.clip(c1, -8, 8) / 4.0
    dk = darkness_map(g)
    return np.stack([c0, c1], -1).astype(np.float32), dk


def grid(h: int, w: int, p: int = P, s: int = STRIDE):
    if h < p or w < p:
        return []
    ys = list(range(0, h - p + 1, s)); xs = list(range(0, w - p + 1, s))
    if ys[-1] != h - p: ys.append(h - p)
    if xs[-1] != w - p: xs.append(w - p)
    return [(x, y) for y in ys for x in xs]


def pad_small(chans: np.ndarray, dk: np.ndarray, mask=None):
    h, w = dk.shape
    ph, pw = max(0, P - h), max(0, P - w)
    if ph or pw:
        chans = np.pad(chans, ((0, ph), (0, pw), (0, 0)), mode="edge")
        dk = np.pad(dk, ((0, ph), (0, pw)), mode="edge")
        if mask is not None:
            mask = np.pad(mask, ((0, ph), (0, pw)))
    return chans, dk, mask


def extract(gray: np.ndarray, mask: np.ndarray | None = None):
    """All grid patches of a panel. Returns patches, positions, band flag, label frac."""
    chans, dk = panel_channels(gray)
    chans, dk, mask = pad_small(chans, dk, mask)
    pos = grid(*dk.shape)
    X = np.stack([chans[y:y + P, x:x + P] for x, y in pos]) if pos else np.zeros((0, P, P, 2), np.float32)
    band = np.array([(dk[y:y + P, x:x + P] > 15).mean() > 0.12 for x, y in pos], bool)
    frac = (np.array([mask[y:y + P, x:x + P].mean() for x, y in pos], np.float32)
            if mask is not None else np.zeros(len(pos), np.float32))
    return X, pos, band, frac


# ---------------------------------------------------------------- model
def init_params(seed: int = 0):
    rng = np.random.default_rng(seed)

    def conv(cin, cout):
        return {"w": (rng.standard_normal((3, 3, cin, cout)) * np.sqrt(2 / (9 * cin))).astype(np.float32),
                "b": np.zeros(cout, np.float32)}
    return {"c1": conv(2, 16), "c2": conv(16, 16), "c3": conv(16, 32), "c4": conv(32, 32),
            "c5": conv(32, 64),
            "fc1": {"w": (rng.standard_normal((128, 64)) * np.sqrt(2 / 128)).astype(np.float32),
                    "b": np.zeros(64, np.float32)},
            "fc2": {"w": (rng.standard_normal((64, 1)) * np.sqrt(1 / 64)).astype(np.float32),
                    "b": np.zeros(1, np.float32)}}


def apply(params, x):
    import jax
    import jax.numpy as jnp

    def conv(p, h):
        h = jax.lax.conv_general_dilated(h, p["w"], (1, 1), "SAME",
                                         dimension_numbers=("NHWC", "HWIO", "NHWC"))
        return jax.nn.relu(h + p["b"])

    def pool(h):
        return jax.lax.reduce_window(h, -jnp.inf, jax.lax.max, (1, 2, 2, 1), (1, 2, 2, 1), "VALID")

    h = conv(params["c1"], x); h = conv(params["c2"], h); h = pool(h)
    h = conv(params["c3"], h); h = conv(params["c4"], h); h = pool(h)
    h = conv(params["c5"], h)
    h = jnp.concatenate([h.mean(axis=(1, 2)), h.max(axis=(1, 2))], -1)
    h = jax.nn.relu(h @ params["fc1"]["w"] + params["fc1"]["b"])
    return (h @ params["fc2"]["w"] + params["fc2"]["b"])[:, 0]


_PREDICT = None


def predict(params, X: np.ndarray, batch: int = 1024, logits: bool = False) -> np.ndarray:
    """Patch probabilities (or raw logits, which do not saturate near 1)."""
    import jax
    global _PREDICT
    if _PREDICT is None:
        _PREDICT = jax.jit(apply)
    out = []
    for i in range(0, len(X), batch):
        xb = np.asarray(X[i:i + batch], np.float32)
        n = len(xb)
        if n < batch:  # pad to a fixed shape so jit compiles once
            xb = np.concatenate([xb, np.zeros((batch - n, *xb.shape[1:]), xb.dtype)])
        out.append(np.asarray(_PREDICT(params, xb))[:n])
    z = np.concatenate(out) if out else np.zeros(0, np.float32)
    return z if logits else 1.0 / (1.0 + np.exp(-z))


RULES = ("top", "outlier", "z")


def relative(logits: np.ndarray, band: np.ndarray, rule: str, k: int = 3):
    """Per patch values on the rule's scale, and the panel score.

    top:     raw logit (how forged a patch looks in absolute terms)
    outlier: logit minus the median band patch of the SAME panel
    z:       outlier divided by the panel's spread, so a busy blot is not
             flagged just for being busy
    The panel score is the mean of the top k band patches on that scale.
    """
    use = band if band.sum() >= 3 else np.ones(len(logits), bool)
    p = logits[use]
    if len(p) == 0:
        return logits, 0.0
    med = float(np.median(p))
    spread = float(np.median(np.abs(p - med))) * 1.4826 + 0.25
    if rule == "outlier":
        r = logits - med
    elif rule == "z":
        r = (logits - med) / spread
    else:
        r = logits
    pr = r[use]
    return r, float(np.sort(pr)[-min(k, len(pr)):].mean())


def save(params, path: Path = WEIGHTS, **meta):
    flat = {f"{k}.{kk}": np.asarray(vv) for k, v in params.items() for kk, vv in v.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **flat, **{f"meta.{k}": np.asarray(v) for k, v in meta.items()})


def load(path: Path = WEIGHTS):
    z = np.load(path)
    params, meta = {}, {}
    for k in z.files:
        a, b = k.split(".", 1)
        if a == "meta":
            meta[b] = z[k].item() if z[k].shape == () else z[k]
        else:
            params.setdefault(a, {})[b] = z[k]
    return params, meta
