"""Per window forensic features for splice detection.

The idea: a pasted or edited region was made under different conditions
(another camera, another blot, a generator, a resampling step) so its noise,
compression and edge statistics disagree with the rest of the same panel.
Every feature is therefore expressed RELATIVE to the panel it came from,
which is what lets one model work across blots with very different looks.
"""
from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image

from .bands import darkness_map
from .imgio import normalize_polarity

WIN = 24
STRIDE = 12

FEATURE_NAMES = [
    "res_std_ratio", "res_std_bg_ratio", "res_kurt", "lap_energy_ratio",
    "ela90_ratio", "ela75_ratio", "block_ratio", "block_phase_ratio",
    "grad_max_ratio", "edge_sharp_ratio", "bg_level_dev", "bg_flat_ratio",
    "dark_mean", "dark_frac", "clip_frac_dev", "quant_hist_dev",
    "res_std_vs_neighbors", "ela_vs_neighbors", "noise_hf_ratio", "smooth_frac_ratio",
]


def _ela(g8: np.ndarray, q: int) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(g8).save(buf, format="JPEG", quality=q)
    buf.seek(0)
    re = np.asarray(Image.open(buf).convert("L"), dtype=np.float32)
    return np.abs(g8.astype(np.float32) - re)


def _blockiness(g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per pixel JPEG 8x8 grid discontinuity, aligned and misaligned."""
    dx = np.abs(np.diff(g, axis=1, prepend=g[:, :1]))
    dy = np.abs(np.diff(g, axis=0, prepend=g[:1, :]))
    cols = np.arange(g.shape[1]) % 8 == 0
    rows = np.arange(g.shape[0]) % 8 == 0
    aligned = np.zeros_like(g)
    aligned[:, cols] += dx[:, cols]
    aligned[rows, :] += dy[rows, :]
    off = np.zeros_like(g)
    cols4 = np.arange(g.shape[1]) % 8 == 4
    rows4 = np.arange(g.shape[0]) % 8 == 4
    off[:, cols4] += dx[:, cols4]
    off[rows4, :] += dy[rows4, :]
    return aligned, off


def _box_mean(a: np.ndarray, k: int) -> np.ndarray:
    return cv2.boxFilter(a.astype(np.float32), -1, (k, k), normalize=True,
                         borderType=cv2.BORDER_REFLECT)


def panel_feature_maps(gray: np.ndarray) -> dict[str, np.ndarray]:
    g, _ = normalize_polarity(gray)
    g = g.astype(np.float32)
    g8 = np.clip(g, 0, 255).astype(np.uint8)
    med = cv2.medianBlur(g8, 3).astype(np.float32)
    res = g - med
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    hf = g - cv2.GaussianBlur(g, (0, 0), 0.8)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    dark = darkness_map(g)
    aligned, off = _blockiness(g)
    return {
        "g": g, "res": res, "res2": res * res, "res4": res ** 4, "lap2": lap * lap,
        "hf2": hf * hf, "ela90": _ela(g8, 90), "ela75": _ela(g8, 75), "blk": aligned,
        "blk_off": off, "grad": grad, "dark": dark,
        "clip": ((g8 <= 2) | (g8 >= 253)).astype(np.float32),
        "smooth": (np.abs(res) < 0.5).astype(np.float32),
        "bgmask": (dark < 6).astype(np.float32),
    }


def window_grid(h: int, w: int, win: int = WIN, stride: int = STRIDE):
    ys = list(range(0, max(1, h - win + 1), stride)) or [0]
    xs = list(range(0, max(1, w - win + 1), stride)) or [0]
    if ys[-1] + win < h:
        ys.append(max(0, h - win))
    if xs[-1] + win < w:
        xs.append(max(0, w - win))
    return [(x, y, min(win, w - x), min(win, h - y)) for y in ys for x in xs]


def window_features(gray: np.ndarray, win: int = WIN, stride: int = STRIDE):
    """Return (windows, X) where X has one row of FEATURE_NAMES per window."""
    m = panel_feature_maps(gray)
    h, w = m["g"].shape
    wins = window_grid(h, w, win, stride)
    eps = 1e-3

    # Panel level references (robust medians over windows).
    def wmean(a, x, y, ww, hh):
        return float(a[y:y + hh, x:x + ww].mean())

    raw = []
    for (x, y, ww, hh) in wins:
        sl = (slice(y, y + hh), slice(x, x + ww))
        bgm = m["bgmask"][sl]
        r2 = m["res2"][sl]
        res_std = np.sqrt(r2.mean())
        res_std_bg = np.sqrt((r2 * bgm).sum() / (bgm.sum() + eps)) if bgm.sum() > 8 else np.nan
        kurt = m["res4"][sl].mean() / (r2.mean() ** 2 + eps)
        grad = m["grad"][sl]
        gbg = m["g"][sl][bgm > 0]
        q = np.round(m["g"][sl]).astype(int)
        hist = np.bincount(np.clip(q.ravel(), 0, 255), minlength=256)
        gaps = float(np.mean(hist[q.min():q.max() + 1] == 0)) if q.max() > q.min() else 0.0
        raw.append([
            res_std, res_std_bg, kurt, m["lap2"][sl].mean(),
            m["ela90"][sl].mean(), m["ela75"][sl].mean(), m["blk"][sl].mean(),
            m["blk"][sl].mean() / (m["blk_off"][sl].mean() + eps),
            float(np.percentile(grad, 98)), float(grad.max()),
            float(np.median(gbg)) if gbg.size > 8 else np.nan,
            float(np.std(gbg)) if gbg.size > 8 else np.nan,
            m["dark"][sl].mean(), float((m["dark"][sl] > 15).mean()), m["clip"][sl].mean(),
            gaps, res_std, m["ela90"][sl].mean(), m["hf2"][sl].mean(), m["smooth"][sl].mean(),
        ])
    raw = np.array(raw, dtype=np.float64)

    # Reference statistics. Background windows are the fairest reference for
    # noise; band windows for edges.
    is_bg = raw[:, 13] < 0.15
    with np.errstate(all='ignore'):
        ref_all = np.nanmedian(raw, axis=0)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ref_bg = np.nanmedian(raw[is_bg], axis=0) if is_bg.sum() >= 3 else ref_all
        ref_band = np.nanmedian(raw[~is_bg], axis=0) if (~is_bg).sum() >= 3 else ref_all
    ref_bg = np.where(np.isnan(ref_bg), ref_all, ref_bg)
    ref_band = np.where(np.isnan(ref_band), ref_all, ref_band)

    X = np.zeros((len(wins), len(FEATURE_NAMES)), dtype=np.float32)
    ratio = lambda v, r: np.log((v + 1e-3) / (r + 1e-3))
    # Compare like with like: band windows against the panel's other bands,
    # background windows against the panel's background.
    band = ~is_bg
    ref = np.where(band[:, None], ref_band[None, :], ref_bg[None, :])
    for col in (0, 3, 4, 5, 6, 7, 18, 19):
        X[:, col] = ratio(raw[:, col], ref[:, col])
    X[:, 1] = np.nan_to_num(ratio(raw[:, 1], ref_bg[1]), nan=0.0)
    X[:, 2] = np.log(raw[:, 2] + 1) - np.log(ref[:, 2] + 1)
    X[:, 8] = ratio(raw[:, 8], ref_band[8])
    X[:, 9] = ratio(raw[:, 9], ref_band[9])
    X[:, 10] = np.nan_to_num((raw[:, 10] - ref_bg[10]) / 20.0, nan=0.0)
    X[:, 11] = np.nan_to_num(ratio(raw[:, 11], ref_bg[11]), nan=0.0)
    # Band density relative to the panel's bands, not absolute (absolute
    # darkness only says "this is a band", which is a shortcut).
    X[:, 12] = (raw[:, 12] - ref_band[12]) / 50.0
    X[:, 13] = raw[:, 13]
    X[:, 14] = raw[:, 14] - ref[:, 14]
    X[:, 15] = raw[:, 15] - ref[:, 15]
    # Neighbourhood contrast: compare each window with the windows around it.
    pos = np.array([(x, y) for (x, y, _, _) in wins])
    for col, src in ((16, 0), (17, 4)):
        v = raw[:, src]
        nb = np.zeros(len(wins))
        for i, (x, y) in enumerate(pos):
            d = np.abs(pos - [x, y]).max(axis=1)
            sel = (d > 0) & (d <= 2 * STRIDE)
            nb[i] = np.median(v[sel]) if sel.any() else v[i]
        X[:, col] = ratio(v, nb)
    X[:, 18] = ratio(raw[:, 18], ref_all[18])
    X[:, 19] = ratio(raw[:, 19], ref_all[19])
    X = np.nan_to_num(X, nan=0.0, posinf=5.0, neginf=-5.0)
    return wins, X


def window_labels(wins, mask: np.ndarray, frac: float = 0.2) -> np.ndarray:
    y = np.zeros(len(wins), dtype=np.uint8)
    for i, (x, yy, ww, hh) in enumerate(wins):
        if mask[yy:yy + hh, x:x + ww].mean() >= frac:
            y[i] = 1
    return y


def heatmap(wins, scores, shape) -> np.ndarray:
    acc = np.zeros(shape, np.float32)
    cnt = np.zeros(shape, np.float32)
    for (x, y, ww, hh), s in zip(wins, scores):
        acc[y:y + hh, x:x + ww] = np.maximum(acc[y:y + hh, x:x + ww], s)
        cnt[y:y + hh, x:x + ww] += 1
    return acc
