"""Make labeled forgeries of real blot panels.

Every forgery returns the edited panel plus a binary mask of exactly which
pixels were changed, so the benchmark has pixel level ground truth.

Methods (names are used for held out evaluation):
  copy_move     a band copied from another lane of the same blot
  cross_splice  a band taken from a different real blot
  ai_splice     a band taken from an AI generated blot (Gemini)
  intensify     an existing band darkened and thickened in place
  erase         a band removed by inpainting the background over it
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from .bands import Band, find_bands, background
from .imgio import to_gray, normalize_polarity

METHODS = ["copy_move", "cross_splice", "ai_splice", "intensify", "erase"]


@dataclass
class Forgery:
    image: Image.Image
    mask: np.ndarray  # uint8 0/1, same HxW as image
    method: str
    target: tuple[int, int, int, int]  # x0, y0, x1, y1 of the edited region
    blend: str


def _pad_box(b: Band, pad_x: float, pad_y: float, W: int, H: int):
    px, py = int(b.w * pad_x) + 2, int(b.h * pad_y) + 2
    return max(0, b.x - px), max(0, b.y - py), min(W, b.x + b.w + px), min(H, b.y + b.h + py)


def _band_patch(gray: np.ndarray, band: Band, rng: random.Random):
    H, W = gray.shape
    x0, y0, x1, y1 = _pad_box(band, rng.uniform(0.05, 0.25), rng.uniform(0.3, 0.9), W, H)
    return gray[y0:y1, x0:x1].copy()


def _match_background(patch: np.ndarray, target_bg: float) -> np.ndarray:
    src_bg = float(np.percentile(patch, 75))  # most of a band patch is background
    if src_bg < 1:
        return patch
    return np.clip(patch * (target_bg / src_bg), 0, 255)


def _paste(gray: np.ndarray, patch: np.ndarray, cx: int, cy: int, blend: str,
           rng: random.Random):
    """Paste patch centered at (cx, cy). Returns new gray and changed mask."""
    H, W = gray.shape
    ph, pw = patch.shape
    x0, y0 = int(cx - pw / 2), int(cy - ph / 2)
    x0, y0 = max(0, min(W - pw, x0)), max(0, min(H - ph, y0))
    region = gray[y0:y0 + ph, x0:x0 + pw]
    if region.shape != patch.shape:
        return None, None
    if blend == "hard":
        alpha = np.ones_like(patch)
    elif blend == "feather":
        alpha = np.ones_like(patch)
        k = max(1, min(ph, pw) // 6)
        alpha[:k] *= np.linspace(0, 1, k)[:, None]
        alpha[-k:] *= np.linspace(1, 0, k)[:, None]
        alpha[:, :k] *= np.linspace(0, 1, k)[None, :]
        alpha[:, -k:] *= np.linspace(1, 0, k)[None, :]
    else:  # "darkness": only the band itself is transferred, like the manual composite
        bg = float(np.median(np.concatenate([patch[0], patch[-1]])))
        alpha = np.clip((bg - patch) / max(10.0, rng.uniform(25, 70)), 0, 1)
        alpha = cv2.GaussianBlur(alpha.astype(np.float32), (3, 3), 0)
    out = gray.copy()
    if blend == "hard":
        src = patch
    else:  # a pasted band can only add darkness on top of the membrane
        src = np.minimum(region, patch)
    out[y0:y0 + ph, x0:x0 + pw] = region * (1 - alpha) + src * alpha
    mask = np.zeros_like(gray, dtype=np.uint8)
    changed = np.abs(out[y0:y0 + ph, x0:x0 + pw] - region) > 2.0
    mask[y0:y0 + ph, x0:x0 + pw] = changed.astype(np.uint8)
    return out, mask


def _target_location(bands: list[Band], src: Band, W: int, rng: random.Random):
    """Pick a location in the same row as the source band but another lane."""
    row = [b for b in bands if abs((b.y + b.h / 2) - (src.y + src.h / 2)) < max(4, src.h)]
    others = [b for b in row if b.id != src.id]
    if others and rng.random() < 0.85:
        t = rng.choice(others)
        return t.x + t.w // 2, t.y + t.h // 2
    # Otherwise an empty spot in the same row.
    cx = rng.randint(src.w // 2 + 1, max(src.w // 2 + 2, W - src.w // 2 - 1))
    return cx, src.y + src.h // 2


def _scale_patch(patch: np.ndarray, w: int, h: int) -> np.ndarray:
    w, h = max(4, w), max(3, h)
    return cv2.resize(patch.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA
                      if w < patch.shape[1] else cv2.INTER_CUBIC)


def forge(panel: Image.Image, method: str, rng: random.Random,
          donor_panels: list[Image.Image] | None = None,
          ai_panels: list[Image.Image] | None = None) -> Forgery | None:
    gray0 = to_gray(panel)
    gray, inverted = normalize_polarity(gray0)
    H, W = gray.shape
    bands = [b for b in find_bands(gray) if b.w >= 8 and b.h >= 3
             and b.y > 3 and b.y + b.h < H - 3 and b.x > 2 and b.x + b.w < W - 2]
    if len(bands) < 2:
        return None
    bg_level = float(np.median(background(gray)))
    r = rng.random()
    blend = "darkness" if r < 0.55 else ("feather" if r < 0.85 else "hard")

    if method in ("copy_move", "cross_splice", "ai_splice"):
        target_band = rng.choice(bands)
        if method == "copy_move":
            vis = [b for b in bands if b.darkness > 20] or bands
            src_gray, src_band = gray, rng.choice(vis)
        else:
            pool = donor_panels if method == "cross_splice" else ai_panels
            if not pool:
                return None
            for _ in range(6):
                donor = rng.choice(pool)
                dg, _ = normalize_polarity(to_gray(donor))
                dh, dw = dg.shape
                db = [b for b in find_bands(dg) if b.w >= 10 and b.h >= 4 and b.darkness > 20
                      and b.y > 2 and b.y + b.h < dh - 2 and b.x > 2 and b.x + b.w < dw - 2]
                if db:
                    src_gray, src_band = dg, rng.choice(db)
                    break
            else:
                return None
        patch = _band_patch(src_gray, src_band, rng)
        # Resize the donor band to the size of the band it will sit next to.
        sx = target_band.w / max(1, src_band.w) * rng.uniform(0.85, 1.25)
        sy = target_band.h / max(1, src_band.h) * rng.uniform(0.85, 1.4)
        patch = _scale_patch(patch, int(patch.shape[1] * sx), int(patch.shape[0] * sy))
        patch = _match_background(patch, bg_level)
        if patch.shape[0] >= H or patch.shape[1] >= W:
            return None
        cx, cy = _target_location(bands, target_band, W, rng)
        out, mask = _paste(gray, patch, cx, cy, blend, rng)
        if out is None:
            return None

    elif method == "intensify":
        b = rng.choice(bands)
        x0, y0, x1, y1 = _pad_box(b, 0.15, 0.6, W, H)
        region = gray[y0:y1, x0:x1]
        bgl = float(np.median(np.concatenate([region[0], region[-1]])))
        dark = np.clip(bgl - region, 0, None)
        gain = rng.uniform(1.4, 2.6)
        grow = rng.choice([1, 3, 5])
        dark2 = cv2.dilate(dark, np.ones((grow, grow), np.uint8)) if grow > 1 else dark
        new = np.clip(bgl - dark2 * gain, 0, 255)
        out = gray.copy(); out[y0:y1, x0:x1] = new
        mask = np.zeros((H, W), np.uint8)
        mask[y0:y1, x0:x1] = (np.abs(new - region) > 2).astype(np.uint8)
        blend = f"gain{gain:.1f}"

    elif method == "erase":
        b = rng.choice(bands)
        x0, y0, x1, y1 = _pad_box(b, 0.05, 0.25, W, H)
        m = np.zeros((H, W), np.uint8); m[y0:y1, x0:x1] = 1
        g8 = np.clip(gray, 0, 255).astype(np.uint8)
        algo = cv2.INPAINT_TELEA if rng.random() < 0.5 else cv2.INPAINT_NS
        out = cv2.inpaint(g8, m, 3, algo).astype(np.float32)
        mask = (np.abs(out - gray) > 2).astype(np.uint8) * m
        blend = "inpaint"
    else:
        raise ValueError(method)

    if mask.sum() < 12:
        return None
    if inverted:
        out = 255.0 - out
    img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).convert("RGB")
    ys, xs = np.nonzero(mask)
    return Forgery(img, mask, method, (int(xs.min()), int(ys.min()), int(xs.max()) + 1,
                                       int(ys.max()) + 1), blend)
