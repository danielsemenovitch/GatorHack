"""Find duplicated bands: the classic copy and paste fraud.

Two different real bands can look alike in SHAPE, but they never share the
same NOISE. A copied region carries its pixel noise with it. So for every
band we take its high pass residual (shape mostly removed, noise kept) and
search the blot for the same residual pattern at native resolution, also
trying horizontal and vertical flips. A strong residual match somewhere
other than the band itself means those pixels were copied.
"""
from __future__ import annotations

import cv2
import numpy as np

from .bands import Band, find_bands
from .imgio import normalize_polarity

FLIPS = {
    "none": lambda a: a,
    "flip_h": lambda a: a[:, ::-1],
    "flip_v": lambda a: a[::-1, :],
    "rot180": lambda a: a[::-1, ::-1],
}


def _residual(g: np.ndarray) -> np.ndarray:
    """Grain only: high pass, then remove line structure (band edges, frames).

    Band edges and frame lines are long horizontal or vertical structures
    that look alike in every band; subtracting local row and column means
    removes them and leaves the pixel grain that a copy carries along.
    """
    g = g.astype(np.float32)
    r = g - cv2.GaussianBlur(g, (0, 0), 0.8)
    r = r - cv2.boxFilter(r, -1, (15, 1), borderType=cv2.BORDER_REFLECT)
    r = r - cv2.boxFilter(r, -1, (1, 7), borderType=cv2.BORDER_REFLECT)
    return r


def _template_box(b: Band, H: int, W: int):
    py, px = max(2, b.h // 2), 3
    return max(0, b.x - px), max(0, b.y - py), min(W, b.x + b.w + px), min(H, b.y + b.h + py)


MIN_W, MIN_H = 24, 8  # below this, "noise" is mostly band shape and real bands match


def find_duplicates(panels: list[np.ndarray], threshold: float = 0.5,
                    shape_floor: float = 0.75, max_pairs: int = 20,
                    min_w: int = MIN_W, min_h: int = MIN_H) -> list[dict]:
    """panels: grayscale arrays. Returns copied band pairs, strongest first."""
    prep = []
    for gray in panels:
        g, _ = normalize_polarity(gray)
        g = g.astype(np.float32)
        H, W = g.shape
        bands = [b for b in find_bands(g) if b.w >= min_w and b.h >= min_h
                 and b.y > 2 and b.y + b.h < H - 2 and b.x > 2 and b.x + b.w < W - 2]
        prep.append((g, _residual(g), bands))
    found = {}
    for pi, (g, res, bands) in enumerate(prep):
        H, W = g.shape
        for b in bands:
            x0, y0, x1, y1 = _template_box(b, H, W)
            t_res, t_raw = res[y0:y1, x0:x1], g[y0:y1, x0:x1]
            th, tw = t_res.shape
            if th < 6 or tw < 12 or t_res.std() < 0.3:
                continue
            for pj, (g2, res2, _) in enumerate(prep):
                if pj < pi or g2.shape[0] < th or g2.shape[1] < tw:
                    continue
                for name, f in FLIPS.items():
                    R = cv2.matchTemplate(res2, np.ascontiguousarray(f(t_res)), cv2.TM_CCOEFF_NORMED)
                    if pj == pi:  # ignore the band itself and its own mirrored halves next to it
                        R[max(0, y0 - th // 2):y0 + th // 2 + 1, max(0, x0 - tw):x0 + tw + 1] = -1
                    _, mx, _, loc = cv2.minMaxLoc(R)
                    if mx < threshold:
                        continue
                    lx, ly = loc
                    cand = g2[ly:ly + th, lx:lx + tw]
                    shape = float(cv2.matchTemplate(cand, np.ascontiguousarray(f(t_raw)), cv2.TM_CCOEFF_NORMED)[0, 0])
                    if shape < shape_floor:
                        continue
                    key = (pi, b.id, pj, lx // 4, ly // 4)
                    rev = (pj, None, pi, x0 // 4, y0 // 4)
                    if key in found or rev in found:
                        continue
                    found[key] = {
                        "a": {"panel": pi, **b.to_dict()},
                        "b": {"panel": pj, "id": -1, "x": int(lx + (b.x - x0)), "y": int(ly + (b.y - y0)),
                              "w": int(b.w), "h": int(b.h)},
                        "transform": name, "noise_similarity": round(float(mx), 3),
                        "shape_similarity": round(shape, 3),
                    }
    out = sorted(found.values(), key=lambda d: -d["noise_similarity"])
    # Collapse symmetric duplicates (A matches B and B matches A).
    seen, uniq = set(), []
    for d in out:
        k = tuple(sorted([(d["a"]["panel"], d["a"]["x"] // 6, d["a"]["y"] // 6),
                          (d["b"]["panel"], d["b"]["x"] // 6, d["b"]["y"] // 6)]))
        if k not in seen:
            seen.add(k); uniq.append(d)
    return uniq[:max_pairs]
