"""Find bands (and the lanes they sit in) inside a western blot panel.

A band is a horizontal blob that is darker than the local background. We
estimate the background with a large median filter, threshold the darkness
map with Otsu, and keep connected components that look like bands.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import cv2
import numpy as np

from .imgio import normalize_polarity


@dataclass
class Band:
    id: int
    x: int
    y: int
    w: int
    h: int
    darkness: float  # mean darkness above background, 0..255

    def box(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.x + self.w, self.y + self.h

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: (float(v) if k == "darkness" else int(v)) for k, v in d.items()}


def background(gray: np.ndarray) -> np.ndarray:
    """Membrane background with dark horizontal bands removed.

    Grey closing with a vertical line removes any dark feature shorter than
    the line, whatever its width, which is exactly what a band is. A light
    smoothing afterwards suppresses the blockiness of the closing.
    """
    h, w = gray.shape
    L = int(np.clip(h // 3, 9, 61)) | 1
    g8 = np.clip(gray, 0, 255).astype(np.uint8)
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (1, L))
    bg = cv2.morphologyEx(g8, cv2.MORPH_CLOSE, se)
    bg = cv2.morphologyEx(bg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)))
    k = int(np.clip(min(h, w) // 10, 5, 31)) | 1
    return cv2.GaussianBlur(bg.astype(np.float32), (k, k), 0)


def darkness_map(gray: np.ndarray) -> np.ndarray:
    g, _ = normalize_polarity(gray)
    bg = np.maximum(background(g), g)  # background can only be lighter than bands
    return np.clip(bg - g, 0, 255)


def find_bands(gray: np.ndarray, min_w: int = 6, min_h: int = 2,
               max_frac: float = 0.6) -> list[Band]:
    """Detect bands: lane profile method first, blob method as a fallback."""
    bands = find_bands_lanes(gray, min_w, min_h)
    if len(bands) >= 2:
        return bands
    return find_bands_blobs(gray, min_w, min_h, max_frac)


def find_bands_lanes(gray: np.ndarray, min_w: int = 6, min_h: int = 2) -> list[Band]:
    """Gel analyzer style: lanes from the column profile, bands from row peaks."""
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks, peak_widths
    h, w = gray.shape
    if h < 10 or w < 20:
        return []
    d = cv2.GaussianBlur(darkness_map(gray), (3, 3), 0)
    noise = float(np.median(np.abs(d - np.median(d)))) * 1.4826 + 1e-3
    floor = max(8.0, 4.0 * noise)
    # Lane centers: peaks of the column profile of the darkest rows.
    colp = np.percentile(d, 90, axis=0)
    colp = gaussian_filter1d(colp, max(1.0, w / 200))
    if colp.max() < floor:
        return []
    min_lane = max(6, w // 40)
    peaks, _ = find_peaks(colp, height=floor * 0.6, distance=min_lane,
                          prominence=0.03 * (colp.max() - colp.min()) + 1e-6)
    if len(peaks) == 0:
        return []
    # Lane boundaries: midpoints (valley minima) between neighbouring peaks.
    bounds = [0]
    for a, b in zip(peaks[:-1], peaks[1:]):
        bounds.append(int(a + np.argmin(colp[a:b + 1])))
    bounds.append(w)
    bands: list[Band] = []
    for li, pc in enumerate(peaks):
        lx0, lx1 = bounds[li], bounds[li + 1]
        # Central part of the lane avoids neighbour bleed.
        cw = max(2, int((lx1 - lx0) * 0.3))
        rowp = d[:, max(lx0, pc - cw):min(lx1, pc + cw + 1)].mean(axis=1)
        rowp = gaussian_filter1d(rowp, 0.8)
        rp, props = find_peaks(rowp, height=floor, prominence=max(floor * 0.6, 0.08 * rowp.max()),
                               distance=3)
        if len(rp) == 0:
            continue
        widths = peak_widths(rowp, rp, rel_height=0.5)
        for k, r in enumerate(rp):
            lo = 0 if k == 0 else int(rp[k - 1] + np.argmin(rowp[rp[k - 1]:r + 1]))
            hi = h if k == len(rp) - 1 else int(r + np.argmin(rowp[r:rp[k + 1] + 1]))
            y0 = max(lo, int(np.floor(widths[2][k]))); y1 = min(hi, int(np.ceil(widths[3][k])) + 1)
            y0, y1 = max(0, y0), min(h, max(y1, y0 + min_h))
            if (y0 <= 1 or y1 >= h - 1) and (y1 - y0) <= 4:
                continue  # frame line at the panel edge
            # Horizontal extent: columns in the lane where this band row is dark.
            strip = d[y0:y1, lx0:lx1].mean(axis=0)
            on = strip > max(floor * 0.7, 0.35 * strip.max())
            # Keep the contiguous run that contains the lane center.
            c = min(max(pc - lx0, 0), len(on) - 1)
            if not on[c]:
                idx = np.nonzero(on)[0]
                if idx.size == 0:
                    continue
                c = int(idx[np.argmin(np.abs(idx - c))])
            a = c
            while a > 0 and on[a - 1]:
                a -= 1
            b = c
            while b < len(on) - 1 and on[b + 1]:
                b += 1
            x0, x1 = lx0 + a, lx0 + b + 1
            bw, bh = x1 - x0, y1 - y0
            if bw < min_w or bh < min_h or bw > 0.5 * w or bh > 0.6 * h:
                continue
            if bw < 0.8 * bh:
                continue
            bands.append(Band(len(bands), x0, y0, bw, bh, float(d[y0:y1, x0:x1].mean())))
    return bands


def find_bands_blobs(gray: np.ndarray, min_w: int = 6, min_h: int = 2,
                     max_frac: float = 0.6) -> list[Band]:
    """Blob based band detection (fallback for panels without clear lanes)."""
    h, w = gray.shape
    dk = darkness_map(gray)
    if dk.max() < 8:
        return []
    d = cv2.GaussianBlur(dk, (3, 3), 0)
    # Threshold relative to the noise floor, not to the strongest band, so a
    # saturated band does not hide faint ones.
    noise = float(np.median(np.abs(d - np.median(d)))) * 1.4826 + 1e-3
    thr = max(8.0, 4.0 * noise, 0.10 * float(np.percentile(d, 99.5)))
    mask = (d > thr).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((1, 3), np.uint8))
    mask = _split_lanes_and_rows(mask, d)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
    bands: list[Band] = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if bw < min_w or bh < min_h:
            continue
        if bw > 0.5 * w or (bh > max_frac * h and h > 30):
            continue  # a frame line or border, not a band
        if bw < 0.8 * bh:  # bands are wider than tall (allow mild blobs)
            continue
        if area < 0.35 * bw * bh:
            continue
        region = dk[y:y + bh, x:x + bw][lab[y:y + bh, x:x + bw] == i]
        bands.append(Band(len(bands), int(x), int(y), int(bw), int(bh), float(region.mean())))
    return bands


def _valleys(profile: np.ndarray, rel_prom: float = 0.12, min_dist: int = 4) -> np.ndarray:
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks
    if profile.size < 8 or profile.max() <= 0:
        return np.array([], int)
    p = gaussian_filter1d(profile.astype(np.float64), 1.2)
    rng = p.max() - p.min()
    idx, _ = find_peaks(-p, prominence=rel_prom * rng + 1e-6, distance=min_dist)
    return idx


def _split_lanes_and_rows(mask: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Cut merged blobs at darkness valleys between lanes, then between rows."""
    out = mask.copy()
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        if w < 12 and h < 12:
            continue
        sub = (lab[y:y + h, x:x + w] == i)
        dk = d[y:y + h, x:x + w] * sub
        # Lanes: valleys in the column profile.
        if w >= 2 * max(6, h // 2):
            for c in _valleys(dk.mean(axis=0), 0.08, max(4, h // 4)):
                out[y:y + h, x + c] = 0
        # Rows: valleys in the row profile, computed per lane chunk.
        sub2 = out[y:y + h, x:x + w] & sub
        m2, lab2, st2, _ = cv2.connectedComponentsWithStats(sub2.astype(np.uint8), connectivity=4)
        for j in range(1, m2):
            xx, yy, ww, hh, _ = st2[j]
            if hh < 7:
                continue
            col = (d[y + yy:y + yy + hh, x + xx:x + xx + ww] * (lab2[yy:yy + hh, xx:xx + ww] == j)).mean(axis=1)
            for r in _valleys(col, 0.1, 3):
                out[y + yy + r, x + xx:x + xx + ww] = 0
    return out


def lanes_from_bands(bands: list[Band], width: int) -> list[tuple[int, int]]:
    """Group band x ranges into lanes by merging overlapping column spans."""
    spans = sorted((b.x, b.x + b.w) for b in bands)
    lanes: list[list[int]] = []
    for a, b in spans:
        if lanes and a <= lanes[-1][1] - 0.3 * (b - a):
            lanes[-1][1] = max(lanes[-1][1], b)
        else:
            lanes.append([a, b])
    return [(int(a), int(b)) for a, b in lanes]


def blot_likeness(rgb: np.ndarray, gray: np.ndarray) -> float:
    """Heuristic 0..1 score: how much does this panel look like a blot?

    Blots are near grayscale, have several band shaped blobs, and those blobs
    line up in rows.
    """
    h, w = gray.shape
    if h < 20 or w < 40:
        return 0.0
    hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)
    sat = float(np.mean(hsv[..., 1])) / 255.0
    color_pen = max(0.0, 1.0 - 4.0 * sat)
    bands = find_bands(gray)
    if len(bands) < 2:
        return 0.0
    # Row alignment: how many bands share a y center with another band.
    ys = np.array([b.y + b.h / 2 for b in bands])
    hs = np.array([b.h for b in bands])
    aligned = sum(1 for i in range(len(ys))
                  if np.sum(np.abs(ys - ys[i]) < np.maximum(3, hs[i] * 0.6)) >= 2)
    align = aligned / len(bands)
    count = min(1.0, len(bands) / 6.0)
    flat_aspect = float(np.mean([min(1.0, b.w / (3.0 * b.h)) for b in bands]))
    return float(color_pen * (0.45 * align + 0.35 * count + 0.2 * flat_aspect))
