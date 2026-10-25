"""Split a figure into panels and keep the ones that look like blots.

Blot strips in published figures are usually solid rectangles of non white
pixels (the membrane background) sitting on a white page, so we look for
connected regions that fill most of their bounding box, then score each one
with a blot likeness heuristic.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from .bands import blot_likeness
from .imgio import normalize_polarity, to_gray


@dataclass
class Panel:
    id: int
    x: int
    y: int
    w: int
    h: int
    blot_score: float

    def box(self):
        return self.x, self.y, self.x + self.w, self.y + self.h

    def crop(self, im: Image.Image) -> Image.Image:
        return im.crop(self.box())

    def to_dict(self) -> dict:
        return {"id": self.id, "x": self.x, "y": self.y, "w": self.w, "h": self.h,
                "blot_score": round(self.blot_score, 3)}


FRAME_DARK = 110      # frame lines are near black on any figure
FRAME_FILL = 0.85     # a frame edge row/column is at least 85% dark
MAX_FRAME = 14        # thickest frame line we trim, in pixels (2,000 px figures draw 4 to 8)


def _edge_run(dark: np.ndarray, axis_lines) -> int:
    """Number of consecutive frame-like lines from one edge (0 if none)."""
    n = 0
    for line in axis_lines:
        if line.mean() < FRAME_FILL:
            break
        n += 1
        if n > MAX_FRAME:
            return 0  # a thick dark run is content (a saturated band), not a frame
    return n


def trim_frame(gray: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> tuple[int, int, int, int]:
    """Shrink a box past any published frame line on each side, at any resolution.

    The old fixed 2 px trim left 4 to 8 px black frames inside crops of high
    resolution PDF figures; the CNN then scored frame and band junctions.
    """
    sub = gray[y0:y1, x0:x1]
    dark = sub < FRAME_DARK
    h, w = dark.shape
    if h < 8 or w < 8:
        return x0, y0, x1, y1
    # Dark background panels (gels, inverted fluorescence): a black frame is
    # indistinguishable from the background there, and trimming would cut
    # into real lanes. Leave them to the fixed 2 px margin.
    if normalize_polarity(sub)[1]:
        return x0, y0, x1, y1
    top = _edge_run(dark, (dark[i] for i in range(h)))
    bottom = _edge_run(dark, (dark[h - 1 - i] for i in range(h)))
    left = _edge_run(dark, (dark[:, i] for i in range(w)))
    right = _edge_run(dark, (dark[:, w - 1 - i] for i in range(w)))
    # Frames of 1 to 2 px (web resolution figures, like every training panel)
    # are left to the caller's fixed 2 px margin, so those crops stay exactly
    # as the CNN saw them in training. Thicker frames (high resolution PDF
    # figures) are trimmed fully, plus one pixel of anti-aliased halo.
    pad = lambda n: n + 1 if n > 2 else 0
    nx0, ny0, nx1, ny1 = x0 + pad(left), y0 + pad(top), x1 - pad(right), y1 - pad(bottom)
    if nx1 - nx0 < 16 or ny1 - ny0 < 8:
        return x0, y0, x1, y1
    return nx0, ny0, nx1, ny1


def find_frames(gray: np.ndarray, min_w: int = 40, min_h: int = 15) -> list[tuple[int, int, int, int]]:
    """Rectangles drawn as dark outlines (published blot frames), as (x0, y0, x1, y1).

    A frame is a box whose four edges are each mostly dark within a few pixels
    of the edge, and whose interior is mostly NOT dark. Nested duplicates
    (outer and inner contour of the same line) are merged.
    """
    dark = (gray < FRAME_DARK).astype(np.uint8)
    contours, _ = cv2.findContours(dark, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < min_w or h < min_h:
            continue
        sub = dark[y:y + h, x:x + w].astype(bool)
        k = min(4, h // 4, w // 4)
        if k < 1:
            continue
        edges = (sub[:k].mean(axis=1).max(), sub[-k:].mean(axis=1).max(),
                 sub[:, :k].mean(axis=0).max(), sub[:, -k:].mean(axis=0).max())
        if min(edges) < FRAME_FILL:
            continue
        inner = sub[k:h - k, k:w - k]
        if inner.size == 0 or inner.mean() > 0.6:
            continue
        boxes.append((x, y, x + w, y + h))
    boxes.sort(key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))
    out: list[tuple[int, int, int, int]] = []
    for b in boxes:  # drop a box that is the same frame as a bigger one (its inner contour)
        if any(b[0] >= o[0] - 1 and b[1] >= o[1] - 1 and b[2] <= o[2] + 1 and b[3] <= o[3] + 1
               and (b[2] - b[0]) >= (o[2] - o[0]) - 2 * MAX_FRAME - 2
               and (b[3] - b[1]) >= (o[3] - o[1]) - 2 * MAX_FRAME - 2 for o in out):
            continue
        out.append(b)
    return out


def _band_in_strip(box, frame) -> bool:
    """True for a blob that is one band of a framed blot strip.

    A strip frame is elongated (at least 2.5 times wider than tall); a blob
    covering under 60% of its width is a band or lane group inside it. Frames
    drawn around a whole figure are not elongated, so strips inside them stay.
    """
    if not _inside(box, frame):
        return False
    bw, fw, fh = box[2] - box[0], frame[2] - frame[0], frame[3] - frame[1]
    return fw >= 2.5 * fh and bw < 0.6 * fw


def _inside(box, frame, slack: int = 2) -> bool:
    return (box[0] >= frame[0] - slack and box[1] >= frame[1] - slack
            and box[2] <= frame[2] + slack and box[3] <= frame[3] + slack)


def find_panels(im: Image.Image, min_area: int = 1200, min_score: float = 0.3,
                white: int = 246) -> list[Panel]:
    rgb = np.asarray(im.convert("RGB"))
    gray = to_gray(im)
    H, W = gray.shape
    mask = (gray < white).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
    frames = find_frames(gray)
    boxes: list[tuple[int, int, int, int]] = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if w * h < min_area or w < 40 or h < 15:
            continue
        if area < 0.7 * w * h:
            continue
        box = (int(x), int(y), int(x + w), int(y + h))
        # A framed strip with arrows, labels or markers touching its frame is
        # one connected region; the frame is the real panel boundary.
        area = (box[2] - box[0]) * (box[3] - box[1])
        inner = [f for f in frames if _inside(f, box) and (f[2] - f[0]) * (f[3] - f[1]) >= 0.6 * area]
        if inner:
            f = max(inner, key=lambda f: (f[2] - f[0]) * (f[3] - f[1]))
            # Only when something real is attached (more than a few pixels of
            # anti-aliasing outside the frame); otherwise keep the component
            # box exactly as the training panels were cut.
            if max(f[0] - box[0], f[1] - box[1], box[2] - f[2], box[3] - f[3]) > 6:
                box = f
        # A blob INSIDE a framed strip that spans most of the strip's height is
        # one band on a white membrane, not a panel of its own (Oncotarget p62).
        # Strips inside a frame drawn around a whole figure are kept: they are
        # short relative to that frame.
        if any(_band_in_strip(box, f) for f in frames):
            continue
        boxes.append(box)
    # Framed strips whose membrane is white enough that the filled-region test
    # above missed them (their only non-white pixels are the frame and bands).
    # A frame that already contains a kept panel is a figure border, not a strip.
    for f in frames:
        fw, fh = f[2] - f[0], f[3] - f[1]
        if fw * fh < min_area or fw < 1.2 * fh:
            continue
        if any(_inside(b, f, slack=MAX_FRAME) or _inside(f, b) for b in boxes) or any(
                _inside(b, f, slack=MAX_FRAME) and b != f for b in frames if (b[2] - b[0]) * (b[3] - b[1]) >= min_area
                and (b[2] - b[0]) >= 1.2 * (b[3] - b[1]) and (b[3] - b[1]) < 0.8 * fh):
            continue
        boxes.append(f)
    out: list[Panel] = []
    for box in sorted(boxes, key=lambda b: (b[1] // 20, b[0])):
        # Trim published frame lines (any thickness up to MAX_FRAME), then the
        # old 2 px safety margin for anti-aliased edges without a frame.
        tx0, ty0, tx1, ty1 = trim_frame(gray, *box)
        # never less than the old 2 px margin on any side
        x0, y0, x1, y1 = max(tx0, box[0] + 2), max(ty0, box[1] + 2), min(tx1, box[2] - 2), min(ty1, box[3] - 2)
        w, h = x1 - x0, y1 - y0
        if w < 40 or h < 15:
            continue
        sub_rgb = rgb[y0:y1, x0:x1]
        sub_gray = gray[y0:y1, x0:x1]
        s = blot_likeness(sub_rgb, sub_gray)
        if s >= min_score:
            out.append(Panel(len(out), int(x0), int(y0), int(w), int(h), s))
    # If the whole image is one blot (no white page around it), treat it as a panel.
    if not out:
        s = blot_likeness(rgb, gray)
        if s >= min_score:
            out.append(Panel(0, 0, 0, W, H, s))
    return out
