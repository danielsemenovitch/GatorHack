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
from .imgio import to_gray


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


def find_panels(im: Image.Image, min_area: int = 1200, min_score: float = 0.3,
                white: int = 246) -> list[Panel]:
    rgb = np.asarray(im.convert("RGB"))
    gray = to_gray(im)
    H, W = gray.shape
    mask = (gray < white).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
    out: list[Panel] = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if w * h < min_area or w < 40 or h < 15:
            continue
        if area < 0.7 * w * h:
            continue
        # Trim a 1 px frame line that many figures draw around blots.
        x0, y0, x1, y1 = x + 2, y + 2, x + w - 2, y + h - 2
        w, h = x1 - x0, y1 - y0
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
