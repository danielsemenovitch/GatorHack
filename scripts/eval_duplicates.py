"""Evaluate the copied band detector on exact pixel copies.

Positives: a band (with a margin of background) copied over another band of
the same held out test panel, optionally flipped. Negatives: the same panels
untouched. Both pass through the same publishing step. The threshold is
fixed from clean panels with no publishing step (5% false flags).
Writes results/duplicates_metrics.json.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

from btw.bands import find_bands
from btw.duplicates import find_duplicates
from btw.imgio import load_rgb, to_gray
from btw.publish import TRANSFORMS

ROOT = Path(__file__).resolve().parents[1]


def make_copy(im: Image.Image, rng: random.Random):
    g = to_gray(im)
    H, W = g.shape
    bands = [b for b in find_bands(g) if b.w >= 14 and b.h >= 5 and b.y > 3 and b.x > 3
             and b.x + b.w < W - 3 and b.y + b.h < H - 3]
    if len(bands) < 2:
        return None
    a = np.asarray(im).copy()
    s = rng.choice(bands)
    t = rng.choice([b for b in bands if b.id != s.id])
    py = max(2, s.h // 2)
    src = a[s.y - py:s.y + s.h + py, s.x - 2:s.x + s.w + 2].copy()
    if rng.random() < 0.3:
        src = src[:, ::-1]
    ty = max(0, min(H - src.shape[0], t.y - py)); tx = max(0, min(W - src.shape[1], t.x - 2))
    a[ty:ty + src.shape[0], tx:tx + src.shape[1]] = src
    return Image.fromarray(a)


def top(im):
    d = find_duplicates([to_gray(im)], threshold=0.2, shape_floor=0.5, min_w=14, min_h=5)
    return d[0]["noise_similarity"] if d else 0.0


def main():
    rng = random.Random(7)
    ix = json.loads((ROOT / "data/benchmark/index.json").read_text())
    panels = sorted({r["panel"] for r in ix if r["split"] == "test"})
    pairs = []
    for pn in panels:
        im = load_rgb(ROOT / "data/real_panels" / pn)
        f = make_copy(im, rng)
        if f is not None:
            pairs.append((im, f))
    conds = {"none": "No publishing step", "jpeg_q85": "JPEG quality 85", "pdf_roundtrip": "PDF round trip",
             "resize_50": "Resized to 50%"}
    scores = {}
    for c in conds:
        tf = TRANSFORMS[c]
        scores[c] = (np.array([top(tf(f)) for _, f in pairs]), np.array([top(tf(im)) for im, _ in pairs]))
    thr = float(np.quantile(scores["none"][1], 0.95))
    rows = []
    for c, label in conds.items():
        pos, neg = scores[c]
        rows.append({"transform": c, "label": label, "detect_rate": round(float((pos > thr).mean()), 3),
                     "false_flag_rate": round(float((neg > thr).mean()), 3)})
    out = {"threshold": round(thr, 3), "n_panels": len(pairs), "rows": rows,
           "note": "Exact pixel copies of one band within a held out test panel."}
    (ROOT / "results/duplicates_metrics.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
