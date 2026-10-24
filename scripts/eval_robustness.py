"""Robustness matrix for our detector, mirroring the SynthID experiment.

For every held out test panel: paste one AI band into it (ai_splice), then
apply ONE publishing step at a time and check whether the saved detector
(and its saved threshold) still flags the panel. The same steps are applied
to the untouched panel to measure false flags.
Writes results/robustness.json.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from btw.detector import Detector
from btw.imgio import load_rgb
from btw.publish import TRANSFORMS
from btw.sources import ai_panels
from btw.splice import forge

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"none": "None (straight after the edit)", "resave": "Re-saved", "resize_50": "Resized to 50%",
          "jpeg_q85": "JPEG quality 85", "jpeg_q60": "JPEG quality 60", "contrast": "Contrast adjusted",
          "pdf_roundtrip": "PDF round trip"}


def main():
    from btw.detector import load_best
    det = load_best()
    rng = random.Random(11)
    ix = json.loads((ROOT / "data/benchmark/index.json").read_text())
    panels = sorted({r["panel"] for r in ix if r["split"] == "test"})
    ai = ai_panels()
    pairs = []
    for pn in panels:
        im = load_rgb(ROOT / "data/real_panels" / pn)
        for _ in range(5):
            f = forge(im, "ai_splice", rng, ai_panels=ai)
            if f is not None:
                pairs.append((im, f.image)); break
    rows = []
    for t, label in LABELS.items():
        tf = TRANSFORMS[t]
        pos = np.array([det.analyze(tf(f)).score > det.threshold for _, f in pairs])
        neg = np.array([det.analyze(tf(im)).score > det.threshold for im, _ in pairs])
        rows.append({"transform": t, "label": label, "detect_rate": round(float(pos.mean()), 3),
                     "false_flag_rate": round(float(neg.mean()), 3)})
        print(rows[-1])
    out = {"n_panels": len(pairs), "threshold": det.threshold, "rows": rows,
           "note": "One AI generated band pasted into each held out real panel, then one publishing step."}
    (ROOT / "results/robustness.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
