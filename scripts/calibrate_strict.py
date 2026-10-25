"""Choose the optional strict alert line from calibration papers, then measure it on test papers.

The Standard line (display index 0.50) was set so 5% of clean panels from the
calibration papers are flagged. The old strict line (0.90) caught only about 4%
of pasted AI bands, which is not a usable setting. This script picks the strict
line from calibration papers only (training papers held out from fitting, never
test papers): the lowest line on a fixed grid that flags at most --target of the
clean calibration panels. It then reports what every grid line catches on the
held-out test papers of the local benchmark.

Run:  python -m scripts.calibrate_strict [--target 0.01]
Writes results/strict_line.json.
"""
from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import numpy as np

from btw import cnn
from btw.detector import CnnDetector
from btw.imgio import load_rgb, to_gray

ROOT = Path(__file__).resolve().parents[1]
EDITS = ("ai_splice", "cross_splice", "copy_move", "intensify", "erase")


def score_rows(det, rows):
    out = []
    for r in rows:
        X, _, band, _ = cnn.extract(to_gray(load_rgb(ROOT / "data/benchmark/images" / f"{r['id']}.png")))
        if not len(X):
            out.append(0.0)
            continue
        _, raw = cnn.relative(cnn.predict(det.params, X, logits=True), band, det.score_method)
        out.append(float(det._display(raw)))
    return np.array(out)


def rates(scores, methods, t):
    res = {"clean_flagged": round(float((scores[methods == "clean"] > t).mean()), 3)}
    for m in EDITS:
        if (methods == m).any():
            res[f"{m}_caught"] = round(float((scores[methods == m] > t).mean()), 3)
    edited = np.isin(methods, EDITS)
    res["any_edit_caught"] = round(float((scores[edited] > t).mean()), 3)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=0.01, help="max clean flag rate on calibration papers")
    ap.add_argument("--model", default="v3", help="v3, or a newer models/splice_cnn_vN.npz")
    args = ap.parse_args()

    det = CnnDetector.load(args.model)
    ix = json.loads((ROOT / "data/benchmark/index.json").read_text())
    # Same calibration split as scripts/train_cnn.py.
    calib = [r for r in ix if r["split"] == "train" and zlib.crc32(r["pmcid"].encode()) % 5 == 0 and r["method"] == "clean"]
    test = [r for r in ix if r["split"] == "test"]

    grid = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9]
    cal_scores = score_rows(det, calib)
    line = next(t for t in grid if (cal_scores > t).mean() <= args.target)
    test_scores = score_rows(det, test)
    methods = np.array([r["method"] for r in test])

    candidates = grid
    out = {
        "model": args.model,
        "rule": f"lowest grid line flagging at most {args.target:.0%} of clean panels on calibration papers",
        "n_calibration_clean": len(calib),
        "n_calibration_papers": len({r["pmcid"] for r in calib}),
        "n_test_images": len(test),
        "n_test_papers": len({r["pmcid"] for r in test}),
        "chosen_line": round(line, 3),
        "calibration_clean_flagged": {f"{t}": round(float((cal_scores > t).mean()), 3) for t in candidates},
        "test": {f"{t}": rates(test_scores, methods, t) for t in candidates},
    }
    name = "strict_line.json" if args.model == "v3" else f"strict_line_{args.model}.json"
    (ROOT / "results" / name).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
