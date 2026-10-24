"""Evaluate frozen production v2 CNN on the SAME expanded benchmark as CNN v3.

Run AFTER build_benchmark and BEFORE train_cnn replaces models/splice_cnn.npz.
Do not use test labels to choose hyperparameters. Both calibration and test
are held-out by paper. Outputs results/cnn_v2_expanded_test.json.
"""
from __future__ import annotations

import json
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from btw import cnn
from btw.imgio import load_rgb, to_gray

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "data" / "benchmark"


def score(rows, params, meta):
    result = []
    rule = str(meta.get("rule", "top"))
    space = str(meta.get("space", "prob"))
    for idx, row in enumerate(rows):
        im = load_rgb(BENCH / "images" / f"{row['id']}.png")
        X, _, band, _ = cnn.extract(to_gray(im))
        z = cnn.predict(params, X, batch=128, logits=(space == "logit"))
        if space == "logit":
            score = cnn.relative(z, band, rule)[1]
        else:
            use = z[band] if band.sum() >= 3 else z
            val = float(np.sort(use)[-min(3, len(use)):].mean()) if len(use) else 0.
            if rule == "outlier" and len(use):
                val = float(np.clip(val - np.median(use), 0, 1))
            score = val
        result.append(score)
        if (idx + 1) % 500 == 0:
            print("OLD MODEL", idx + 1, "/", len(rows), flush=True)
    return np.asarray(result, dtype=float)


def matrix(rows, values, threshold):
    arr = np.asarray(values)
    y = np.asarray([r["label"] for r in rows])
    methods = defaultdict(list)
    for row, sc in zip(rows, arr):
        methods[row["method"]].append(sc > threshold)
    return {
        "n": len(rows),
        "auroc": round(float(roc_auc_score(y, arr)), 4),
        "clean_false_flag_rate": round(float((arr[y == 0] > threshold).mean()), 4),
        "detect_by_method": {k: round(float(np.mean(v)), 4) for k, v in methods.items()},
        "threshold": round(float(threshold), 8),
    }


def main():
    index = json.loads((BENCH / "index.json").read_text())
    tr = [r for r in index if r["split"] == "train"]
    paper_ids = {r["pmcid"] for r in tr}
    calib_papers = {p for p in paper_ids if zlib.crc32(p.encode()) % 5 == 0}
    cal = [r for r in tr if r["pmcid"] in calib_papers]
    test = [r for r in index if r["split"] == "test"]
    params, meta = cnn.load()
    print("OLD v2 model meta:", {k:str(v) for k,v in meta.items()},flush=True)
    sc_cal = score(cal, params, meta)
    y_cal = np.array([r["label"] for r in cal])
    recalibrated = float(np.quantile(sc_cal[y_cal == 0], 0.95))
    sc_test = score(test, params, meta)
    out = {
        "model": "frozen production CNN v2",
        "note": "Same rebuilt expanded benchmark and held-out test papers as pending CNN v3.",
        "n_calibration_papers": len(calib_papers),
        "n_test_papers": len(set(r["pmcid"] for r in test)),
        "saved_v2_operating_point": matrix(test, sc_test, float(meta.get("threshold", 0.5))),
        "recalibrated_on_expanded_clean_calibration_papers": matrix(test, sc_test, recalibrated),
    }
    dst = ROOT / "results" / "cnn_v2_expanded_test.json"
    dst.write_text(json.dumps(out, indent=2))
    print("RESULT v2 baseline", json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
