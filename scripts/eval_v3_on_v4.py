"""Compare FROZEN production v3 and newly trained v4 on the exact same test papers.

Calibrate a v3 research cutoff using CLEAN images from the same paper-disjoint
calibration subset used for v4. Neither model sees TEST labels when training,
choosing thresholds, or choosing a policy. This is still a synthetic-edit
benchmark and is not a test of real-world forgery detection.
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
BENCH = ROOT / "data/benchmark"
RESULTS = ROOT / "results"


def evaluate_v3(rows, params, meta, log_prefix):
    out = []
    rule = meta.get("rule", "top")
    for i, row in enumerate(rows):
        gray = to_gray(load_rgb(BENCH / "images" / (row["id"]+".png")))
        x, _, bands, _ = cnn.extract(gray)
        logits = cnn.predict(params, x, batch=128, logits=True)
        out.append(cnn.relative(logits, bands, rule)[1])
        if (i+1) % 500 == 0:
            print(f"{log_prefix} {i+1}/{len(rows)}", flush=True)
    return np.asarray(out)


def summarize(rows, scores, thr):
    labels = np.asarray([int(r["label"]) for r in rows])
    det = scores > thr
    groups = defaultdict(list)
    per_paper = defaultdict(list)
    for r, d in zip(rows, det):
        groups[r["method"]].append(bool(d))
        if not r["label"]:
            per_paper[r["pmcid"]].append(bool(d))
    resized = np.asarray(["resize_50" in r.get("publish", "") for r in rows], bool)
    is_ai = np.asarray([r["method"] == "ai_splice" for r in rows], bool)
    ai50 = is_ai & resized
    return {
        "n_test_images": len(rows),
        "n_test_papers": len({r["pmcid"] for r in rows}),
        "auroc": round(float(roc_auc_score(labels, scores)), 4),
        "threshold": round(float(thr), 6),
        "clean_false_flag_fraction": round(float(det[labels==0].mean()), 4),
        "clean_papers_with_any_flags_fraction": round(float(np.mean([any(x) for x in per_paper.values()])), 4),
        "detect_by_method": {k: round(float(np.mean(v)), 4) for k, v in groups.items()},
        "ai_splice_resize_50_recall": round(float(det[ai50].mean()),4) if ai50.any() else None,
        "ai_splice_resize_50_n": int(ai50.sum()),
    }


def main():
    rows = json.loads((BENCH/"index.json").read_text())
    train = [r for r in rows if r["split"] == "train"]
    papers = {r["pmcid"] for r in train}
    calib_papers = {p for p in papers if zlib.crc32(p.encode()) % 5 == 0}
    cal = [r for r in train if r["pmcid"] in calib_papers and r["label"] == 0]
    test = [r for r in rows if r["split"] == "test"]
    v4 = json.loads((RESULTS/"cnn_v4_metrics.json").read_text())
    frozen, meta = cnn.load(ROOT/"models/splice_cnn_v3.npz")
    if meta.get("space") != "logit":
        raise AssertionError("V3 expected raw-logit stored checkpoint")
    print(f"V3 BASELINE: {len(cal)} clean calibration images, {len(test)} test images",flush=True)
    s_cal = evaluate_v3(cal, frozen, meta, "V3 CAL")
    thr = float(np.quantile(s_cal, .95))
    s_test = evaluate_v3(test, frozen, meta, "V3 TEST")
    v3 = summarize(test, s_test, thr)
    v4_metrics = {
        "n_test_images": v4["n_test_images"],
        "n_test_papers": v4["n_test_papers"],
        "auroc": v4["image_auroc"],
        "threshold": v4["threshold"],
        "clean_false_flag_fraction": v4["false_flag_rate_clean"],
        "detect_by_method": v4["detect_rate_by_method"],
    }
    same = (v3["n_test_images"] == v4["n_test_images"] and v3["n_test_papers"] == v4["n_test_papers"])
    if not same:
        raise AssertionError("Cannot compare V3 and V4: test partitions differ")
    improved = (v4_metrics["auroc"] >= v3["auroc"] + 0.01
                and v4_metrics["detect_by_method"].get("ai_splice", 0) > v3["detect_by_method"].get("ai_splice", 0)
                and v4_metrics["clean_false_flag_fraction"] <= v3["clean_false_flag_fraction"] + .015)
    comparison = {
        "benchmark": "same new v4 held-out-paper benchmark; both thresholds calibrated using same separate clean-paper subset",
        "v3_frozen_calibrated": v3,
        "v4_new": v4_metrics,
        "delta_auroc": round(v4_metrics["auroc"] - v3["auroc"], 4),
        "delta_ai_splice_recall": round(v4_metrics["detect_by_method"].get("ai_splice", 0)
                                        - v3["detect_by_method"].get("ai_splice", 0), 4),
        "delta_clean_false_flags": round(v4_metrics["clean_false_flag_fraction"]
                                          - v3["clean_false_flag_fraction"], 4),
        "passes_predeclared_improvement_screen": bool(improved),
        "next_step": "Validate on whole untouched real papers and generator-disjoint AI edits; do NOT auto-promote.",
        "limitations": [
            "Forgeries are synthetic; samples from the two Gemini AI donor blots still lack generator diversity.",
            "v4 metrics are used after one fixed 5-epoch training job; not tuned on heldout test.",
            "Source papers must be checked for license and paper-disjointness, and the older v3-trained set is only partially reproduced from public Europe PMC.",
            "A change in AUROC/recall on this expanded benchmark is not proof of better real-world integrity screening.",
        ],
    }
    (RESULTS/"cnn_v4_comparison.json").write_text(json.dumps(comparison, indent=2))
    print("V4_COMPARISON", json.dumps(comparison, indent=2), flush=True)


if __name__=="__main__":
    main()
