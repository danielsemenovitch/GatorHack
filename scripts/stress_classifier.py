"""Stress test the committed CNN and a scale-aware logistic meta-classifier.

Requires only committed CC-BY panels and Gemini donor sources, not the generated
benchmark. Source-paper split is identical to build_benchmark.py. The frozen CNN
is NEVER retrained or selected using test papers. New logistic features train on
training papers; clean calibration papers set its 5% flag threshold.

Use: python -m scripts.stress_classifier
Writes results/stress_classifier.json. No changes to production models.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from btw import cnn
from btw.imgio import load_rgb, to_gray
from btw.publish import TRANSFORMS, jpeg, resize
from btw.sources import ai_panels
from btw.splice import forge

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results" / "stress_classifier.json"
TRANS = {
    "none": lambda x: x,
    "resize_75": lambda x: resize(x, .75),
    "resize_50": lambda x: resize(x, .50),
    "resize_33": lambda x: resize(x, .33),
    "resize_50_then_upscale": lambda x: resize(x, .50).resize(x.size, Image.Resampling.LANCZOS),
    "jpeg_q85": TRANSFORMS["jpeg_q85"],
    "jpeg_q60": TRANSFORMS["jpeg_q60"],
    "contrast": TRANSFORMS["contrast"],
    "pdf_roundtrip": TRANSFORMS["pdf_roundtrip"],
}


def group(pmcid):
    value = int(hashlib.md5(pmcid.encode()).hexdigest(), 16) % 100
    if value < 30:
        return "test"
    if zlib.crc32(pmcid.encode()) % 5 == 0:
        return "calib"
    return "train"


def pairs():
    source = {p["panel"]: p for p in json.loads((DATA / "real_panels" / "panels.json").read_text())}
    kept = sorted(json.loads((DATA / "real_panels" / "keep.json").read_text()))
    donors = ai_panels()
    result = defaultdict(list)
    attempts = defaultdict(int)
    for filename in kept:
        if filename not in source:
            continue
        meta = source[filename]
        pmcid = meta.get("pmcid") or meta["figure"]
        clean = load_rgb(DATA / "real_panels" / filename)
        # Dedicated deterministic RNG per panel; generation doesn't depend on file ordering.
        seed = int(hashlib.sha256(filename.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        forgery = None
        for attempt in range(7):
            attempts[group(pmcid)] += 1
            forgery = forge(clean, "ai_splice", rng, ai_panels=donors)
            if forgery is not None:
                break
        if forgery is None:
            continue
        result[group(pmcid)].append((filename, pmcid, clean, forgery.image))
    print("usable pairs:", {k: len(v) for k, v in result.items()}, flush=True)
    return result


def patch_summary(im, params, meta):
    gray = to_gray(im)
    x, _, band, _ = cnn.extract(gray)
    if not len(x):
        return None, None
    # Small fixed JAX batch avoids the original 1024-patch padding overhead.
    z = cnn.predict(params, x, batch=32, logits=True).astype(np.float64)
    p = z[band] if band.sum() >= 3 else z
    med = float(np.median(p))
    mad = float(np.median(np.abs(p - med)) * 1.4826 + .25)
    top = np.sort(p)[-min(3, len(p)):]
    avg = float(top.mean())
    top1 = float(p.max())
    q25, q75, q90 = np.percentile(p, [25, 75, 90])
    feat = np.array([avg, top1, med, mad, avg-med, (avg-med)/mad,
                     top1-med, (top1-med)/mad, float(p.mean()),
                     float(p.std()), float(q75-q25), float(q90-med),
                     float((p > med+mad).mean())], dtype=np.float64)
    # Exact legacy v2 scoring, including the model's original flag threshold.
    space, rule = str(meta.get("space", "prob")), str(meta.get("rule", "top"))
    if space == "logit":
        baseline = cnn.relative(z, band, rule)[1]
    else:
        probs = 1 / (1 + np.exp(-np.clip(z, -80, 80)))
        b = probs[band] if band.sum() >= 3 else probs
        baseline = float(np.sort(b)[-min(3, len(b)):].mean())
        if rule == "outlier":
            baseline = float(np.clip(baseline - np.median(b), 0, 1))
    return feat, baseline


def interval(hits, total):
    if total == 0:
        return [None, None]
    z = 1.96
    p = hits / total
    mid = (p + z*z/(2*total)) / (1+z*z/total)
    rad = z * np.sqrt(p*(1-p)/total + z*z/(4*total*total)) / (1+z*z/total)
    return [round(float(mid-rad), 3), round(float(mid+rad), 3)]


def score_pairs(data, transforms, params, meta):
    output = {}
    for t in transforms:
        func = TRANS[t]
        records = []
        for filename, pmcid, clean, fake in data:
            neg, rawneg = patch_summary(func(clean), params, meta)
            pos, rawpos = patch_summary(func(fake), params, meta)
            if neg is None or pos is None:
                continue
            records.append((filename, pmcid, neg, pos, rawneg, rawpos))
        output[t] = records
        print("  scored", t, len(records), flush=True)
    return output


def fit_meta(scored):
    X, y = [], []
    for t in ("none", "resize_75", "resize_50", "jpeg_q60"):
        for _, _, neg, pos, _, _ in scored[t]:
            X.extend([neg, pos])
            y.extend([0, 1])
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, class_weight="balanced", max_iter=1500))
    model.fit(np.asarray(X), np.asarray(y))
    return model, len(y)


def as_metrics(y0, y1, threshold):
    clean = np.asarray(y0, float)
    fake = np.asarray(y1, float)
    n = len(clean)
    fp = int((clean > threshold).sum())
    hit = int((fake > threshold).sum())
    auc = round(float(roc_auc_score(np.r_[np.zeros(n), np.ones(n)],
                                    np.r_[clean, fake])), 3) if n else None
    return {"n_pairs": n, "detected": hit, "false_flags": fp,
            "detection_rate": round(hit/n, 3) if n else None,
            "false_flag_rate": round(fp/n, 3) if n else None,
            "auroc": auc,
            "detection_rate_95ci_wilson": interval(hit, n),
            "false_flag_rate_95ci_wilson": interval(fp, n)}


def main():
    started = time.time()
    params, meta = cnn.load()
    groups = pairs()
    if not groups["train"] or not groups["calib"] or not groups["test"]:
        raise RuntimeError("Paper-disjoint train/calibration/test groups needed")
    print("TRAIN features", flush=True)
    tr = score_pairs(groups["train"], ("none", "resize_75", "resize_50", "jpeg_q60"), params, meta)
    model, n_train = fit_meta(tr)

    # Use clean held-out CALIBRATION papers at BOTH native and half resolution.
    # Test labels and test performance never affect selection or threshold.
    print("CALIBRATION", flush=True)
    ca = score_pairs(groups["calib"], ("none", "resize_50"), params, meta)
    clean_cal = np.vstack([r[2] for t in ca for r in ca[t]])
    prob_cal = model.predict_proba(clean_cal)[:, 1]
    meta_threshold = float(np.quantile(prob_cal, .95))

    print("TEST stress matrix", flush=True)
    te = score_pairs(groups["test"], tuple(TRANS), params, meta)
    baseline_threshold = float(meta.get("threshold", 0.5))
    rows = []
    for t, rr in te.items():
        negb = [r[4] for r in rr]
        posb = [r[5] for r in rr]
        negp = model.predict_proba(np.vstack([r[2] for r in rr]))[:, 1]
        posp = model.predict_proba(np.vstack([r[3] for r in rr]))[:, 1]
        rows.append({"transform": t,
                     "current_cnn": as_metrics(negb, posb, baseline_threshold),
                     "scale_augmented_logistic_head": as_metrics(negp, posp, meta_threshold)})
        print(json.dumps(rows[-1]), flush=True)

    out = {
        "experiment": "frozen original CNN + scale-augmented logistic head; article-disjoint test",
        "seed": "SHA256(panel filename), first successful of up to 7 forgeries",
        "original_cnn_model": "models/splice_cnn.npz, UNCHANGED",
        "cnn_model_score_rule": str(meta.get("rule", "top")),
        "cnn_model_score_space": str(meta.get("space", "prob")),
        "cnn_model_threshold": baseline_threshold,
        "new_logistic_threshold": round(meta_threshold, 8),
        "new_logistic_training_samples": n_train,
        "pairs_by_paper_split": {k: len(v) for k, v in groups.items()},
        "papers_by_split": {k: len(set(pmcid for _, pmcid, _, _ in v)) for k, v in groups.items()},
        "training_transforms": ["none", "resize_75", "resize_50", "jpeg_q60"],
        "calibration_clean_transforms": ["none", "resize_50"],
        "rows": rows,
        "seconds": round(time.time()-started, 1),
        "limitations": [
            "Synthetic AI splices from the SAME TWO Gemini donor sources as the original benchmark; this is not generator-held-out generalization.",
            "Only one successful deterministic synthetic splice per panel. Some generated masks may be too small after resize to remain semantically detectable.",
            "Paper-disjoint test, but two images from one paper are correlated; Wilson intervals ignore within-paper clustering.",
            "Trained logistic head, not a retrained CNN. New model has not been deployed to app.",
            "Original CNN was trained on other forgeries and these same original training papers; test papers remain unseen.",
            "Original CNN threshold and new logistic threshold use different calibration protocols; compare observed false-flag rates alongside detection.",
            "Not evidence that a particular real paper is manipulated.",
        ]
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print("Saved", OUT, flush=True)


if __name__ == "__main__":
    main()
