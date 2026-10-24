"""Train and evaluate the splice detector on the benchmark.

Reports, on the held out test papers:
  * image level AUROC and precision / recall at the chosen threshold
  * per forgery method detection rate
  * window (localization) AUROC
  * a held out METHOD experiment: train without ai_splice, test on it
Writes results/detector_metrics.json and models/splice_hgb.joblib.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from btw.detector import Detector, panel_score
from btw.features import FEATURE_NAMES, window_features, window_labels
from btw.imgio import load_rgb, to_gray

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "data" / "benchmark"
RESULTS = ROOT / "results"


def featurize(row):
    img = load_rgb(BENCH / "images" / f"{row['id']}.png")
    mask = np.asarray(Image.open(BENCH / "masks" / f"{row['id']}.png")) > 0
    wins, X = window_features(to_gray(img))
    y = window_labels(wins, mask)
    return row["id"], X, y


def make_model(seed=0):
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                          l2_regularization=1.0, class_weight="balanced",
                                          random_state=seed)


def stack(rows, feats, subsample_neg=0.2, seed=0):
    """All forged windows, ALL real band windows (hard negatives), and a
    sample of background windows."""
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for r in rows:
        X, y = feats[r["id"]]
        band = X[:, 13] > 0.15
        keep = (y == 1) | band | (rng.random(len(y)) < subsample_neg)
        Xs.append(X[keep]); ys.append(y[keep])
    return np.vstack(Xs), np.concatenate(ys)


SCORE_METHOD = "outlier"


def image_scores(model, rows, feats, method=None):
    out = {}
    for r in rows:
        X, _ = feats[r["id"]]
        out[r["id"]] = panel_score(model.predict_proba(X)[:, 1], X, method or SCORE_METHOD)
    return out


def main():
    global SCORE_METHOD
    t0 = time.time()
    index = json.loads((BENCH / "index.json").read_text())
    cache = ROOT / "results" / "feature_cache.joblib"
    import joblib
    if cache.exists() and joblib.load(cache).get("n") == len(index):
        feats = joblib.load(cache)["feats"]
        print("loaded cached features")
    else:
        res = Parallel(n_jobs=2)(delayed(featurize)(r) for r in index)
        feats = {i: (X, y) for i, X, y in res}
        joblib.dump({"n": len(index), "feats": feats}, cache)
    print(f"featurized {len(index)} images in {time.time() - t0:.0f}s")

    train = [r for r in index if r["split"] == "train"]
    test = [r for r in index if r["split"] == "test"]
    Xtr, ytr = stack(train, feats)
    model = make_model().fit(Xtr, ytr)
    # Pick the panel scoring rule on TRAINING images only.
    ytr_img = np.array([r["label"] for r in train])
    cand = {}
    for m in ("top", "outlier"):
        sc = image_scores(model, train, feats, m)
        cand[m] = roc_auc_score(ytr_img, [sc[r["id"]] for r in train])
    SCORE_METHOD = max(cand, key=cand.get)
    print("train AUROC by scoring rule:", {k: round(v, 3) for k, v in cand.items()}, "->", SCORE_METHOD)

    # Threshold chosen on TRAIN images only (no peeking at test): the score at
    # which at most 5% of clean training panels would be flagged.
    tr_scores = image_scores(model, train, feats)
    clean_tr = np.array([tr_scores[r["id"]] for r in train if r["label"] == 0])
    thr = float(np.quantile(clean_tr, 0.95)) if len(clean_tr) else 0.5

    te_scores = image_scores(model, test, feats)
    y_true = np.array([r["label"] for r in test])
    s = np.array([te_scores[r["id"]] for r in test])
    pred = (s > thr).astype(int)
    p, rc, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
    by_method = defaultdict(list)
    for r, sc in zip(test, s):
        by_method[r["method"]].append(sc > thr)
    by_publish = defaultdict(lambda: [0, 0])
    for r, sc in zip(test, s):
        k = r["publish"].split("+")[-1].split("_q")[0]
        if r["label"]:
            by_publish[k][0] += int(sc > thr); by_publish[k][1] += 1
    # Window level localization on forged test images.
    wy, wp = [], []
    for r in test:
        if r["label"]:
            X, y = feats[r["id"]]
            wy.append(y); wp.append(model.predict_proba(X)[:, 1])
    win_auc = roc_auc_score(np.concatenate(wy), np.concatenate(wp)) if wy else None

    # Held out method: train without ai_splice, evaluate ai_splice vs clean.
    ho = {}
    for held in ("ai_splice", "erase"):
        tr_h = [r for r in train if r["method"] != held]
        m_h = make_model().fit(*stack(tr_h, feats))
        sc_tr = image_scores(m_h, tr_h, feats)
        thr_h = float(np.quantile([sc_tr[r["id"]] for r in tr_h if r["label"] == 0], 0.95))
        te_h = [r for r in test if r["method"] in (held, "clean")]
        sc = image_scores(m_h, te_h, feats)
        yt = np.array([r["label"] for r in te_h]); ss = np.array([sc[r["id"]] for r in te_h])
        ho[held] = {"auroc": round(float(roc_auc_score(yt, ss)), 3),
                    "detect_rate": round(float(np.mean(ss[yt == 1] > thr_h)), 3),
                    "false_flag_rate": round(float(np.mean(ss[yt == 0] > thr_h)), 3)}

    imp = {}
    try:
        from sklearn.inspection import permutation_importance
        Xte, yte = stack(test, feats, subsample_neg=0.2)
        pi = permutation_importance(model, Xte, yte, scoring="roc_auc", n_repeats=3, random_state=0)
        imp = dict(sorted(zip(FEATURE_NAMES, pi.importances_mean.round(4).tolist()), key=lambda t: -t[1]))
    except Exception as e:  # noqa
        imp = {"error": str(e)}

    metrics = {
        "n_train_images": len(train), "n_test_images": len(test),
        "n_train_papers": len({r["pmcid"] for r in train}), "n_test_papers": len({r["pmcid"] for r in test}),
        "threshold": round(thr, 4),
        "image_auroc": round(float(roc_auc_score(y_true, s)), 3),
        "precision": round(float(p), 3), "recall": round(float(rc), 3), "f1": round(float(f1), 3),
        "false_flag_rate_clean": round(float(np.mean(s[y_true == 0] > thr)), 3),
        "detect_rate_by_method": {k: round(float(np.mean(v)), 3) for k, v in by_method.items()},
        "detect_rate_by_last_publish_step": {k: round(v[0] / max(1, v[1]), 3) for k, v in by_publish.items()},
        "window_auroc_localization": None if win_auc is None else round(float(win_auc), 3),
        "held_out_method": ho,
        "feature_importance": imp,
        "seconds": round(time.time() - t0),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "detector_metrics.json").write_text(json.dumps(metrics, indent=1))
    # Final model trained on everything in train (already fit) is saved for the app.
    metrics["score_method"] = SCORE_METHOD
    Detector(model, thr, SCORE_METHOD).save(metrics=metrics)
    print(json.dumps({k: v for k, v in metrics.items() if k != "feature_importance"}, indent=1))


if __name__ == "__main__":
    main()
