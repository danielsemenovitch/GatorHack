"""Paper baselines on the frozen V5 benchmark (paper-disjoint test papers).

Subcommands (run from the repo root after scripts.build_v5_benchmark):
  cnn         score frozen V3 and V5 on calibration-clean and test images, save per-image scores
  b0          train an ImageNet-pretrained EfficientNet-B0 on raw 32x32 patches (the recipe of
              Manjunath et al. 2024, arXiv:2408.13786) using the SAME training papers, patch
              positions and labels as V5, then score calibration-clean and test images
  trufor-prep copy calibration-clean and test images into a folder for TruFor (Guillaro et al., CVPR 2023)
  trufor-read collect TruFor whole-image integrity scores from its .npz outputs
  compare     merge every available score file and report paired, paper-clustered statistics

Rules shared by every detector, fixed before any test score is seen:
  * threshold = 95th percentile of that detector's scores on clean images from separate
    calibration papers (CRC32(PMCID) % 5 == 0 among training papers), as for V3 and V5;
  * AUROC and recall are reported for AI splices vs clean, per condition (realistic, stress);
  * confidence intervals resample whole source papers (2000 bootstrap draws), and V5 is
    compared with each other detector on the same resampled papers (paired).
Synthetic, team-made edits. None of this measures real-world misconduct detection.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import time
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "data/benchmark"
OUT = ROOT / "results/paper"
P = 32


def rows_for_scoring():
    rows = json.loads((BENCH / "index.json").read_text())
    train = [r for r in rows if r["split"] == "train"]
    calib_papers = {r["pmcid"] for r in train if zlib.crc32(r["pmcid"].encode()) % 5 == 0}
    calib = [dict(r, role="calibration") for r in train if r["pmcid"] in calib_papers and r["label"] == 0]
    test = [dict(r, role="test") for r in rows if r["split"] == "test"]
    fit = [r for r in train if r["pmcid"] not in calib_papers]
    return fit, calib, test


def meta(r):
    return {k: r.get(k) for k in ("id", "pmcid", "method", "profile", "publish", "label", "role")}


def save_scores(name, recs):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"scores_{name}.json").write_text(json.dumps(recs))
    print(f"SAVED scores_{name}.json with {len(recs)} images", flush=True)


# ---------------------------------------------------------------- V3 and V5
def cmd_cnn(_):
    from btw import cnn, cnn_v5
    from btw.imgio import load_rgb, to_gray
    p3, m3 = cnn.load(ROOT / "models/splice_cnn_v3.npz")
    p5, m5 = cnn_v5.load(ROOT / "models/splice_cnn_v5.npz")
    rule3, rule5 = str(m3.get("rule", "top")), str(m5.get("rule", "top"))
    _, calib, test = rows_for_scoring()
    recs = []
    t0 = time.time()
    for i, r in enumerate(calib + test):
        g = to_gray(load_rgb(BENCH / "images" / f"{r['id']}.png"))
        x3, _, b3, _ = cnn.extract(g)
        x5, _, b5, _ = cnn_v5.extract(g)
        s3 = cnn.relative(cnn.predict(p3, x3, batch=128, logits=True), b3, rule3)[1]
        s5 = cnn_v5.relative(cnn_v5.predict(p5, x5, batch=128, logits=True), b5, rule5)[1]
        recs.append({**meta(r), "v3": float(s3), "v5": float(s5)})
        if (i + 1) % 500 == 0:
            print("CNN_SCORE", i + 1, len(calib) + len(test), round(time.time() - t0), "s", flush=True)
    save_scores("cnn", recs)
    # Reproducibility: does this rebuilt benchmark reproduce run 40225195719's V5 test scores?
    old = ROOT / "results/v5_run_40225195719/v5_test_scores.json"
    if old.exists():
        prev = {d["id"]: d["score"] for d in json.loads(old.read_text())}
        same = [(prev[r["id"]], r["v5"]) for r in recs if r["role"] == "test" and r["id"] in prev]
        diffs = [abs(a - b) for a, b in same]
        rep = {"ids_in_both": len(same), "ids_now": sum(r["role"] == "test" for r in recs), "ids_before": len(prev),
               "max_abs_score_diff": max(diffs) if diffs else None,
               "n_within_1e-3": sum(d < 1e-3 for d in diffs)}
        (OUT / "reproducibility_v5.json").write_text(json.dumps(rep, indent=1))
        print("REPRODUCIBILITY", rep, flush=True)


# ---------------------------------------------------------- EfficientNet-B0
def raw_patches(gray, mask=None):
    """Raw grayscale 32x32 patches at exactly V5's grid positions, with V5's band flags and labels."""
    from btw import cnn_v5
    _, pos, band, frac = cnn_v5.extract(gray, mask)
    g = gray.astype(np.float32)
    h, w = g.shape
    ph, pw = max(0, P - h), max(0, P - w)
    if ph or pw:
        g = np.pad(g, ((0, ph), (0, pw)), mode="edge")
    X = np.stack([g[y:y + P, x:x + P] for x, y in pos]) if pos else np.zeros((0, P, P), np.float32)
    return X.astype(np.uint8), band, frac


def b0_featurize(r):
    from PIL import Image
    from btw.imgio import load_rgb, to_gray
    g = to_gray(load_rgb(BENCH / "images" / f"{r['id']}.png"))
    m = (np.asarray(Image.open(BENCH / "masks" / f"{r['id']}.png")) > 0).astype(np.uint8)
    X, band, frac = raw_patches(g, m)
    # Same sampling as scripts/train_cnn_v5.featurize: all definite positives (>=3.5% edited),
    # up to 40 clean band patches, 8% of clean background patches capped at 24.
    pos = frac >= .035
    nb = np.flatnonzero(band & (frac == 0))
    ng = np.flatnonzero((~band) & (frac == 0))
    rng = np.random.default_rng((zlib.crc32(r["id"].encode()) + 2026) & 0xffffffff)
    bsel = rng.choice(nb, min(40, len(nb)), replace=False)
    gsel = ng[rng.random(len(ng)) < .08]
    if len(gsel) > 24:
        gsel = rng.choice(gsel, 24, replace=False)
    ids = np.concatenate([np.flatnonzero(pos), bsel, gsel]).astype(int)
    return X[ids], pos[ids].astype(np.float32)


def b0_model(pretrained=True):
    import torch
    import torchvision
    w = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    m = torchvision.models.efficientnet_b0(weights=w)
    m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, 1)
    return m


def b0_tensor(X):
    import torch
    t = torch.from_numpy(X.astype(np.float32) / 255.0)[:, None].repeat(1, 3, 1, 1)
    mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
    return (t - mean) / std


def cmd_b0(a):
    import torch
    from joblib import Parallel, delayed
    from btw import cnn
    from btw.imgio import load_rgb, to_gray
    torch.manual_seed(0)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    fit, calib, test = rows_for_scoring()
    if a.limit:
        fit, calib, test = fit[:a.limit], calib[:a.limit // 4], test[:a.limit // 2]
    t0 = time.time()
    vals = Parallel(n_jobs=2)(delayed(b0_featurize)(r) for r in fit)
    X = np.concatenate([v[0] for v in vals])
    y = np.concatenate([v[1] for v in vals])
    if len(X) > 280_000:  # same cap rule as V5: keep every positive, subsample negatives
        posi, negi = np.flatnonzero(y > 0), np.flatnonzero(y == 0)
        keep = np.sort(np.concatenate([posi, np.random.default_rng(2026).choice(negi, 280_000 - len(posi), replace=False)]))
        X, y = X[keep], y[keep]
    print("B0_PATCHES", len(X), "positives", int(y.sum()), round(time.time() - t0), "s", flush=True)
    model = b0_model(pretrained=not a.no_pretrained)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = a.epochs
    steps = epochs * int(np.ceil(len(X) / 250))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    posw = torch.tensor(float((y == 0).sum() / max(1, (y > 0).sum())))
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=posw)
    rng = np.random.default_rng(0)
    model.train()
    for ep in range(epochs):
        order = rng.permutation(len(X))
        tot, n = 0.0, 0
        for b in range(0, len(X), 250):  # batch size 250 as in Manjunath et al.
            ix = order[b:b + 250]
            xb, yb = b0_tensor(X[ix]), torch.from_numpy(y[ix])
            if rng.random() < .5:
                xb = torch.flip(xb, dims=[3])
            opt.zero_grad()
            loss = lossf(model(xb)[:, 0], yb)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss) * len(ix)
            n += len(ix)
        print(f"B0 epoch {ep + 1}/{epochs} loss {tot / n:.4f} elapsed {round(time.time() - t0)} s", flush=True)
    model.eval()
    recs = []
    with torch.no_grad():
        for i, r in enumerate(calib + test):
            g = to_gray(load_rgb(BENCH / "images" / f"{r['id']}.png"))
            Xp, band, _ = raw_patches(g)
            logits = np.concatenate([model(b0_tensor(Xp[j:j + 512]))[:, 0].numpy() for j in range(0, len(Xp), 512)]) if len(Xp) else np.zeros(0)
            s = cnn.relative(logits, band, "top")[1] if len(logits) else 0.0
            recs.append({**meta(r), "b0": float(s)})
            if (i + 1) % 500 == 0:
                print("B0_SCORE", i + 1, len(calib) + len(test), flush=True)
    save_scores("b0", recs)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUT / "b0_patch_baseline.pt")


# ------------------------------------------------------------------ TruFor
def cmd_trufor_prep(a):
    from PIL import Image
    _, calib, test = rows_for_scoring()
    dst = Path(a.dir)
    dst.mkdir(parents=True, exist_ok=True)
    padded = 0
    for r in calib + test:
        im = Image.open(BENCH / "images" / f"{r['id']}.png").convert("RGB")
        w, h = im.size
        if w < 64 or h < 64:  # TruFor's encoder downsamples 32x; pad tiny panels by edge replication
            arr = np.asarray(im)
            arr = np.pad(arr, ((0, max(0, 64 - h)), (0, max(0, 64 - w)), (0, 0)), mode="edge")
            im = Image.fromarray(arr)
            padded += 1
        im.save(dst / f"{r['id']}.png")
    (OUT).mkdir(parents=True, exist_ok=True)
    (OUT / "trufor_prep.json").write_text(json.dumps({"images": len(calib) + len(test), "padded_to_64": padded}))
    print("TRUFOR_PREP", len(calib) + len(test), "padded", padded, flush=True)


def cmd_trufor_read(a):
    _, calib, test = rows_for_scoring()
    recs, missing = [], 0
    for r in calib + test:
        f = Path(a.dir) / f"{r['id']}.png.npz"
        if not f.exists():
            missing += 1
            continue
        d = np.load(f)
        recs.append({**meta(r), "trufor": float(d["score"]) if "score" in d else float(np.max(d["map"]))})
    print("TRUFOR_READ", len(recs), "missing", missing, flush=True)
    save_scores("trufor", recs)


# ----------------------------------------------------------------- compare
def auroc(pos, neg):
    from sklearn.metrics import roc_auc_score
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(roc_auc_score([1] * len(pos) + [0] * len(neg), list(pos) + list(neg)))


def cmd_compare(a):
    merged = {}
    for f in sorted(OUT.glob("scores_*.json")):
        for d in json.loads(f.read_text()):
            merged.setdefault(d["id"], {}).update(d)
    detectors = [k for k in ("v3", "v5", "b0", "trufor") if any(k in d for d in merged.values())]
    rows = [d for d in merged.values() if all(k in d for k in detectors)]
    cal = [d for d in rows if d["role"] == "calibration"]
    test = [d for d in rows if d["role"] == "test"]
    thr = {k: float(np.quantile([d[k] for d in cal], .95)) for k in detectors}
    papers = sorted({d["pmcid"] for d in test})
    by_paper = {p: [d for d in test if d["pmcid"] == p] for p in papers}

    def stat(sample, k, cond):
        ai = [d[k] for d in sample if d["method"] == "ai_splice" and (cond == "overall" or d["profile"] == cond)]
        cl = [d[k] for d in sample if d["method"] == "clean" and (cond == "overall" or d["profile"] == cond)]
        return {"auroc": auroc(ai, cl), "ai_recall": float(np.mean(np.array(ai) > thr[k])) if ai else float("nan"),
                "clean_fpr": float(np.mean(np.array(cl) > thr[k])) if cl else float("nan")}

    rng = random.Random(0)
    B = a.boot
    draws = [[d for p in (rng.choice(papers) for _ in papers) for d in by_paper[p]] for _ in range(B)]
    res = {"n_test_images": len(test), "n_test_papers": len(papers), "n_calibration_clean": len(cal),
           "thresholds_95pct_calibration_clean": thr, "bootstrap": f"{B} paper-level resamples", "detectors": {}, "paired_vs_v5": {}}
    for cond in ("realistic", "stress", "overall"):
        for k in detectors:
            pt = stat(test, k, cond)
            bs = [stat(s, k, cond) for s in draws]
            ci = {m: [float(np.nanpercentile([b[m] for b in bs], 2.5)), float(np.nanpercentile([b[m] for b in bs], 97.5))] for m in pt}
            res["detectors"].setdefault(k, {})[cond] = {**pt, "ci95": ci}
        if "v5" in detectors:
            for k in detectors:
                if k == "v5":
                    continue
                diffs = [stat(s, "v5", cond)["auroc"] - stat(s, k, cond)["auroc"] for s in draws]
                res["paired_vs_v5"].setdefault(k, {})[cond] = {
                    "auroc_diff_v5_minus": stat(test, "v5", cond)["auroc"] - stat(test, k, cond)["auroc"],
                    "ci95": [float(np.nanpercentile(diffs, 2.5)), float(np.nanpercentile(diffs, 97.5))],
                    "fraction_of_draws_v5_higher": float(np.mean(np.array(diffs) > 0))}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "baselines.json").write_text(json.dumps(res, indent=1))
    for cond in ("realistic", "stress"):
        for k in detectors:
            d = res["detectors"][k][cond]
            print(f"{cond:9s} {k:7s} AUROC {d['auroc']:.3f} [{d['ci95']['auroc'][0]:.3f},{d['ci95']['auroc'][1]:.3f}] "
                  f"recall {d['ai_recall']:.3f} fpr {d['clean_fpr']:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("cnn")
    b = sub.add_parser("b0")
    b.add_argument("--epochs", type=int, default=4)
    b.add_argument("--limit", type=int, default=0, help="smoke test on a few images")
    b.add_argument("--no-pretrained", action="store_true")
    for name in ("trufor-prep", "trufor-read"):
        s = sub.add_parser(name)
        s.add_argument("--dir", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    {"cnn": cmd_cnn, "b0": cmd_b0, "trufor-prep": cmd_trufor_prep, "trufor-read": cmd_trufor_read,
     "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
