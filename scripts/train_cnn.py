"""Train the CNN patch detector on the benchmark and evaluate it per panel.

Same protocol as train_detector.py: splits by paper, threshold fixed on
training clean panels (5% false flags), per method rates, held out method.
Writes results/cnn_metrics.json and models/splice_cnn.npz.
"""
from __future__ import annotations

import json
import zlib
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

from btw import cnn
from btw.imgio import load_rgb, to_gray

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "data" / "benchmark"


def load_row(row):
    img = load_rgb(BENCH / "images" / f"{row['id']}.png")
    mask = (np.asarray(Image.open(BENCH / "masks" / f"{row['id']}.png")) > 0).astype(np.float32)
    return cnn.extract(to_gray(img), mask)


def featurize(row, seed=0, bg_keep=0.08, max_band=40):
    """Sample training patches from one image: all forged patches, up to
    max_band real band patches (hard negatives), a few background patches."""
    rng = np.random.default_rng(zlib.crc32(row["id"].encode()) + seed)
    X, pos_xy, band, frac = load_row(row)
    pos = frac >= 0.25
    neg_band = np.nonzero(band & (frac == 0))[0]
    neg_bg = np.nonzero(~band & (frac == 0))[0]
    nb = rng.choice(neg_band, min(max_band, len(neg_band)), replace=False) if len(neg_band) else np.zeros(0, int)
    ng = neg_bg[rng.random(len(neg_bg)) < bg_keep]
    idx = np.concatenate([np.nonzero(pos)[0], nb, ng]).astype(int)
    return row["id"], X[idx].astype(np.float16), pos[idx].astype(np.float32), row["method"]


def sample(rows, data, rng=None):
    Xs = [data[r["id"]][0] for r in rows]; ys = [data[r["id"]][1] for r in rows]
    return np.concatenate(Xs).astype(np.float32), np.concatenate(ys)


def train(X, y, epochs=8, seed=0, lr=2e-3, batch=256):
    import jax
    import jax.numpy as jnp
    import optax
    params = jax.tree_util.tree_map(jnp.asarray, cnn.init_params(seed))
    opt = optax.adamw(optax.cosine_decay_schedule(lr, epochs * (len(X) // batch + 1)), weight_decay=1e-4)
    state = opt.init(params)
    w_pos = float((y == 0).sum() / max(1, (y == 1).sum()))

    @jax.jit
    def step(p, s, xb, yb):
        def loss(p):
            lg = cnn.apply(p, xb)
            l = optax.sigmoid_binary_cross_entropy(lg, yb)
            return jnp.mean(l * jnp.where(yb > 0.5, w_pos, 1.0))
        l, g = jax.value_and_grad(loss)(p)
        u, s = opt.update(g, s, p)
        return optax.apply_updates(p, u), s, l

    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        perm = rng.permutation(len(X)); tot = 0
        for i in range(0, len(X) - batch + 1, batch):
            b = perm[i:i + batch]
            xb = X[b]
            flip = rng.random(len(b)) < 0.5
            xb[flip] = xb[flip][:, :, ::-1]
            params, state, l = step(params, state, xb, y[b]); tot += float(l)
        print(f"  epoch {ep + 1}/{epochs} loss {tot / max(1, len(X) // batch):.4f}", flush=True)
    return jax.tree_util.tree_map(np.asarray, params)


def panel_score(probs, band, k=3):
    p = probs[band] if band.sum() >= 3 else probs
    if len(p) == 0:
        return 0.0
    return float(np.sort(p)[-min(k, len(p)):].mean())


_PATCH_CACHE: dict = {}


def scores(params, rows, data=None):
    """Score whole panels, extracting all their patches on the fly."""
    out = {}
    for r in rows:
        X, _, band, _ = load_row(r)
        pr = cnn.predict(params, X)
        out[r["id"]] = panel_score(pr, band)
    return out


def main():
    t0 = time.time()
    index = json.loads((BENCH / "index.json").read_text())
    train_rows = [r for r in index if r["split"] == "train"]
    test_rows = [r for r in index if r["split"] == "test"]
    res = Parallel(n_jobs=2)(delayed(featurize)(r) for r in train_rows)
    data = {i: (X, y) for i, X, y, _ in res}
    print(f"patches ready in {time.time() - t0:.0f}s", flush=True)
    Xtr, ytr = sample(train_rows, data)
    print(f"training on {len(Xtr)} patches ({int(ytr.sum())} forged)", flush=True)
    params = train(Xtr, ytr)

    # Threshold from a sample of clean TRAINING panels (scoring every train image is slow on CPU).
    clean_tr = [r for r in train_rows if r["label"] == 0]
    tr_s = scores(params, clean_tr)
    thr = float(np.quantile(list(tr_s.values()), 0.95))
    te_s = scores(params, test_rows)
    y = np.array([r["label"] for r in test_rows]); s = np.array([te_s[r["id"]] for r in test_rows])
    by_m = defaultdict(list)
    for r, v in zip(test_rows, s):
        by_m[r["method"]].append(v > thr)
    size_auc = {}
    for name, f in (("small (<10k px)", lambda r: r["w"] * r["h"] < 10000),
                    ("medium", lambda r: 10000 <= r["w"] * r["h"] < 40000),
                    ("large (>=40k px)", lambda r: r["w"] * r["h"] >= 40000)):
        rr = [(r, v) for r, v in zip(test_rows, s) if f(r)]
        yy = [r["label"] for r, _ in rr]
        if 0 < sum(yy) < len(yy):
            size_auc[name] = {"n": len(rr), "auroc": round(float(roc_auc_score(yy, [v for _, v in rr])), 3)}

    # Held out method: retrain without AI splices, test AI splice vs clean.
    tr_h = [r for r in train_rows if r["method"] != "ai_splice"]
    p_h = train(*sample(tr_h, data), epochs=6)
    sh = scores(p_h, [r for r in tr_h if r["label"] == 0])
    thr_h = float(np.quantile(list(sh.values()), 0.95))
    te_h = [r for r in test_rows if r["method"] in ("ai_splice", "clean")]
    sth = scores(p_h, te_h)
    yh = np.array([r["label"] for r in te_h]); vh = np.array([sth[r["id"]] for r in te_h])

    m = {"model": "cnn", "n_train_images": len(train_rows), "n_test_images": len(test_rows),
         "n_train_papers": len({r["pmcid"] for r in train_rows}), "n_test_papers": len({r["pmcid"] for r in test_rows}),
         "threshold": round(thr, 4), "image_auroc": round(float(roc_auc_score(y, s)), 3),
         "false_flag_rate_clean": round(float(np.mean(s[y == 0] > thr)), 3),
         "detect_rate_by_method": {k: round(float(np.mean(v)), 3) for k, v in by_m.items()},
         "auroc_by_panel_size": size_auc,
         "held_out_method": {"ai_splice": {"auroc": round(float(roc_auc_score(yh, vh)), 3),
                                           "detect_rate": round(float(np.mean(vh[yh == 1] > thr_h)), 3),
                                           "false_flag_rate": round(float(np.mean(vh[yh == 0] > thr_h)), 3)}},
         "seconds": round(time.time() - t0)}
    (ROOT / "results" / "cnn_metrics.json").write_text(json.dumps(m, indent=1))
    cnn.save(params, threshold=thr)
    print(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
