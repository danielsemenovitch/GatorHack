"""Independent held-out AI-generation test for trained V5 and frozen V3.

The entire third OpenAI source generation is reserved in ai_sources_holdout,
outside ai_sources() so no image from it can enter training. This is leave-
one-source-generation-out, NOT leave-one-generator-family-out: two other
OpenAI-generated blot crops enter training. The model may still overfit the
shared OpenAI generator family. A positive is a team-made synthetic splice
into a published, unmodified CC BY paper blot; it says nothing about authors.
"""
from __future__ import annotations

import hashlib
import json
import random
import zlib
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from btw import cnn, cnn_v5
from btw.imgio import load_rgb, to_gray
from btw.publish import jpeg, resize
from btw.sources import ai_panels
from btw.splice import forge

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data"
HOLDOUT = BASE / "ai_sources_holdout"
BENCH = BASE / "benchmark"


def verify_donors():
    manifest = json.loads((BASE / "v4_ai_donor_manifest.json").read_text())
    assert manifest["origin"].startswith("OpenAI"), "Donor provenance changed unexpectedly"
    for record in manifest["donors"]:
        fp = ROOT / record["path"]
        assert fp.is_file(), f"Missing {fp}"
        sha = hashlib.sha256(fp.read_bytes()).hexdigest()
        assert sha == record["sha256"], f"Invalid donor bytes: {record['path']}"
        assert (str(fp.parent.name) == "ai_sources_holdout") == (record["role"] == "test_only")
    training = sorted(p.name for p in (BASE / "ai_sources").iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    assert "openai_donor_g1_train.jpg" in training
    assert "openai_donor_g2_train.jpg" in training
    assert "openai_donor_g3_holdout.jpg" not in training
    assert "openai_donor_g3_holdout.jpg" in {p.name for p in HOLDOUT.iterdir()}
    return manifest


def score_image(im, params, rule, model):
    X, _, band, _ = model.extract(to_gray(im))
    if not len(X):
        return 0.
    logits = model.predict(params,X,batch=64,logits=True)
    return float(model.relative(logits,band,rule)[1])


def ci_wilson(hits, n):
    if n == 0:
        return None
    p = hits / n
    z = 1.96
    d = 1 + z*z/n
    center = (p+z*z/(2*n))/d
    rad = z*((p*(1-p)/n+z*z/(4*n*n))**.5)/d
    return [round(center-rad,3),round(center+rad,3)]


def metrics(clean, fake, cutoff):
    c, f = np.asarray(clean), np.asarray(fake)
    n = min(len(c),len(f))
    c, f = c[:n], f[:n]
    y = np.r_[np.zeros(n), np.ones(n)]
    scores = np.r_[c, f]
    flags = int((f>cutoff).sum())
    false = int((c>cutoff).sum())
    return {
        "n":n, "detected":flags, "clean_flags":false,
        "sensitivity":round(flags/n,3) if n else None,
        "false_flag_rate":round(false/n,3) if n else None,
        "sensitivity_95ci":ci_wilson(flags,n),
        "false_flag_rate_95ci":ci_wilson(false,n),
        "auroc":round(float(roc_auc_score(y,scores)),3) if n else None
    }


def main():
    manifest=verify_donors()
    unseen=ai_panels(HOLDOUT)
    if not unseen:
        raise RuntimeError("Holdout image has no detectable AI donor panels")
    print(f"AI DONORS: {len(ai_panels())} training band panels, {len(unseen)} held-out band panels",flush=True)

    index=json.loads((BENCH/"index.json").read_text())
    calib_papers={r["pmcid"] for r in index if r["split"]=="train"
                  and zlib.crc32(r["pmcid"].encode()) %5 ==0}
    cal=[r for r in index if r["split"]=="train" and r["label"]==0 and r["pmcid"] in calib_papers]
    cal=sorted(cal,key=lambda r:r["id"])[::max(1,len(cal)//200)][:220]
    test={r["panel"]:r for r in index if r["split"]=="test" and r["method"]=="clean"}
    test_rows=[test[k] for k in sorted(test)]
    rng=random.Random(20261225)
    selected=[]
    for r in test_rows:
        real=load_rgb(BASE/"real_panels"/r["panel"])
        forged=None
        for _ in range(9):
            candidate=forge(real,"ai_splice",rng,ai_panels=unseen)
            if candidate is not None and np.count_nonzero(candidate.mask)>25:
                forged=candidate.image
                break
        if forged is not None:
            selected.append((r,real,forged))
        if len(selected)>=120:
            break
    if len(selected)<40:
        raise RuntimeError(f"Only {len(selected)} successful heldout-source splices, need >=40")
    print(f"UNSEEN_SOURCE generated {len(selected)} test pairs",flush=True)
    summary={
        "synthetic_test_selection":"120 heldout-paper pairs sampled deterministically from entire independent generation; training never saw this source.",
        "source_family":"OpenAI image generation",
        "holdout":"entire source generation g3",
        "n_independent_holdout_image_generations":1,
        "n_test_pairs":len(selected),
        "is_generator_family_disjoint":False,
        "n_calibration_clean_images":len(cal),
        "limitations":["Two other OpenAI generations appear in training; source-generation holdout only.",
                       "Synthetic team-made edits and source papers must not be used to accuse original authors.",
                       "Only one heldout generation: wide sampling uncertainty, not independent proof of generalization."]
    }
    for name,path,model in [("v3","models/splice_cnn_v3.npz",cnn),("v5","models/splice_cnn_v5.npz",cnn_v5)]:
        params,meta=model.load(ROOT/path)
        rule=str(meta.get("rule","top"))
        calibration=[score_image(load_rgb(BENCH/"images"/f"{r['id']}.png"),params,rule,model) for r in cal]
        cutoff=float(np.quantile(calibration,.95))
        score_set={}
        for kind,transform in [("native",lambda im:im),
                               ("resize_50",lambda im:resize(im,.5)),
                               ("jpeg_q75",lambda im:jpeg(im,75))]:
            clean=[score_image(transform(a),params,rule,model) for _,a,_ in selected]
            edited=[score_image(transform(b),params,rule,model) for _,_,b in selected]
            score_set[kind]=metrics(clean,edited,cutoff)
            print(f"UNSEEN_SOURCE {name} {kind}: {score_set[kind]}",flush=True)
        summary[name]={"rule":rule,"calib_cutoff":round(cutoff,4),"results":score_set}
    dest=ROOT/"results"/"v5_unseen_ai_source_eval.json"
    dest.write_text(json.dumps(summary,indent=2))
    print("SAVED",dest,flush=True)


if __name__=="__main__":
    main()
