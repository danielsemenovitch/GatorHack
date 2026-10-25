"""Frozen V3 versus trained V5, paired heldout papers and preregistered difficulty.

Test scores NEVER used to select V5's score rule or threshold; those were
chosen on separate calibration papers by train_cnn_v5.
"""
from __future__ import annotations
import json,zlib
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from btw import cnn
from btw.imgio import load_rgb,to_gray

ROOT=Path(__file__).resolve().parents[1]
BENCH=ROOT/"data/benchmark"
RESULTS=ROOT/"results"

def score_v3(params,meta,rows):
    s=[]
    for i,r in enumerate(rows):
        x,_,band,_=cnn.extract(to_gray(load_rgb(BENCH/"images"/f"{r['id']}.png")))
        logits=cnn.predict(params,x,batch=128,logits=True)
        s.append(cnn.relative(logits,band,str(meta.get("rule","top")))[1])
        if (i+1)%500==0:print("FROZEN_V3_SCORE",i+1,len(rows),flush=True)
    return np.asarray(s)

def metrics(rows,s,thr):
    y=np.asarray([r["label"]==1 for r in rows],bool)
    hit=s>thr
    res={"n":len(rows),"n_papers":len({r["pmcid"] for r in rows}),
         "threshold":round(float(thr),6),
         "auroc":round(float(roc_auc_score(y,s)),4) if len(set(y))==2 else None,
         "clean_false_flag_rate":round(float(hit[~y].mean()),4) if (~y).any() else None,
         "n_clean":int((~y).sum()),
         "methods":{}}
    for method in sorted({r["method"] for r in rows}):
        ix=np.array([r["method"]==method for r in rows])
        res["methods"][method]={"n":int(ix.sum()),"flag_fraction":round(float(hit[ix].mean()),4)}
    ix=np.array(["resize_0.50" in r["publish"] for r in rows])
    ai=np.array([r["method"]=="ai_splice" for r in rows])
    if (ix&ai).any():
        res["ai_resize50"]={"n":int((ix&ai).sum()),"recall":round(float(hit[ix&ai].mean()),4)}
    return res

def main():
    rows=json.loads((BENCH/"index.json").read_text())
    train=[r for r in rows if r["split"]=="train"]
    test=[r for r in rows if r["split"]=="test"]
    papers={r["pmcid"] for r in train}
    calib_papers={p for p in papers if zlib.crc32(p.encode())%5==0}
    cal=[r for r in train if r["pmcid"] in calib_papers and r["label"]==0]
    params,meta=cnn.load(ROOT/"models/splice_cnn_v3.npz")
    if meta.get("space")!="logit":raise RuntimeError("Unexpected V3 score scale")
    print(f"V5_EVAL frozen V3: {len(cal)} clean calibration, {len(test)} test images",flush=True)
    scores_cal=score_v3(params,meta,cal)
    t3=float(np.quantile(scores_cal,.95))
    scores_v3=score_v3(params,meta,test)
    scores_v5_data=json.loads((RESULTS/"v5_test_scores.json").read_text())
    score_lookup={r["id"]:r["score"] for r in scores_v5_data}
    assert len(score_lookup)==len(test) and all(r["id"] in score_lookup for r in test)
    scores_v5=np.array([score_lookup[r["id"]] for r in test],float)
    v5_train=json.loads((RESULTS/"cnn_v5_metrics.json").read_text())
    t5=float(v5_train["threshold"])
    output={"benchmark":"fixed, paper-disjoint V5 realistic and stress panels, same candidates for frozen V3 and V5",
            "source":"synthetic controlled forgeries, not real-world misconduct",
            "calibration":"separate, paper-disjoint calibration; each classifier calibrated to 5% false flags on pooled clean controls",
            "target":{"realistic_ai_splice_recall":.5,"realistic_clean_fpr_max":.05,
                      "outrank_v3_auroc":True,"stress_must_not_be_omitted":True},
            "split_counts":{"calibration_clean":len(cal),"test":len(test),
                            "test_papers":len({r["pmcid"] for r in test})},
            "v3":{},"v5":{}}
    for profile in ["overall","realistic","stress"]:
        chosen=list(range(len(test))) if profile=="overall" else [i for i,r in enumerate(test) if r["profile"]==profile]
        rr=[test[i] for i in chosen]
        output["v3"][profile]=metrics(rr,scores_v3[chosen],t3)
        output["v5"][profile]=metrics(rr,scores_v5[chosen],t5)
        print("V5_MATCHED_PROFILE",profile,"V3",output["v3"][profile],"V5",output["v5"][profile],flush=True)
    real=output["v5"]["realistic"]
    v3=output["v3"]["realistic"]
    recall=real["methods"]["ai_splice"]["flag_fraction"]
    fpr=real["clean_false_flag_rate"]
    output["primary_target_met"]=bool(recall>=.5 and fpr<=.05 and real["auroc"]>v3["auroc"])
    output["should_auto_deploy"]=False
    output["limitations"]=[
        "Realistic subgroup is prespecified readable larger edit crops, not random misconduct or universal real-world performance.",
        "Other edits and 50% compression/downscale remain important; stress subgroup reports these separately.",
        "Only four independent AI donors in training (two Gemini plus two OpenAI), and one held-out OpenAI source generation.",
        "Matching thresholds are calibrated separately on clean calibration papers; model selection did not use test data.",
        "No automatic promotion without external real-paper validation and model wiring."
    ]
    dest=RESULTS/"cnn_v5_vs_v3.json"
    dest.write_text(json.dumps(output,indent=2))
    print("V5_FINAL_PAIRED_RESULTS",json.dumps({
        "v3_realistic":output["v3"]["realistic"],
        "v5_realistic":output["v5"]["realistic"],
        "v3_stress":output["v3"]["stress"],
        "v5_stress":output["v5"]["stress"],
        "primary_target_met":output["primary_target_met"]
    },indent=2),flush=True)

if __name__=="__main__":main()
