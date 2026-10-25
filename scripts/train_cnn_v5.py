"""Final V5 residual CNN training. Predeclared patch sampling and 6 epochs.

Never overwrite V3. Never select a cutoff or architecture from test results.
Read data/benchmark/index.json created with build_v5_benchmark.
"""
from __future__ import annotations
import json,os,time,zlib
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
from joblib import Parallel,delayed
from sklearn.metrics import roc_auc_score
from btw import cnn_v5
from btw.imgio import load_rgb,to_gray

ROOT=Path(__file__).resolve().parents[1]
BENCH=ROOT/"data/benchmark"
RESULTS=ROOT/"results"
MODEL=ROOT/"models/splice_cnn_v5.npz"
EPOCHS=6
BATCH=256
MAX_PATCHES=280_000

def patch_labels(frac, band):
    # Prior >=25% patch-overlap was too harsh for thin bands. Use >=3.5%
    # for definite positives; NEVER call a partially edited patch clean.
    pos=frac>=.035
    neg_band=np.flatnonzero(band & (frac==0))
    neg_bg=np.flatnonzero((~band) & (frac==0))
    return pos,neg_band,neg_bg

def featurize(row):
    img=load_rgb(BENCH/"images"/f"{row['id']}.png")
    mask=(np.asarray(Image.open(BENCH/"masks"/f"{row['id']}.png"))>0).astype(np.uint8)
    x,_,band,frac=cnn_v5.extract(to_gray(img),mask)
    pos,nb,ng=patch_labels(frac,band)
    rng=np.random.default_rng((zlib.crc32(row["id"].encode())+2026)&0xffffffff)
    bsel=rng.choice(nb,min(40,len(nb)),replace=False)
    gsel=ng[rng.random(len(ng))<.08]
    if len(gsel)>24:gsel=rng.choice(gsel,24,replace=False)
    ids=np.concatenate([np.flatnonzero(pos),bsel,gsel])
    stats={"id":row["id"],"method":row["method"],"profile":row.get("profile"),
           "image_changed":bool(row["label"]),"eligible_positive":int(pos.sum()),
           "positive_candidates":int((frac>0).sum()),
           "ambiguous_excluded":int(((frac>0)&(~pos)).sum()),
           "negative_count":len(ids)-int(pos.sum())}
    return x[ids].astype(np.float16),pos[ids].astype(np.float32),stats

def pack(rows):
    start=time.time()
    # Use max 2 workers; the patch tensor may consume >2 GB RAM.
    vals=Parallel(n_jobs=2)(delayed(featurize)(r) for r in rows)
    X=np.concatenate([v[0] for v in vals]); y=np.concatenate([v[1] for v in vals])
    stats=[v[2] for v in vals]
    if len(X)>MAX_PATCHES:
        # Retain *all* definite positive examples. Deterministically
        # subsample only negatives so class balance is not distorted by
        # silently throwing away rare edited pixels.
        pos=np.flatnonzero(y>0); neg=np.flatnonzero(y==0)
        if len(pos)>MAX_PATCHES:raise RuntimeError("Too many positives for memory budget")
        rng=np.random.default_rng(2026)
        keep=np.sort(np.concatenate([pos,rng.choice(neg,MAX_PATCHES-len(pos),replace=False)]))
        X,y=X[keep],y[keep]
    missing=[r for r in stats if r["image_changed"] and not r["eligible_positive"]]
    positives=sum(int(y0.sum()) for _,y0,_ in vals)
    report={"images":len(rows),"patches":len(X),"forged_patches":int(y.sum()),
            "forged_before_cap":positives,"missing_positive_images":len(missing),
            "missing_positive_examples":missing[:35],
            "ambiguous_patches_excluded":sum(s["ambiguous_excluded"] for s in stats),
            "elapsed_seconds":round(time.time()-start,1)}
    print("V5_PATCH_AUDIT",json.dumps(report),flush=True)
    if len(y)<60_000 or int(y.sum())<3000:
        raise RuntimeError("Not enough labeled training patches; stop before training")
    if len(missing)/max(1,sum(r["label"] for r in rows))>.23:
        raise RuntimeError("More than 23% of edited train panels have zero definite positive patches")
    return X,y,report

def train(X,y):
    import jax,jax.numpy as jnp,optax
    p=jax.tree_util.tree_map(jnp.asarray,cnn_v5.init_params(seed=0))
    steps=EPOCHS*(len(X)//BATCH+1)
    schedule=optax.warmup_cosine_decay_schedule(init_value=3e-4,peak_value=1.5e-3,
              warmup_steps=max(1,int(steps*.05)),decay_steps=steps,end_value=8e-6)
    opt=optax.adamw(learning_rate=schedule,weight_decay=1e-4)
    state=opt.init(p)
    ratio=float((y==0).sum())/max(1,(y==1).sum())
    pos_w=float(min(6.,max(1.5,np.sqrt(ratio))))
    @jax.jit
    def step(params,opt_state,xb,yb):
        def loss(pp):
            logit=cnn_v5.apply(pp,xb)
            l=optax.sigmoid_binary_cross_entropy(logit,yb)
            return jnp.mean(l*jnp.where(yb>.5,pos_w,1.))
        val,grads=jax.value_and_grad(loss)(params)
        upd,next_state=opt.update(grads,opt_state,params)
        return optax.apply_updates(params,upd),next_state,val
    rng=np.random.default_rng(0)
    history=[]
    for ep in range(EPOCHS):
        perm=rng.permutation(len(X))
        total=0.;n=0
        for i in range(0,len(X)-BATCH+1,BATCH):
            ids=perm[i:i+BATCH]
            xb=X[ids].astype(np.float32)
            # Same deterministic augmentation for every epoch: orientation,
            # mild global contrast and noise (no model-dependent tuning).
            flips=rng.random(len(xb))<.5
            xb[flips]=xb[flips,:,::-1,:]
            vflips=rng.random(len(xb))<.25
            xb[vflips]=xb[vflips,::-1,:,:]
            xb[:,:,:,0]*=rng.uniform(.85,1.18,(len(xb),1,1)).astype(np.float32)
            xb[:,:,:,1]*=rng.uniform(.8,1.25,(len(xb),1,1)).astype(np.float32)
            if not np.isfinite(xb).all():raise ValueError("Non-finite features")
            p,state,value=step(p,state,xb,jnp.asarray(y[ids]))
            loss=float(value)
            if not np.isfinite(loss):raise FloatingPointError("V5 training diverged")
            total+=loss;n+=1
        history.append(round(total/max(1,n),5))
        print(f"V5 epoch {ep+1}/{EPOCHS} weighted_loss {history[-1]}",flush=True)
    return jax.tree_util.tree_map(np.asarray,p),history,pos_w

def evaluate_scores(params,rows,rule):
    scores=[]
    for i,row in enumerate(rows):
        im=load_rgb(BENCH/"images"/f"{row['id']}.png")
        x,_,band,_=cnn_v5.extract(to_gray(im))
        logits=cnn_v5.predict(params,x,batch=128,logits=True)
        scores.append(cnn_v5.relative(logits,band,rule)[1])
        if (i+1)%500==0:print("V5_SCORE",i+1,"of",len(rows),flush=True)
    return np.asarray(scores)

def main():
    start=time.time()
    all_rows=json.loads((BENCH/"index.json").read_text())
    all_train=[r for r in all_rows if r["split"]=="train"]
    test=[r for r in all_rows if r["split"]=="test"]
    calib_papers={p for p in {r["pmcid"] for r in all_train} if zlib.crc32(p.encode())%5==0}
    tr=[r for r in all_train if r["pmcid"] not in calib_papers]
    cal=[r for r in all_train if r["pmcid"] in calib_papers]
    assert not ({r["pmcid"] for r in tr} & {r["pmcid"] for r in cal})
    assert not ({r["pmcid"] for r in tr+cal} & {r["pmcid"] for r in test})
    X,y,audit=pack(tr)
    del tr # retain only rows? count previously needed use total
    params,history,pos_weight=train(X,y)
    del X,y
    cal_logits={}
    for rule in cnn_v5.RULES:
        cal_logits[rule]=evaluate_scores(params,cal,rule)
    labels=np.array([r["label"] for r in cal])
    rule=max(cnn_v5.RULES,key=lambda k:roc_auc_score(labels,cal_logits[k]))
    clean=cal_logits[rule][labels==0]
    threshold=float(np.quantile(clean,.95))
    q25,q75=np.quantile(clean,[.25,.75])
    scale=float(max(.05,(q75-q25)/1.349))
    print("V5_CALIBRATION",json.dumps({"rule":rule,"threshold":threshold,
          "auroc_by_rule":{r:round(float(roc_auc_score(labels,cal_logits[r])),4) for r in cnn_v5.RULES}}),flush=True)
    # Keep calibrated model even if evaluation later fails, and preserve logs.
    cnn_v5.save(params,path=MODEL,threshold=threshold,rule=rule,scale=scale,
          space="logit",architecture="v5_residual_3_channel",
          epochs=EPOCHS,weight_pos=pos_weight)
    test_scores=evaluate_scores(params,test,rule)
    ytest=np.asarray([r["label"] for r in test],bool)
    def metric(rows,s):
        labels=np.asarray([r["label"] for r in rows],bool)
        det=s>threshold
        if not len(rows) or len(set(labels))<2:return {"n":len(rows),"auroc":None}
        ai=np.asarray([r["method"]=="ai_splice" for r in rows])
        return {"n":len(rows),"papers":len({r["pmcid"] for r in rows}),
                "auroc":round(float(roc_auc_score(labels,s)),4),
                "ai_splice_recall":round(float(det[ai].mean()),4) if ai.any() else None,
                "n_ai_splice":int(ai.sum()),
                "clean_fpr":round(float(det[~labels].mean()),4)}
    results={"model":"v5_experimental","trained_epochs":EPOCHS,
             "weight_pos":pos_weight,"loss_curve":history,
             "n_train_papers":len({r["pmcid"] for r in all_train})-len(calib_papers),
             "n_calib_papers":len(calib_papers),"n_test_papers":len({r["pmcid"] for r in test}),
             "threshold":threshold,"score_rule":rule,"patch_label_rule":"fraction >=0.035; ambiguous (>0 but <0.035) excluded",
             "train_audit":audit,"overall":metric(test,test_scores),
             "profiles":{},"elapsed_seconds":round(time.time()-start,1)}
    for profile in ["realistic","stress"]:
        ix=[i for i,r in enumerate(test) if r["profile"]==profile]
        results["profiles"][profile]=metric([test[i] for i in ix],test_scores[ix])
    out=RESULTS/"cnn_v5_metrics.json"
    out.write_text(json.dumps(results,indent=2))
    print("V5_METRICS",json.dumps(results,indent=2),flush=True)

if __name__=="__main__":main()
