"""Hard-stop preflight for V5 data. Read-only except provenance result JSON.

Any failure blocks training. No held-out scores are used in these tests.
"""
from __future__ import annotations
import hashlib,json,zlib
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from scripts.eval_v4_extra_unseen_donor import verify_donors
from scripts.train_cnn_v5 import patch_labels
from btw import cnn_v5

ROOT=Path(__file__).resolve().parents[1]

def unit():
    f=np.array([0.,.001,.02,.035,.04,.25],np.float32)
    band=np.array([1,1,1,1,1,1],bool)
    pos,nb,_=patch_labels(f,band)
    assert pos.tolist()==[False,False,False,True,True,True]
    assert len(nb)==1 and nb[0]==0
    verify_donors()
    cnn_v5.self_test()
    print("V5_UNIT_PREFLIGHT_OK",flush=True)

def dataset():
    index=json.loads((ROOT/"data/benchmark/index.json").read_text())
    assert len(index)>4800
    tr={x["pmcid"] for x in index if x["split"]=="train"}
    test={x["pmcid"] for x in index if x["split"]=="test"}
    cal={p for p in tr if zlib.crc32(p.encode())%5==0}
    fitted=tr-cal
    assert not tr&test and not fitted&cal
    assert len(fitted)>=150 and len(cal)>=25 and len(test)>=40
    assert len(set(x["id"] for x in index))==len(index)
    profiles=Counter((x["split"],x["profile"],x["method"]) for x in index)
    for profile in ["realistic","stress"]:
        assert profiles[("test",profile,"clean")]>=200
        assert profiles[("test",profile,"ai_splice")]>=200
        assert profiles[("train",profile,"clean")]>=400
        assert profiles[("train",profile,"ai_splice")]>=400
    bypanel=defaultdict(set)
    for x in index:
        assert x["profile"] in ("realistic","stress")
        assert x["label"]==int(x["method"]!="clean")
        bypanel[(x["split"],x["panel"])].add(x["profile"])
    assert len(bypanel)>=450
    # The published source image content must be real, licensed,
    # and never the same PNG bytes on BOTH sides of a paper split.
    source_sha=defaultdict(set)
    for x in index:
        if x["method"]=="clean":
            path=ROOT/"data/real_panels"/x["panel"]
            source_sha[hashlib.sha256(path.read_bytes()).hexdigest()].add(x["split"])
    leakage=sum(len(split)>1 for split in source_sha.values())
    if leakage:raise RuntimeError(f"{leakage} identical source images crossed paper splits")
    # Sample all edit masks: no zero-delta 'edited' images in benchmark.
    for i,x in enumerate(index):
        if x["label"]:
            p=ROOT/"data/benchmark/masks"/(x["id"]+".png")
            if not p.exists() or not np.any(np.asarray(Image.open(p))):
                raise AssertionError(f"Edited row has zero or missing edit mask: {x['id']}")
    result={"n":len(index),"train_papers":len(fitted),"calibration_papers":len(cal),
            "test_papers":len(test),"by_split_profile_method":{
                "|".join(key):n for key,n in sorted(profiles.items())},
            "cross_split_exact_duplicate_sources":leakage,
            "holdout_source":"openai_donor_g3_holdout.jpg (excluded from training)",
            "test_label_policy":"realistic and stress profiles preassigned before model training"}
    (ROOT/"results/v5_preflight.json").write_text(json.dumps(result,indent=2))
    print("V5_DATA_PREFLIGHT_OK",json.dumps(result),flush=True)

if __name__=="__main__":
    import sys
    if sys.argv[-1]=="--unit":unit()
    elif sys.argv[-1]=="--dataset":dataset()
    else:raise SystemExit("Use --unit or --dataset")
