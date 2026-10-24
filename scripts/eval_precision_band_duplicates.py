"""Precision-first copied-band matching check on kept article panels.

Treated images are synthetic edits made by this script. Untouched clean
samples are presumed controls, not adjudicated journal misconduct cases.
The test is NOT used to tune the model thresholds.
"""
from __future__ import annotations
import json
import random
from pathlib import Path
import numpy as np
from btw.imgio import load_rgb, to_gray
from btw.duplicates import find_duplicates
from scripts.eval_duplicates import make_copy

ROOT=Path(__file__).resolve().parents[1]
def hit(im):
    return bool(find_duplicates([to_gray(im)], threshold=0.90, shape_floor=0.90))

def main():
    keep=json.loads((ROOT/'data/real_panels/keep.json').read_text())
    rng=random.Random(7)
    positives=0
    clean_hit=0
    edited_hit=0
    for name in sorted(keep):
        clean=load_rgb(ROOT/'data/real_panels'/name)
        modified=make_copy(clean,rng)
        if modified is None:
            continue
        positives+=1
        clean_hit+=bool(hit(clean))
        edited_hit+=bool(hit(modified))
        if positives%50==0:
            print("scored", positives, flush=True)
    if not positives:
        raise RuntimeError('No eligible bands for controlled copy test')
    res={
        "method":"Conservative copied-band review candidate operating point",
        "threshold_noise":0.90, "threshold_shape":0.90,
        "n_pairs":positives,
        "false_flag_rate_clean":round(clean_hit/positives,4),
        "detect_rate_synthetic_copy":round(edited_hit/positives,4),
        "false_flags":clean_hit,"hits":edited_hit,
        "note":"In-sample panel-set sanity check; not held-out by paper, not comparable to old metrics. This high-confidence policy sacrifices recall.",
    }
    dst=ROOT/'results/duplicates_conservative_sanity.json'
    dst.write_text(json.dumps(res,indent=2))
    print("RESULT",json.dumps(res),flush=True)

if __name__=="__main__":main()
