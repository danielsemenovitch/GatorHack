"""Build fixed, stratified REALISTIC + STRESS heldout V5 benchmark.

No edited paper is claimed to be fraudulent: edited bands and masks are
synthetically generated and labeled. More source papers, NOT more epochs.
Train/calibration/test are paper-disjoint; donors never cross split boundaries.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

from btw.imgio import load_rgb
from btw.publish import random_publish, jpeg, resize
from btw.splice import METHODS, forge
from btw.sources import ai_panels
from scripts.build_benchmark import extract_panels, filter_panels, split_of

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def chosen_panels(cap=1750, per_paper=5):
    pd = DATA / "real_panels"
    rows = json.loads((pd / "panels.json").read_text())
    keep = set(json.loads((pd / "keep.json").read_text()))
    by_paper = collections.defaultdict(list)
    for p in rows:
        if p["panel"] in keep and p.get("pmcid"):
            by_paper[p["pmcid"]].append(p)
    # Deterministic interleaving by paper ensures many distinct authors/papers
    # are represented rather than favoring an arbitrary first batch.
    papers = sorted(by_paper, key=lambda x: hashlib.sha256(x.encode()).hexdigest())
    out = []
    for depth in range(per_paper):
        for pid in papers:
            candidates = sorted(by_paper[pid], key=lambda r: r["panel"])
            if depth < len(candidates):
                out.append(candidates[depth])
                if len(out) >= cap:
                    return out
    return out


def quality(fg, method, w, h):
    """Reject degenerate, tiny or unrealistic pasted regions."""
    mask = fg.mask
    yy, xx = np.nonzero(mask)
    if len(xx) < 20:
        return False
    bw, bh = int(xx.max() - xx.min() + 1), int(yy.max() - yy.min() + 1)
    if bw < 8 or bh < 3 or bw * bh > .5 * w * h:
        return False
    if not (.35 <= bw / max(1, bh) <= 25):
        return False
    changed = len(xx) / max(1, w * h)
    if not (.0004 <= changed <= .25):
        return False
    # Some synthetic forgeries are an implausible near-solid rectangle. Don't
    # teach the model that obvious rectangles are the definition of an edit.
    if method in ("ai_splice", "cross_splice", "copy_move"):
        density = len(xx) / max(1, bw * bh)
        if density > .98 and bw * bh >= 60:
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-panels", type=int, default=1750)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    extract_panels()
    filter_panels()
    panels = chosen_panels(cap=args.max_panels)
    paper_ids = {p["pmcid"] for p in panels}
    if len(paper_ids) < 180:
        raise RuntimeError(f"Only {len(paper_ids)} usable source papers - not enough v4 diversity")
    pd = DATA / "real_panels"
    cache = {p["panel"]: load_rgb(pd / p["panel"]) for p in panels}
    by_split = collections.defaultdict(list)
    for p in panels:
        by_split[split_of(p["pmcid"])].append(p)
    # Only ai_sources/ is used for AI splice creation. The third OpenAI
    # generation lives in ai_sources_holdout/ and MUST NOT enter this pool.
    manifest = json.loads((DATA / "v4_ai_donor_manifest.json").read_text())
    import hashlib as _hashlib
    for record in manifest["donors"]:
        donor_path = ROOT / record["path"]
        if _hashlib.sha256(donor_path.read_bytes()).hexdigest() != record["sha256"]:
            raise RuntimeError("AI donor checksum mismatch: " + str(donor_path))
        if (record["role"] == "train") != (donor_path.parent.name == "ai_sources"):
            if record["role"] == "test_only" and donor_path.parent.name == "ai_sources_holdout":
                continue
            raise RuntimeError("AI donor role/location mismatch; potential holdout leakage")
    if (DATA / "ai_sources" / "openai_donor_g3_holdout.jpg").exists():
        raise RuntimeError("Entire held-out AI donor generation has leaked into training")
    ai = ai_panels()
    print("V4_EXTRA_DONORS: training donor panel crops =", len(ai),
          "; independent source generations added = 2; holdout generation = 1",
          flush=True)
    out = DATA / "benchmark"
    idir, mdir = out / "images", out / "masks"
    idir.mkdir(parents=True, exist_ok=True)
    mdir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    index = []
    rejected = 0
    # Predetermined paired difficulty: one clean + one AI splice per profile
    # for every selected panel. Never select cases based on detector scores.
    jobs = [("clean","realistic"),("clean","stress"),("ai_splice","realistic"),
            ("ai_splice","stress"),("copy_move","realistic"),
            ("cross_splice","stress"),("intensify","realistic"),("erase","stress")]
    def publish(im,mask,profile):
        if profile=="realistic":
            r=rng.random()
            if r<.20: steps=["native"]
            elif r<.65:
                im=jpeg(im,rng.choice([82,88,92,96])); steps=["jpeg_82_plus"]
            else:
                factor=rng.choice([.8,.85,.9,1.0])
                im=resize(im,factor)
                im=jpeg(im,rng.choice([82,88,92]))
                steps=[f"resize_{factor:.2f}","jpeg_82_plus"]
        else:
            factor=rng.choice([.5,.6,.65,.7])
            im=resize(im,factor)
            im=jpeg(im,rng.choice([60,70,78]))
            steps=[f"resize_{factor:.2f}","jpeg_60_78"]
        if im.size!=(mask.shape[1],mask.shape[0]):
            mask=np.asarray(Image.fromarray((mask>0).astype(np.uint8)*255).resize(im.size,Image.Resampling.NEAREST))>127
        return im,mask,steps
    difficulty=collections.Counter()
    zero_masks=0
    accepted_by_profile=collections.Counter()
    for i, p in enumerate(panels):
        base = cache[p["panel"]]
        W, H = base.size
        split = split_of(p["pmcid"])
        # A bounded donor subset prevents O(n^2) memory/time and never includes
        # the same source paper or a paper in the other split.
        donor_rows = [q for q in by_split[split] if q["pmcid"] != p["pmcid"]]
        donor_rows = rng.sample(donor_rows, min(35, len(donor_rows)))
        donors = [cache[q["panel"]] for q in donor_rows]
        for k, (kind, profile) in enumerate(jobs):
            if kind == "clean":
                im = base
                mask = np.zeros((H, W), dtype=np.uint8)
                blend = "none"
            else:
                fg = None
                for attempt in range(5):
                    cur = forge(base, kind, rng, donor_panels=donors, ai_panels=ai)
                    if cur and quality(cur,kind,W,H):
                        # Prespecified visible edit floor for the primary test
                        # and training stratum. Holdout membership independent
                        # of classifier scores or model output.
                        if profile=="realistic":
                            n=int(np.count_nonzero(cur.mask))
                            yy,xx=np.nonzero(cur.mask)
                            if (n/max(1,W*H)<.002 or
                                xx.max()-xx.min()+1<12 or yy.max()-yy.min()+1<4):
                                continue
                        fg = cur
                        break
                if fg is None:
                    rejected += 1
                    continue
                im, mask, blend = fg.image, fg.mask, fg.blend
            im, mask, steps = publish(im,mask,profile)
            if kind!="clean" and not np.count_nonzero(mask):
                zero_masks+=1
                continue
            difficulty[profile]+=1
            accepted_by_profile[f"{profile}_{kind}"]+=1
            id_ = f"{Path(p['panel']).stem}_{k:02d}_{kind}"
            im.save(idir / f"{id_}.png")
            Image.fromarray((mask > 0).astype(np.uint8)*255).save(mdir / f"{id_}.png")
            index.append({
                "id": id_, "split": split, "pmcid": p["pmcid"], "panel": p["panel"],
                "label": int(kind != "clean"), "method": kind, "blend": blend, "profile": profile,
                "publish": "+".join(steps), "w": im.width, "h": im.height,
                "forged_px": int(np.count_nonzero(mask)),
            })
        if (i + 1) % 250 == 0:
            print(f"V4_BENCHMARK {i+1}/{len(panels)} panels; examples {len(index)}", flush=True)
    (out / "index.json").write_text(json.dumps(index, indent=1))
    total = collections.Counter(r["split"] for r in index)
    by_method = dict(collections.Counter(r["method"] for r in index))
    report = {
        "selected_panels": len(panels), "selected_papers": len(paper_ids),
        "profile_counts":dict(difficulty), "method_by_profile":dict(accepted_by_profile),
        "zero_edit_masks_skipped":zero_masks,
        "primary_test_definition": "Visible edits >=0.2% edited pixel fraction, >=12px wide and 4px high before publication; native/quality>=82 JPEG or 80-100% rescale. Controls use comparable transforms.",
        "stress_definition":"50%-70% downscale with JPEG quality 60-78; no minimum edited fraction above base quality filter.",
        "by_split": dict(total), "by_method": by_method,
        "examples": len(index), "rejected_edits": rejected,
        "n_test_papers": len({r["pmcid"] for r in index if r["split"]=="test"}),
        "n_train_papers_before_calibration": len({r["pmcid"] for r in index if r["split"]=="train"}),
        "paper_split": "MD5(PMCID)%100 <30 test, >=30 train (same as v3)",
        "ai_donor_training_original_sources": ["gemini_clean_blot.jpeg", "gemini_film_blot.jpeg",
                                               "openai_donor_g1_train.jpg", "openai_donor_g2_train.jpg"],
        "ai_donor_heldout_original_sources": ["openai_donor_g3_holdout.jpg"],
        "ai_donor_training_panel_crops": len(ai),
        "note": "Synthetic team-created edits with masks. Two original Gemini and two additional OpenAI-generated source blots are TRAINING donor images; independent OpenAI generation g3 is HELD OUT. This is not generator-family-disjoint validation.",
    }
    dest = ROOT / "results" / "v5_benchmark.json"
    dest.write_text(json.dumps(report, indent=2))
    print("V5_BENCHMARK_RESULT", json.dumps(report), flush=True)
    test_ai=[r for r in index if r["split"]=="test" and r["method"]=="ai_splice"]
    primary=sum(r["profile"]=="realistic" for r in test_ai)
    stress=sum(r["profile"]=="stress" for r in test_ai)
    if min(primary,stress)<200 or len(index)<4800 or report["n_test_papers"]<40:
        raise RuntimeError("Expanded benchmark insufficiently diverse; abort v4")
    return report


if __name__ == "__main__":
    main()
