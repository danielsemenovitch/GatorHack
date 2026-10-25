"""Build independent-paper v4 benchmark using newly fetched CC BY figures.

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
    ai = ai_panels()
    out = DATA / "benchmark"
    idir, mdir = out / "images", out / "masks"
    idir.mkdir(parents=True, exist_ok=True)
    mdir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    index = []
    rejected = 0
    # Two independent clean publishing renders; two AI-splice edits; one
    # example each of four other edit mechanisms, each with a pixel truth mask.
    jobs = ["clean", "clean", "ai_splice", "ai_splice",
            "copy_move", "cross_splice", "intensify", "erase"]
    for i, p in enumerate(panels):
        base = cache[p["panel"]]
        W, H = base.size
        split = split_of(p["pmcid"])
        # A bounded donor subset prevents O(n^2) memory/time and never includes
        # the same source paper or a paper in the other split.
        donor_rows = [q for q in by_split[split] if q["pmcid"] != p["pmcid"]]
        donor_rows = rng.sample(donor_rows, min(35, len(donor_rows)))
        donors = [cache[q["panel"]] for q in donor_rows]
        for k, kind in enumerate(jobs):
            if kind == "clean":
                im = base
                mask = np.zeros((H, W), dtype=np.uint8)
                blend = "none"
            else:
                fg = None
                for attempt in range(5):
                    cur = forge(base, kind, rng, donor_panels=donors, ai_panels=ai)
                    if cur and quality(cur, kind, W, H):
                        fg = cur
                        break
                if fg is None:
                    rejected += 1
                    continue
                im, mask, blend = fg.image, fg.mask, fg.blend
            # Ensure strong low-resolution publishing conditions are in both
            # clean and edited variants, without making resize a label cue.
            if rng.random() < .22:
                im = resize(im, .5)
                im = jpeg(im, rng.choice([65, 75, 85, 92]))
                steps = ["resize_50", "jpeg"]
            else:
                im, steps = random_publish(im, rng)
            if im.size != (W, H):
                mask = np.asarray(Image.fromarray(mask * 255).resize(im.size, Image.Resampling.NEAREST)) > 127
            id_ = f"{Path(p['panel']).stem}_{k:02d}_{kind}"
            im.save(idir / f"{id_}.png")
            Image.fromarray((mask > 0).astype(np.uint8)*255).save(mdir / f"{id_}.png")
            index.append({
                "id": id_, "split": split, "pmcid": p["pmcid"], "panel": p["panel"],
                "label": int(kind != "clean"), "method": kind, "blend": blend,
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
        "by_split": dict(total), "by_method": by_method,
        "examples": len(index), "rejected_edits": rejected,
        "n_test_papers": len({r["pmcid"] for r in index if r["split"]=="test"}),
        "n_train_papers_before_calibration": len({r["pmcid"] for r in index if r["split"]=="train"}),
        "paper_split": "MD5(PMCID)%100 <30 test, >=30 train (same as v3)",
        "note": "Synthetic edits and matching pixel masks. Prior works are untouched. AI donors originate from TWO Gemini images: more synthetic edits do not create independent generators.",
    }
    dest = ROOT / "results" / "v4_benchmark.json"
    dest.write_text(json.dumps(report, indent=2))
    print("V4_BENCHMARK_RESULT", json.dumps(report), flush=True)
    if len(index) < 5500 or report["n_test_papers"] < 40:
        raise RuntimeError("Expanded benchmark insufficiently diverse; abort v4")
    return report


if __name__ == "__main__":
    main()
