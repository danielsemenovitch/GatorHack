"""Paper-clustered bootstrap statistics for the V5 matched test set (run 40225195719).

Reads results/v5_run_40225195719/v5_test_scores.json. Image ids end in
_{k:02d}_{method}; k maps to the condition through the fixed job list in
scripts/build_v5_benchmark.py. Resamples whole source papers, because panels
from one paper are correlated. V3 per-image scores were not saved by the run,
so V3 appears only as point estimates from cnn_v5_vs_v3.json.
"""
import json, re, random, sys
from pathlib import Path

RUN = Path("results/v5_run_40225195719")
JOBS = ["clean:realistic", "clean:stress", "ai_splice:realistic", "ai_splice:stress",
        "copy_move:realistic", "cross_splice:stress", "intensify:realistic", "erase:stress"]
THRESH = json.loads((RUN / "cnn_v5_metrics.json").read_text())["threshold"]


def parse(rec):
    m = re.match(r"(PMC\d+)_.*_(\d{2})_([a-z_]+)$", rec["id"])
    paper, k, method = m.group(1), int(m.group(2)), m.group(3)
    kind, cond = JOBS[k].split(":")
    assert kind == method, rec["id"]
    return paper, cond, method, rec["score"]


def auroc(pos, neg):
    if not pos or not neg:
        return float("nan")
    allv = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg])
    rank, i, rsum = 1, 0, 0.0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        avg = (rank + rank + (j - i) - 1) / 2
        rsum += avg * sum(1 for t in allv[i:j] if t[1] == 1)
        rank += j - i
        i = j
    return (rsum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def stats(rows, cond):
    """AUROC here is AI splice vs clean only; the run log reports AUROC over all edit types (0.8718 realistic, reproduced exactly)."""
    pos = [s for p, c, m, s in rows if m == "ai_splice" and (cond == "overall" or c == cond)]
    neg = [s for p, c, m, s in rows if m == "clean" and (cond == "overall" or c == cond)]
    return {"auroc": auroc(pos, neg),
            "ai_recall": sum(s >= THRESH for s in pos) / len(pos),
            "clean_fpr": sum(s >= THRESH for s in neg) / len(neg),
            "n_ai": len(pos), "n_clean": len(neg)}


def main(B=2000, seed=0):
    recs = [parse(r) for r in json.loads((RUN / "v5_test_scores.json").read_text())]
    by_paper = {}
    for r in recs:
        by_paper.setdefault(r[0], []).append(r)
    papers = sorted(by_paper)
    rng = random.Random(seed)
    out = {"run": 40225195719, "threshold": THRESH, "n_images": len(recs), "n_papers": len(papers),
           "bootstrap": f"{B} resamples of whole source papers, percentile 95% intervals", "v5": {}}
    for cond in ["realistic", "stress", "overall"]:
        point = stats(recs, cond)
        boots = {k: [] for k in ["auroc", "ai_recall", "clean_fpr"]}
        for _ in range(B):
            sample = [r for p in (rng.choice(papers) for _ in papers) for r in by_paper[p]]
            st = stats(sample, cond)
            for k in boots:
                boots[k].append(st[k])
        ci = {k: [round(sorted(v)[int(.025 * B)], 4), round(sorted(v)[int(.975 * B) - 1], 4)] for k, v in boots.items()}
        out["v5"][cond] = {**{k: round(v, 4) if isinstance(v, float) else v for k, v in point.items()}, "ci95": ci}
    v3 = json.loads((RUN / "cnn_v5_vs_v3.json").read_text())["v3"]
    out["v3_point_estimates"] = {c: {"auroc": v3[c]["auroc"], "ai_recall": v3[c]["methods"]["ai_splice"]["flag_fraction"],
                                     "clean_fpr": v3[c]["clean_false_flag_rate"]} for c in ["realistic", "stress", "overall"]}
    print(json.dumps(out, indent=1))
    Path("results/paper_v5_stats.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2000)
