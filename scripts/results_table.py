"""Write the README results table from results/*.json (never by hand)."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    f = ROOT / "results" / name
    return json.loads(f.read_text()) if f.exists() else None


def pct(v):
    return f"{round(v * 100)}%"


def main():
    cnn, gbm, rob, dup = (load("cnn_metrics.json"), load("detector_metrics.json"),
                          load("robustness.json"), load("duplicates_metrics.json"))
    lines = []
    if cnn:
        lines += [
            f"**Band splice detector (CNN), {cnn['n_test_papers']} test papers, {cnn['n_test_images']} test images**",
            "",
            "| | |", "|---|---|",
            f"| AUROC, forged vs clean panels | {cnn['image_auroc']} |",
            f"| Clean panels flagged | {pct(cnn['false_flag_rate_clean'])} |",
        ]
        for k, v in cnn["detect_rate_by_method"].items():
            if k != "clean":
                lines.append(f"| {k.replace('_', ' ')} caught | {pct(v)} |")
        ho = (cnn.get("held_out_method") or {}).get("ai_splice")
        if ho:
            lines.append(f"| AI splice caught with NO AI splices in training | {pct(ho['detect_rate'])} at {pct(ho['false_flag_rate'])} false flags (AUROC {ho['auroc']}) |")
        if gbm:
            lines.append(f"| Baseline: gradient boosting on hand built features, AUROC | {gbm['image_auroc']} |")
        lines.append("")
    if rob:
        lines += ["**One publishing step after pasting an AI band** (same steps as the SynthID test)", "",
                  "| Step | Pasted AI bands caught | Clean panels flagged |", "|---|---|---|"]
        lines += [f"| {r['label']} | {pct(r['detect_rate'])} | {pct(r['false_flag_rate'])} |" for r in rob["rows"]]
        lines.append("")
    if dup:
        lines += ["**Copied band detector** (exact pixel copies within a test panel)", "",
                  "| Step | Copies caught | Clean panels flagged |", "|---|---|---|"]
        lines += [f"| {r['label']} | {pct(r['detect_rate'])} | {pct(r['false_flag_rate'])} |" for r in dup["rows"]]
    table = "\n".join(lines)
    readme = (ROOT / "README.md").read_text()
    readme = re.sub(r"<!-- RESULTS_TABLE -->.*?(?=\n## )", lambda m: "<!-- RESULTS_TABLE -->\n\n" + table + "\n", readme, flags=re.S)
    stats = benchmark_stats()
    if stats:
        size = f"{stats['n']:,} western blot images from {stats['papers']} CC BY papers"
        readme = re.sub(r"<!-- BENCH_SIZE -->.*?<!-- /BENCH_SIZE -->", lambda m: f"<!-- BENCH_SIZE -->{size}<!-- /BENCH_SIZE -->", readme)
        card = ROOT / "docs" / "DATA_CARD.md"
        s = card.read_text()
        s = re.sub(r"<!-- DATA_TABLE -->.*?<!-- /DATA_TABLE -->",
                   lambda m: "<!-- DATA_TABLE -->\n" + stats["table"] + "\n<!-- /DATA_TABLE -->", s, flags=re.S)
        card.write_text(s)
    (ROOT / "README.md").write_text(readme)
    print(table)


def benchmark_stats():
    """Data card numbers straight from data/benchmark/index.json."""
    import numpy as np
    f = ROOT / "data" / "benchmark" / "index.json"
    if not f.exists():
        return None
    ix = json.loads(f.read_text())
    n, nf = len(ix), sum(r["label"] for r in ix)
    papers = {r["pmcid"] for r in ix}
    by = {s: [r for r in ix if r["split"] == s] for s in ("train", "test")}
    meta = json.loads((ROOT / "data/real_figures/metadata.json").read_text())
    years = sorted(int(m["year"]) for m in meta if str(m.get("year", "")).isdigit() and m.get("pmcid") in papers)
    clean = [r for r in ix if r["label"] == 0]
    med = int(np.median([r["w"] * r["h"] for r in clean])) if clean else 0
    rows = [
        ("Images", f"{n:,} ({n - nf:,} clean, {nf:,} forged)"),
        ("Source panels", f"{len({r['panel'] for r in ix})} blot panels"),
        ("Source papers", f"{len(papers)} open access papers (CC BY, published {years[0]} to {years[-1]}), via Europe PMC" if years else f"{len(papers)} papers"),
        ("Split", f"by paper: {len({r['pmcid'] for r in by['train']})} papers train ({len(by['train']):,} images), "
                  f"{len({r['pmcid'] for r in by['test']})} papers test ({len(by['test']):,} images)"),
        ("Median panel size", f"about {med:,} pixels"),
        ("Forgery types", "copy move, cross splice, AI splice, intensify, erase"),
        ("Publishing steps", "random chain of contrast, resize, JPEG (q70 to 95) or PDF round trip"),
    ]
    table = "| | |\n|---|---|\n" + "\n".join(f"| {a} | {b} |" for a, b in rows)
    return {"n": n, "papers": len(papers), "table": table}


if __name__ == "__main__":
    main()
