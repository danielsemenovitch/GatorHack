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


def cnn_table(cnn, title):
    lines = [title, "", "| | |", "|---|---|",
             f"| AUROC, forged vs clean panels | {cnn['image_auroc']} |",
             f"| Clean panels flagged | {pct(cnn['false_flag_rate_clean'])} |"]
    for k, v in cnn["detect_rate_by_method"].items():
        if k != "clean":
            lines.append(f"| {k.replace('_', ' ')} caught | {pct(v)} |")
    ho = (cnn.get("held_out_method") or {}).get("ai_splice")
    if ho and not ho.get("note"):  # a copied v1 number must not be shown as this model's
        lines.append(f"| AI splice caught with NO AI splices in training | {pct(ho['detect_rate'])} at {pct(ho['false_flag_rate'])} false flags (AUROC {ho['auroc']}) |")
    return lines


def main():
    v3, v2, gbm, dup, audit = (load("cnn_v3_metrics.json"), load("cnn_metrics.json"), load("detector_metrics.json"),
                               load("duplicates_metrics.json"), load("audit_resolution.json"))
    rob = load("cnn_v3_robustness.json") if v3 else load("robustness.json")
    lines = []
    cnn = v3 or v2
    if cnn:
        name = "CNN v3 (the app's default)" if v3 else "CNN v2"
        lines += cnn_table(cnn, f"**Band splice detector, {name}: {cnn['n_test_papers']} held out test papers, "
                                f"{cnn['n_test_images']:,} test images, research threshold set on separate calibration papers**")
        if audit and v3:
            op = audit["benchmark"]["native"]
            lines.append(f"| At the optional strict line (0.90), on {audit['benchmark']['n_test_papers']} of the test papers | "
                         f"{pct(op['ai_splice_caught@0.9'])} of AI splices caught, {pct(op['clean_flagged@0.9'])} of clean panels flagged |")
        if v3 and v2:
            lines.append(f"| Legacy CNN v2 on its own smaller test set ({v2['n_test_papers']} papers) | AUROC {v2['image_auroc']}, "
                         f"{pct(v2['detect_rate_by_method']['ai_splice'])} of AI splices caught, {pct(v2['false_flag_rate_clean'])} of clean panels flagged |")
        if gbm:
            lines.append(f"| Baseline: gradient boosting on hand built features (earlier benchmark), AUROC | {gbm['image_auroc']} |")
        lines.append("")
    if rob:
        lines += [f"**One publishing step after pasting an AI band** ({'CNN v3' if v3 else 'CNN v2'}, research threshold; same steps as the SynthID test)", "",
                  "| Step | Pasted AI bands caught | Clean panels flagged |", "|---|---|---|"]
        lines += [f"| {r['label']} | {pct(r['detect_rate'])} | {pct(r['false_flag_rate'])} |" for r in rob["rows"]]
        lines.append("")
    fl = load("figure_level_eval.json")
    if fl:
        r, c = fl["summary"]["benchmark"], fl["summary"]["conservative"]
        lines += [f"**Whole figures, end to end** (routing + CNN v3 exactly as the app runs; one Gemini band pasted into one panel of each of "
                  f"{r['figures']} held-out figures from {fl['papers']} papers; `results/figure_level_eval.json`)", "",
                  "| | Research threshold (default) | Strict line |", "|---|---|---|",
                  f"| Edited panel flagged | {r['edited_panel_flagged']} of {r['figures']} | {c['edited_panel_flagged']} of {c['figures']} |",
                  f"| Edited figures where an untouched panel was flagged | {r['edited_figure_with_untouched_panel_flagged']} of {r['figures']} | {c['edited_figure_with_untouched_panel_flagged']} of {c['figures']} |",
                  f"| Unedited figures with any flag | {r['original_figure_with_any_flag']} of {r['figures']} | {c['original_figure_with_any_flag']} of {c['figures']} |",
                  ""]
    if audit and v3:
        b = audit["benchmark"]
        lines += ["**Figure resolution** (CNN v3, research threshold 0.50; `results/audit_resolution.json`)", "",
                  "| Input | AUROC | Clean panels flagged | AI splices caught |", "|---|---|---|---|",
                  f"| Web resolution, as trained | {b['native']['auroc']} | {pct(b['native']['clean_flagged@0.5'])} | {pct(b['native']['ai_splice_caught@0.5'])} |",
                  f"| Same panels enlarged 2.5x (like a high resolution PDF figure) | {b['up2.5']['auroc']} | {pct(b['up2.5']['clean_flagged@0.5'])} | {pct(b['up2.5']['ai_splice_caught@0.5'])} |",
                  f"| Enlarged, then resized back to training scale (what the app now does) | {b['up2.5->norm']['auroc']} | {pct(b['up2.5->norm']['clean_flagged@0.5'])} | {pct(b['up2.5->norm']['ai_splice_caught@0.5'])} |",
                  ""]
    if dup:
        lines += ["**Copied band detector** (exact pixel copies within a test panel)", "",
                  "| Step | Copies caught | Clean panels flagged |", "|---|---|---|"]
        lines += [f"| {r['label']} | {pct(r['detect_rate'])} | {pct(r['false_flag_rate'])} |" for r in dup["rows"]]
        strict = load("duplicates_conservative_sanity.json")
        if strict:
            lines.append(f"| The app's strict setting (noise and shape 0.90), {strict['n_pairs']} in-sample pairs, no publishing step | "
                         f"{strict['detect_rate_synthetic_copy'] * 100:.1f}% | {strict['false_flag_rate_clean'] * 100:.1f}% |")
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
