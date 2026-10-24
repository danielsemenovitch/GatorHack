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
    readme = re.sub(r"<!-- RESULTS_TABLE -->.*?(?=\n## )", "<!-- RESULTS_TABLE -->\n\n" + table + "\n", readme, flags=re.S)
    (ROOT / "README.md").write_text(readme)
    print(table)


if __name__ == "__main__":
    main()
