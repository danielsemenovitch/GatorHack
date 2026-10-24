"""Conservative v2 vs v3 decision after same expanded held-out-paper evaluation."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main():
    old = json.loads((RESULTS / "cnn_v2_expanded_test.json").read_text())
    new = json.loads((RESULTS / "cnn_metrics.json").read_text())
    comparable = old["recalibrated_on_expanded_clean_calibration_papers"]
    old_auc = float(comparable["auroc"])
    new_auc = float(new["image_auroc"])
    old_fpr = float(comparable["clean_false_flag_rate"])
    new_fpr = float(new["false_flag_rate_clean"])
    a = float(comparable["detect_by_method"].get("ai_splice", 0))
    b = float(new["detect_rate_by_method"].get("ai_splice", 0))
    # This is a cautious engineering suggestion, not statistical significance.
    recommend = (new_auc >= old_auc + 0.01
                 and new_fpr <= max(0.06, old_fpr + 0.015)
                 and b >= a)
    outcome = {
        "comparison": "original v2 and expanded-data v3 on identical held-out-paper benchmark",
        "baseline_v2_new_calibration": comparable,
        "v3": {
            "auroc": new_auc,
            "clean_false_flag_rate": new_fpr,
            "ai_splice_detection": b,
            "n_test_images": new["n_test_images"],
            "n_test_papers": new["n_test_papers"],
            "score_rule": new["score_rule"],
        },
        "delta_auc": round(new_auc - old_auc, 4),
        "delta_clean_false_flag_rate": round(new_fpr - old_fpr, 4),
        "delta_ai_splice_recall": round(b - a, 4),
        "recommend_promotion_for_further_validation": recommend,
        "note": "Do not auto-deploy; result is from synthetic edits with only two AI donor blots. "
                "AUROC and sensitivity differences require uncertainty estimates and a truly independent set "
                "before deployment. The held-out method entry in v3 is deliberately inherited from v1 "
                "when SKIP_HELDOUT=1, not a v3 result.",
    }
    (RESULTS / "cnn_expanded_comparison.json").write_text(json.dumps(outcome, indent=2))
    print("V2 VS V3", json.dumps(outcome, indent=2), flush=True)


if __name__ == "__main__":
    main()
