"""Compare 5-epoch and 8-epoch V4.1 trained from scratch on SAME data.

Never select hyperparameters on held-out test results. Both runs use the same
source images, synthetic benchmark rows, model initialization seed, and
paper-disjoint train/calibration/test partitions. The cosine schedule is
recomputed for 5 versus 8 epochs. This does NOT establish generalization to
real forgery cases or a new AI generator family.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load(name):
    return json.loads((RESULTS / name).read_text())


def extract(metric, cmp, unseen):
    held = unseen["v4_extra"]["results"]
    return {
        "epochs": metric.get("training_epochs"),
        "auroc": metric["image_auroc"],
        "ai_splice_recall": metric["detect_rate_by_method"]["ai_splice"],
        "clean_false_flag_rate": metric["false_flag_rate_clean"],
        "other_edit_recall": {k:v for k,v in metric["detect_rate_by_method"].items()
                             if k not in ("clean", "ai_splice")},
        "heldout_image_count": metric["n_test_images"],
        "heldout_papers": metric["n_test_papers"],
        "unseen_donor_native": held["native"],
        "unseen_donor_half_size": held["resize_50"],
        "unseen_donor_jpeg": held["jpeg_q75"],
        "matches_frozen_v3_test": cmp["v3_frozen_calibrated"]["n_test_images"]
                                  == metric["n_test_images"],
    }


def main():
    five_m = load("cnn_v4_5ep_metrics.json")
    eight_m = load("cnn_v4_8ep_metrics.json")
    five_c = load("cnn_v4_5ep_comparison.json")
    eight_c = load("cnn_v4_8ep_comparison.json")
    five_u = load("v4_5ep_unseen_ai_source_eval.json")
    eight_u = load("v4_8ep_unseen_ai_source_eval.json")

    five = extract(five_m, five_c, five_u)
    eight = extract(eight_m, eight_c, eight_u)
    assert five_m["n_test_images"] == eight_m["n_test_images"]
    assert five_m["n_test_papers"] == eight_m["n_test_papers"]
    frozen5 = five_c["v3_frozen_calibrated"]
    frozen8 = eight_c["v3_frozen_calibrated"]
    assert frozen5 == frozen8, "V3 baseline differs: cannot compare experiments"
    assert five_u["n_test_pairs"] == eight_u["n_test_pairs"]
    assert five_m["n_train_images"] == eight_m["n_train_images"]
    assert five_m["n_calib_papers"] == eight_m["n_calib_papers"]

    delta = {
        "auroc": round(eight["auroc"] - five["auroc"], 4),
        "ai_splice_recall": round(eight["ai_splice_recall"] - five["ai_splice_recall"], 4),
        "clean_false_flag_rate": round(eight["clean_false_flag_rate"] - five["clean_false_flag_rate"], 4),
        "unseen_native_recall": round(
            eight["unseen_donor_native"]["sensitivity"] - five["unseen_donor_native"]["sensitivity"], 4),
        "unseen_resized_recall": round(
            eight["unseen_donor_half_size"]["sensitivity"] - five["unseen_donor_half_size"]["sensitivity"], 4),
        "unseen_clean_fpr": round(
            eight["unseen_donor_native"]["false_flag_rate"] -
            five["unseen_donor_native"]["false_flag_rate"], 4),
    }
    # Predeclared exploratory gates; these are decision-support, not claims of
    # statistically significant improvement.
    gates = {
        "beat_five_epoch_auc_by_0p01": delta["auroc"] >= .01,
        "beat_five_epoch_ai_recall_by_0p02": delta["ai_splice_recall"] >= .02,
        "avoid_clean_fpr_increase_over_0p01": delta["clean_false_flag_rate"] <= .01,
        "beat_frozen_v3_auc": eight["auroc"] > frozen5["auroc"],
        "beat_frozen_v3_ai_recall": eight["ai_splice_recall"] >
                                     frozen5["detect_by_method"]["ai_splice"],
        "no_degradation_on_unseen_donor_native": delta["unseen_native_recall"] >= 0,
    }
    report = {
        "purpose": "Controlled training-duration ablation: 5 vs 8 epochs, same V4.1 dataset.",
        "model_training": "Same seed and architecture; cosine LR schedule spans specified epoch count. Neither run is a continuation of the other.",
        "data_split": "Paper-disjoint heldout/test; one completely heldout OpenAI image generation (not generator-family heldout).",
        "v3_frozen": frozen5,
        "v4_1_five_epochs": five,
        "v4_1_eight_epochs": eight,
        "eight_minus_five": delta,
        "exploratory_decision_gates": gates,
        "passes_all_gates": bool(all(gates.values())),
        "important_limitations": [
            "Training uses synthetic manipulations and two previously heldout Gemini donors plus two OpenAI training donor images.",
            "The independent donor evaluation uses ONE OpenAI source generation; 120 edits share that source, so uncertainty is understated by binomial CIs.",
            "The source collection can vary between runs. 5- and 8-epoch models in THIS single workflow share identical sources and examples.",
            "Five versus eight epochs also changes the duration of the cosine learning-rate schedule.",
            "Neither lower training loss nor lower clean false flags alone proves improved scientific image integrity detection.",
            "Do not promote a model to the live app unless paper-level false positives and heldout edit recall are independently validated.",
        ],
    }
    dst = RESULTS / "cnn_v4_1_5vs8_comparison.json"
    dst.write_text(json.dumps(report, indent=2))
    print("V4_1_EPOCH_ABLATION", json.dumps({
        "frozen_v3": {"auroc":frozen5["auroc"],
                      "ai_splice":frozen5["detect_by_method"]["ai_splice"],
                      "clean_fpr":frozen5["clean_false_flag_fraction"]},
        "five": {k:five[k] for k in ("auroc","ai_splice_recall","clean_false_flag_rate")},
        "eight": {k:eight[k] for k in ("auroc","ai_splice_recall","clean_false_flag_rate")},
        "delta": delta, "passes_all_gates": report["passes_all_gates"]
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
