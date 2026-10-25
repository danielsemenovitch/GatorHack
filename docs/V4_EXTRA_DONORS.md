# V4.1 — Additional Synthetic AI Blot Donors

**Status:** Experiment only; not merged or deployed. The current application continues using v3.

## Source integrity

The user requested 2–3 additional **Gemini** blot sources. There is **no connected Gemini image-generation endpoint in this ChatGPT session**. Instead, three **OpenAI-generated** synthetic Western blot compositions were produced in three separate image-generation calls and cropped to exclude headings/labels:

| Image | Origin | Intended use |
|---|---|---|
| `data/ai_sources/openai_donor_g1_train.jpg` | OpenAI generation g1 | Training |
| `data/ai_sources/openai_donor_g2_train.jpg` | OpenAI generation g2 | Training |
| `data/ai_sources_holdout/openai_donor_g3_holdout.jpg` | OpenAI generation g3 | Held-out AI donor evaluation ONLY |

The original Gemini-generated blots in `data/ai_sources/` remain unchanged. The exact synthetic-image generation IDs, SHA-256 hashes, and role assignments are in `data/v4_ai_donor_manifest.json`. **Do not call the added files Gemini images.**

The images are fictional research-style band textures, not experimental measurements. Edits pasted into original published papers are synthetic research examples and are never evidence of wrongdoing by an author.

## Data-leakage protection

- Training data generation uses `btw.sources.ai_panels()`, which reads `data/ai_sources` only.
- The entire third generated source image is stored under `data/ai_sources_holdout`, which is not read by training.
- The benchmark builder explicitly checks both training JPEG hashes and the held-out JPEG hash and asserts the held-out file was not copied into the training directory.
- `scripts/eval_v4_extra_unseen_donor.py` generates new AI-splice examples **only after training** using the withheld OpenAI generation, on originally untouched Western blot panels from the paper-held-out test split.
- This is **leave-one-image-generation-out within the OpenAI family**, not leave-one-AI-generator-family-out: the training data also contains OpenAI donor images, so stronger performance is not guaranteed to generalize to previously unseen AI generators.

## Experiment and evaluation

[GitHub Actions V4.1 experiment run](https://github.com/danielsemenovitch/GatorHack/actions/runs/40213075931).

The complete workflow:
1. Checks donor SHA-256/provenance and split.
2. Rehydrates original papers and discovers additional CC BY full-text Western blot figures on Europe PMC.
3. Generates synthetic edit/mask pairs across paper-disjoint splits; uses pre-existing Gemini plus newly added OpenAI training donors for some AI splices.
4. Trains a **new 5-epoch experimental CNN** with an overall job limit of 170 minutes.
5. Measures v3 vs v4.1 on the same held-out-paper synthetic benchmark.
6. Measures native, 50%-resized, and JPEG-resaved AI splices **from the never-trained-on OpenAI g3 donor**, with clean calibration papers and measured FPR.
7. Saves source manifests, `models/splice_cnn_v4_extra_donors.npz`, `results/cnn_v4_extra_metrics.json`, `results/cnn_v4_extra_comparison.json`, and `results/v4_extra_unseen_ai_source_eval.json` in Actions artifacts / experimental branch only, if successful.

**No accuracy improvement may be claimed until successful metrics have been checked.** Synthetic benchmark performance may still overstate real-world detection. The test has one held-out OpenAI source generation, not enough to certify cross-generator generalization.

## Deployment decision

Never overwrite the committed v2/v3 checkpoints or change the live app simply because a new v4.1 exists. Compare AI-splice sensitivity, AUROC, 50%-resize recall, clean false flags *and* the new unseen-source test with the earlier versions and, ideally, several genuinely untouched papers. Make a promotion decision only with comparable evidence.

The longer, stronger research design would generate genuinely new **Gemini** donors through a connected Gemini API and also test completely unseen generator families and novel prompt distributions. This ChatGPT session did not have such an API connection.
