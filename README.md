# Below the Watermark

**A scientific image integrity review assistant, with experimental Western blot forensics and an open AI-era blot benchmark.**

Built in 48 hours at Gator Hack V (UF AI Days 2026) for the theme "Fresh off the frontier."

## The finding

In the past year, Gemini's image tools made it trivial to fabricate or edit a western blot, and on October 7, 2026 Google opened **SynthID Detector** to the public as the safety net. We tested it on scientific images, by hand, one fixed scenario at a time:

| What was uploaded | SynthID Detector | Content credential (C2PA) |
|---|---|---|
| Whole AI blot: re-saved, resized, JPEG 60, cropped, contrast, in a figure, PDF round trip | detected, every time | stripped by any re-save |
| A single AI band on its own | "not enough details" | stripped |
| An AI band pasted into a published CC BY blot (4 figures) | **not detected, 4 of 4** ([results](results/synthid_manual.json)) | stripped |
| The untouched published figure (control) | not detected, as expected | none |

The watermark is robust when the AI image is present as a whole panel. Much scientific image manipulation happens at the scale of a single band, and there it either had too little to check or missed the band entirely.

## What we built

1. **An open benchmark.** <!-- BENCH_SIZE -->4,725 western blot images from 55 CC BY papers<!-- /BENCH_SIZE -->, clean or forged five ways (copy move, cross splice, AI splice, intensify, erase), each pushed through realistic publishing steps, each with a pixel mask. Split by paper. See [docs/DATA_CARD.md](docs/DATA_CARD.md).
2. **A band splice detector**, trained and tested on that benchmark: a small CNN and a gradient boosted baseline on hand built forensic features, both judging each band against the rest of its own blot.
3. **A scientific image review app** accepting figure images or paper PDFs. It inventories raster figures, checks for exact and near whole-image reuse within the file, shows possible matches side by side, and exports an HTML evidence report. For confidently identified blots only, it also offers the experimental band CNN and copied-band checks; standalone images can have their C2PA credential read.

## Duplication inside and across papers (October 25, 2026)

- **Within a paper:** every pair of figures is compared region by region (`btw/region_match.py`: SIFT keypoints on image content only, one to one matching, RANSAC similarity transform, then the aligned content must correlate at 0.90 or more), so a blot reused under a different label still lines up after cropping, resizing, re-saving or mirroring.
- **Against published figures:** every upload is checked against a reference library (the CC BY figures in `data/real_figures`, 112 figures from 62 papers; the four papers used as demo uploads are left out). It is built on first use and cached in `.cache/`.
- **The search agent** (`btw/search_agent.py`, button "Search Europe PMC" in a report): plans Europe PMC searches from the manuscript's title, abstract and captions (Gemini when `GEMINI_API_KEY` is set, keyword rules otherwise; for a single image Gemini reads the figure's labels), downloads the figures of the open access papers it finds, and runs the same matcher. Every query and paper it checked is returned. Put the key in a git-ignored `.env` file in the repo root (`GEMINI_API_KEY=...`); never commit it.
- **Measured:** a blot panel or micrograph block (25 to 55% of a published figure) reused in a figure from another paper, resized, re-saved, cropped or mirrored, was found 110 of 118 times (`--blocks`, `results/region_match_eval_blocks.json`); a single small strip alone 60 of 120 (`results/region_match_eval.json`); none of the 7,140 pairs of untouched library figures matched (`--fp`). Matching panels against each other inside one figure produced many false matches on shared labels, so it is not used. Without the pixel check, figures of one paper drawn from the same chart template matched each other.
- **Demo:** "A manuscript with two hidden duplications" (`scripts/make_demo_pdf.py`) hides a blot reused as another protein and a figure lifted from a 2023 paper; its last page states the sources and our edits.

## Independent audit update (October 24, 2026, evening)

See [AUDIT_REPORT.md](AUDIT_REPORT.md) for the full audit. The changes that affect results:

- **Resolution mismatch fixed.** CNN v3 learned from web figures about 750 px wide; publisher PDFs embed figures at 1,000 to 2,000+ px. Scoring at full resolution inflated flags on unedited blots (on held-out test papers, enlarging clean panels 2.5x raised clean false flags from 7% to 18% at the research threshold). Blot panels from large figures are now scored on a copy resized to training scale. On the published Oncotarget paper PDF we use for testing (DOI 10.18632/oncotarget.23647) this changes the research threshold count from 18 of 36 screened panels to 0 of 36, and the conservative count from 2 to 0. That paper is unlabeled, so this shows the scores were tracking resolution, not that the paper is clean. With the later frame-aware panel finder (below) the same PDF screens 30 panels and flags 2, both at 0.512, just over the 0.50 line; at this threshold about 1 in 20 unedited panels is flagged by chance, so 1 to 2 flags are expected.
- **The research threshold (now called Standard) is the default.** The old 0.90 line caught about 4% of pasted AI bands on held-out test papers. Since Oct 25 the optional Strict line is 0.60, chosen on calibration papers only (at most 1% of clean calibration panels flagged): on 20 held-out papers it catches 24% of pasted AI bands and flags 2.5% of clean panels (`scripts/calibrate_strict.py`, `results/strict_line.json`). End to end on 22 held-out whole figures with one pasted AI band each, the research threshold flagged the edited panel in 7, flagged no untouched panel in any edited figure, and flagged 1 of the 22 unedited originals (`results/figure_level_eval.json`).
- **Frame-aware panels:** published frame lines are trimmed at any resolution, a band inside a framed strip is no longer screened as its own panel, arrows and labels touching a frame are cut off, and framed tables are not routed. On the bundle 1 figures this keeps 240 of 241 reference panels and the same automatic routing coverage (130 of 241).
- **Bugs fixed:** PDFs with grayscale embedded images no longer crash; a journal logo on every page is no longer reported as image reuse; oversized and unreadable images are listed instead of silently dropped; C2PA text found by byte search is no longer reported as "signed by Google"; heat is drawn only on flagged panels; the demo example now shows where we actually edited it.

## Conservative screening and heat-map update (October 24, 2026, afternoon; superseded where the audit update above differs)

**Default review mode now favors fewer alerts:** CNN v3 uses a conservative
review index threshold of **0.90**, rather than its synthetic-benchmark
operating threshold of **0.50**. Both settings are selectable in the upload
form; API clients may pass `screening_policy=conservative|benchmark`.
For the same uploaded Oncotarget PDF, this changes the count from
**18/37** to **2/37** candidates based on the original v3 score list
(before the added microscopy route filter). **This does not establish a
reliable false-positive rate** and will miss more true edits. The original
v2 CNN uses legacy scoring because its outputs saturate near 1.0.

Automatic blot routing now checks for horizontally oriented blot textures
and excludes a confirmed grayscale-microscopy crop from the user's test
article. Heat maps now display **the patches eligible for panel scoring**,
with thresholding before smoothing, and annotate their limitations.
Even after the fix, heat represents overlapping **32-pixel patch scores**,
not exact pasted-band boundaries or certified image manipulation.

The whole-Gemini example image is NOT automatically accepted as a blot by
the current router. A forced **geometry-only debug** on that image showed
patch heat around molecular-weight markings; do not use it as proof of
localized AI alteration. See
[Actions run 40139731885](https://github.com/danielsemenovitch/GatorHack/actions/runs/40139731885)
for the rendered side-by-side heat audit, and see `HANDOFF.md` for
the full details and V4 research plan.

## CNN v3 and reliable-reporting update (October 24, 2026)

The default **experimental** Western blot CNN is now **v3**, trained with figures
reconstructed from 241 additional CC-BY-labeled article IDs. Users can switch
back to the original **v2** in the upload form. The original checkpoint
(`models/splice_cnn.npz`) remains unchanged; the new checkpoint is at
`models/splice_cnn_v3.npz`. Both models are experimental; neither proves
manipulation or research misconduct.

**Matched expanded held-out benchmark** (2,761 images, 72 held-out papers,
same evaluation data for both models):

| Metric | V2 (recalibrated) | V3 |
|---|---:|---:|
| AUROC | 0.652 | 0.703 |
| AI-band splice detection | 26.4% | 35.4% |
| Clean image false flags | 3.65% | 4.10% |

V3 still detected only **5.1%** of AI splices after resizing to 50%.
Its clean false-positive rate reached **12.7%** after contrast adjustment.
The additional source figures were reconstructed from Europe PMC and were
not checked to be byte-for-byte identical with the uploaded dataset.
See `results/cnn_v3_metrics.json`, `results/cnn_expanded_comparison.json`,
`results/cnn_v2_expanded_test.json`, `results/cnn_v3_robustness.json`.

**Review reliability fixes:** precise anomaly indices (no misleading "100/100"
certainty), experimental model version shown in reports, evidence crops on
both sides of a copied-band candidate, no fictitious "band 0" label, 64
candidate blot panels per PDF by default with explicit skipped counts, and
warnings when many flags come from a single figure.

The app uses a more conservative noise/shape threshold for copied-band
candidates. On an in-sample 204-pair synthetic check it flagged **1.96%**
of untouched controls but detected only **6.86%** of created copies. Treat
this feature as **low recall** and never infer no duplication from a negative
finding. `results/duplicates_conservative_sanity.json` holds the new check.

An unlabeled CC BY journal article was also re-analyzed from public
original **figure JPEGs**; results are not directly comparable with its
journal PDF rendering, and findings do not imply any author misconduct.

## Scientific image integrity review (new MVP)

The app now supports **general whole-image reuse screening** when a paper PDF
contains multiple embedded raster figures. It finds:

- **Exact pixel reuse:** identical decoded RGB image pixels, even when the
  image's container metadata differs.
- **Near-duplicate images:** image pairs whose perceptual hashes are similar
  and whose resized pixels pass a stringent structural-similarity check.
  Example: some JPEG-resaved or resized versions of the same image.

It shows **both source images together** for a human to compare and can
download a self-contained, printable **HTML integrity review report**.

These are **whole-image**, within-upload checks. They do *not* find arbitrary
copied patches inside microscopy images, search other publications, or prove
research misconduct. Reusing an image can be legitimate.

For figures that cannot be confidently identified as Western blots, the
**blot-specific CNN is NOT run**. Only blot-like panels receive experimental
band screening, and the interface says when that check was skipped. Provenance
checks on PDF-embedded images remain not assessed; a missing C2PA credential
is inconclusive.

See `tests/test_integrity_mvp.py` for synthetic control cases and
`HANDOFF.md` for validation status, limitations and remaining work.

## Results

Numbers are written by `python -m scripts.results_table` from `results/*.json`, on held out test papers the models never saw. The research flag line is set so 5% of clean panels from separate calibration papers are flagged. All edits are simulated.

<!-- RESULTS_TABLE -->

**Band splice detector, CNN v3 (the app's default): 72 held out test papers, 2,761 test images, Standard line set on separate calibration papers**

| | |
|---|---|
| AUROC, forged vs clean panels | 0.703 |
| Clean panels flagged | 4% |
| copy move caught | 23% |
| cross splice caught | 31% |
| ai splice caught | 35% |
| intensify caught | 49% |
| erase caught | 20% |
| Strict line (0.60, set on calibration papers), on 20 held-out test papers | 24% of AI splices caught, 2% of clean panels flagged |
| Legacy CNN v2 on its own smaller test set (20 papers) | AUROC 0.643, 22% of AI splices caught, 3% of clean panels flagged |
| Baseline: gradient boosting on hand built features (earlier benchmark), AUROC | 0.543 |

**One publishing step after pasting an AI band** (CNN v3, Standard line; same steps as the SynthID test)

| Step | Pasted AI bands caught | Clean panels flagged |
|---|---|---|
| None (straight after the edit) | 47% | 3% |
| Re-saved | 47% | 3% |
| Resized to 50% | 5% | 1% |
| JPEG quality 85 | 45% | 6% |
| JPEG quality 60 | 42% | 6% |
| Contrast adjusted | 53% | 13% |
| PDF round trip | 44% | 4% |

**Whole figures, end to end** (routing + CNN v3 exactly as the app runs; one Gemini band pasted into one panel of each of 22 held-out figures from 15 papers; `results/figure_level_eval.json`)

| | Standard line (default) | Strict line |
|---|---|---|
| Edited panel flagged | 7 of 22 | 4 of 22 |
| Edited figures where an untouched panel was flagged | 0 of 22 | 0 of 22 |
| Unedited figures with any flag | 1 of 22 | 0 of 22 |

**Figure resolution** (CNN v3, Standard line 0.50; `results/audit_resolution.json`)

| Input | AUROC | Clean panels flagged | AI splices caught |
|---|---|---|---|
| Web resolution, as trained | 0.74 | 7% | 38% |
| Same panels enlarged 2.5x (like a high resolution PDF figure) | 0.622 | 18% | 33% |
| Enlarged, then resized back to training scale (what the app now does) | 0.701 | 3% | 21% |

**Copied band detector** (exact pixel copies within a test panel)

| Step | Copies caught | Clean panels flagged |
|---|---|---|
| No publishing step | 11% | 5% |
| JPEG quality 85 | 2% | 2% |
| PDF round trip | 2% | 1% |
| Resized to 50% | 2% | 2% |
| The app's strict setting (noise and shape 0.90), 204 in-sample pairs, no publishing step | 6.9% | 2.0% |

## Run it

```bash
# backend (Python 3.11+)
pip install -r requirements.txt
pip install "jax[cpu]" optax        # only needed for the CNN detector
uvicorn btw.api:app --port 8000

# frontend
cd web && npm install && npm run dev   # http://localhost:5173, proxies /api to :8000
# or build once and let FastAPI serve it:  npm run build  →  http://localhost:8000
```

Rebuild everything from scratch:

```bash
python -m scripts.build_benchmark --bundle btw_real_blots_bundle.json   # figures → panels → benchmark
python -m scripts.train_detector      # gradient boosted baseline → results/detector_metrics.json
python -m scripts.train_cnn           # CNN → results/cnn_metrics.json
python -m scripts.eval_robustness     # one publishing step at a time → results/robustness.json
python -m scripts.eval_duplicates     # copied band detector → results/duplicates_metrics.json
python -m scripts.make_examples       # demo reports for the app
```

On a laptop with a Gemini API key (`laptop/`): `gemini_blot_library.py` grows the AI donor pool, and `claim_link.py` lists the sentences in a paper that rely on a flagged figure.

## Repository

```
btw/          library: panels, bands, forgeries, publishing steps, features, detectors, duplicates, C2PA, report, API
scripts/      benchmark building, training and evaluation
laptop/       Gemini API scripts (run locally)
web/          React frontend
results/      every number we report
docs/         data card, methods, disclosure note
examples/     demo inputs (CC BY figure with an AI band WE added; the published figure is unaltered)
```

## Limits

- A flag means look closer. It is never proof of misconduct.
- The forgeries are simulated; a careful forger can get past this detector.
- The blot model was trained on web resolution figures (about 750 px wide), where bands can be only about 13 × 6 pixels. Larger figures are scored on a copy resized to that scale, so edits made at full resolution may be harder to detect; a model trained on full resolution figures is future work.
- The SynthID results are a small, hand-run case study. A disclosure draft exists; external submission is not confirmed.
- Experimental copied-band detection has low detection rates even at native resolution and degrades further after resizing or JPEG.
- Automatic blot recognition is conservative; users can request blot-specific screening for images they identify as blots or skip it. Neither setting validates authenticity.

## Ethics and terms

- SynthID Detector was used by hand, within its terms: no automated submissions, no iterative attempts to evade it.
- All blot figures are CC BY, with full attribution in `data/real_figures/metadata.json`. Every forgery in this repository was made by us. **No original author is accused of anything.**
- No unpublished lab data is included.

## How it was built

Built by our team with Claude (Anthropic) as a coding assistant. The team designed the experiment, ran every SynthID test by hand, and reviewed the code and results.

## License

Code: MIT. Benchmark images: derivatives of CC BY 4.0 figures, so attribution is required (see the metadata file).
