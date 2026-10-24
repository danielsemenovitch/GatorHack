# Below the Watermark

**A scientific image integrity review assistant, with experimental Western blot forensics and an open AI-era blot benchmark.**

Built in 48 hours at Gator Hack V (UF AI Days 2026) for the theme "Fresh off the frontier."

## The finding

In the past year, Gemini's image tools made it trivial to fabricate or edit a western blot, and on October 7, 2026 Google opened **SynthID Detector** to the public as the safety net. We tested it on scientific images, by hand, one fixed scenario at a time:

| What was uploaded | SynthID Detector | Content credential (C2PA) |
|---|---|---|
| Whole AI blot: re-saved, resized, JPEG 60, cropped, contrast, in a figure, PDF round trip | detected, every time | stripped by any re-save |
| A single AI band on its own | "not enough details" | stripped |
| An AI band pasted into a published blot | see [results](results/synthid_manual.json) | stripped |

The watermark is robust when the AI image is present as a whole panel. Scientific image fraud happens one band at a time.

## What we built

1. **An open benchmark.** <!-- BENCH_SIZE -->4,725 western blot images from 55 CC BY papers<!-- /BENCH_SIZE -->, clean or forged five ways (copy move, cross splice, AI splice, intensify, erase), each pushed through realistic publishing steps, each with a pixel mask. Split by paper. See [docs/DATA_CARD.md](docs/DATA_CARD.md).
2. **A band splice detector**, trained and tested on that benchmark: a small CNN and a gradient boosted baseline on hand built forensic features, both judging each band against the rest of its own blot.
3. **A scientific image review app** accepting figure images or paper PDFs. It inventories raster figures, checks for exact and near whole-image reuse within the file, shows possible matches side by side, and exports an HTML evidence report. For confidently identified blots only, it also offers the experimental band CNN and copied-band checks; standalone images can have their C2PA credential read.

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

Numbers are from `results/*.json`, on 20 held out test papers the models never saw. The flag line is set on training data (5% of clean training panels flagged).

<!-- RESULTS_TABLE -->

**Band splice detector (CNN), 20 test papers, 2071 test images**

| | |
|---|---|
| AUROC, forged vs clean panels | 0.643 |
| Clean panels flagged | 3% |
| copy move caught | 16% |
| cross splice caught | 19% |
| ai splice caught | 22% |
| intensify caught | 38% |
| erase caught | 2% |
| AI splice caught with NO AI splices in training (earlier v1 experiment; not rerun for v2) | 38% at 8% false flags (AUROC 0.717) |
| Baseline: gradient boosting on hand built features, AUROC | 0.543 |

**One publishing step after pasting an AI band** (same steps as the SynthID test)

| Step | Pasted AI bands caught | Clean panels flagged |
|---|---|---|
| None (straight after the edit) | 38% | 4% |
| Re-saved | 38% | 4% |
| Resized to 50% | 2% | 0% |
| JPEG quality 85 | 39% | 4% |
| JPEG quality 60 | 36% | 2% |
| Contrast adjusted | 38% | 6% |
| PDF round trip | 39% | 3% |

**Copied band detector** (exact pixel copies within a test panel)

| Step | Copies caught | Clean panels flagged |
|---|---|---|
| No publishing step | 11% | 5% |
| JPEG quality 85 | 2% | 2% |
| PDF round trip | 2% | 1% |
| Resized to 50% | 2% | 2% |

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
- Figures are at web resolution, where bands can be only about 13 × 6 pixels. Full resolution originals carry more evidence.
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
