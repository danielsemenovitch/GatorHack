# Below the Watermark

**An image integrity checker for scientific papers. It finds what AI watermarks can't: images reused inside a paper, figures lifted from other papers, and pasted-in bands.**

Built for Gator Hack V / UF AI Days 2026, “Fresh off the frontier.”

## What it does now

Upload a PNG, JPEG, or research-paper PDF (up to 25 MB). The FastAPI + React application:

1. Extracts embedded raster figures and records skipped or unsupported content.
2. Finds **whole-image reuse** within the upload using exact pixel checks and perceptual/structural comparisons.
3. Finds **reused regions between different figures** with SIFT keypoints, one-to-one matching, geometric alignment, and pixel agreement; resized, mirrored, cropped or JPEG-compressed copies can be recognized.
4. Compares images with a built-in reference library of **112 published figures from 62 openly licensed papers**. The reference library is a sample, not a comprehensive publication database.
5. On request, a **search agent** (“Search Europe PMC”) looks for the figures in other published papers: Gemini reads the title, abstract and captions and plans Europe PMC searches (keyword rules if no `GEMINI_API_KEY` is set), the agent downloads the figures of the open-access papers it finds, and the region matcher checks each one. The UI and downloaded report list every query and paper checked.
6. Reads C2PA Content Credentials on supported original image uploads. **PDF-embedded C2PA is not assessed; SynthID must be checked manually**.
7. For identified Western blot panels only, provides **experimental** CNN band-anomaly scoring and low-recall copied-band checks. Neither establishes authenticity or wrongdoing.
8. Produces a standalone downloadable HTML evidence report, including matched regions and their published-source references. If an online search was run before downloading, its completed query log and matches are included too.

**Human review required.** Matching pixels is not proof of misconduct or publication priority. A negative result does not mean the image is authentic. Duplications *within one individual composite figure* remain out of scope for the region matcher.

## Key reproducible results

| Evaluation | Result | Limits |
|---|---|---|
| Larger copied blocks (blot panels / micrographs, about **25–55% of a source figure**) pasted into a different paper's figure | **110/118 detected (93.2%)** | Synthetic controlled reuse with resize, JPEG, crop and mirror transforms. See `results/region_match_eval_blocks.json` and `python -m scripts.eval_region_match --blocks` |
| Individual small blot strips/panels pasted into another figure | **60/120 detected (50%)** | Smaller or uniform strips are more difficult; `results/region_match_eval.json` |
| Untouched reference-library image pairs | **0/7,140 false matches** | Only this evaluated set, **not** a universal false-positive rate |
| Western blot CNN v3, paper-disjoint synthetic held-out test | AUROC **0.703**, AI-band splices **35.4%** caught, **4.1%** clean flags | 2,761 images from 72 papers; experimental and weak after 50% resizing (~5%). See `results/cnn_v3_metrics.json` |
| Actual full-figure edited-input workflow (v3) | **7/22** edited panels flagged; **1/22** untouched original figures flagged | Controlled synthetic AI-band edits; `results/figure_level_eval.json` |
| Manually tested SynthID Detector | **10/10** whole AI blots detected; **4/4** pasted AI bands not detected | Small, fixed-scenario case study; `results/synthid_manual.json` |

<!-- V5: replace this paragraph and the CNN rows above when V5 is promoted. -->
V4.1's controlled eight-epoch experiment has now **finished**: AI-splice recall **23.8%** on its newer shared held-out benchmark (v3 on the same set: **22.74%**), with AUROC **0.640** versus v3 **0.651** and more clean false flags (**4.8%** versus **3.68%**). This is **below the 40% target** and does not warrant promotion. Keep v3 deployed and treat all blot CNNs as experimental. See [recorded comparison](results/cnn_v4_1_8epoch_summary.json). Do not compare this 23.8% directly with v3's earlier 35.4% on a different benchmark.

## Demo

Click **“A manuscript with two hidden duplications.”** We intentionally created a controlled PDF from CC BY figures, with a blot strip reused under another label and a separate figure region sourced from a 2023 paper. The first and last pages disclose our modifications; original scientific authors are not implicated. The app compares the regions directly and against the reference library.

Click **“Search Europe PMC”** to run the agent. In a live run on Oct 25, **Gemini planned the searches and the agent found the source of the lifted figure (PMC10701888, 2023) after checking 4 papers**; keyword planning needed 16 papers and 55 figures ([saved keyword run](results/europepmc_live_smoke.json)). Single demonstrations, not a recall estimate. It needs internet access; test it on the presentation laptop first.

The **“A pasted AI band the model catches”** example is intentionally a *selected* positive, not representative of the CNN's average performance. The landing-page ALK2 edit illustrates a SynthID miss, **not** a CNN success.

## Run locally

Python 3.11+ and Node.js 20+:

```bash
python -m pip install -r requirements.txt
python -m pip install "jax[cpu]" optax
python -m uvicorn btw.api:app --port 8000
```

In another terminal:

```bash
cd web
npm ci
npm run dev
```

Open http://localhost:5173. For the single-origin production build, run `cd web && npm run build` and open http://localhost:8000.

**Gemini planning:** save `GEMINI_API_KEY=...` in a `.env` file in the repository root (git-ignored) and restart the backend. The agent tries `gemini-flash-latest`, then `gemini-flash-lite-latest`, then `gemini-3-flash-preview`; set `GEMINI_TEXT_MODEL=...` in `.env` to choose another. Without a key it plans with keyword rules. Never commit the key.

**Newer CNN:** save it as `models/splice_cnn_vN.npz` (same format as v3), run `python -m scripts.calibrate_strict --model vN`, then set `BTW_MODEL=vN` in `.env`.

**Checks:**

```bash
python -m pytest -q tests
cd web && npm ci && node --test src/reportExport.test.js && npm run build
```

## Project layout and evidence

- `btw/region_match.py` — within-upload and published-library image-region matcher
- `btw/search_agent.py` — optional Europe PMC search and Gemini/keyword planning
- `btw/report.py`, `btw/api.py` — PDF extraction and analysis API
- `web/src/App.jsx`, `web/src/reportExport.js` — web review and standalone HTML export
- `scripts/eval_region_match.py` — both duplication evaluations, including `--blocks` and `--fp`
- `scripts/make_demo_pdf.py`, `scripts/make_examples.py` — rebuild the demo manuscript and example reports
- `scripts/build_benchmark.py`, `train_cnn.py`, `eval_*.py`, `calibrate_strict.py` — benchmark, CNN training and its evaluations
- `laptop/gemini_blot_library.py` — generates the AI donor blots (`data/ai_sources/`) with a Gemini API key
- `docs/METHODS.md`, `docs/DATA_CARD.md` — reproducible methodology and licensing
- `docs/devpost.md`, `docs/video_script.md`, `docs/pitch_outline.md` — up-to-date submission materials
- `docs/AUDIT_REPORT.md` — **historical** independent audit of earlier versions (not current product specification)
- `HANDOFF.md` — current engineering handoff
- `results/` — measured data; `examples/` — source-attributed team-made demonstrations

## Limitations and ethics

The region matcher misses some small, uniform, or heavily transformed regions, and it does not compare regions inside a single figure. Neither the local reference library nor the Europe PMC agent searches all scientific publications. On-demand search can return zero results or be offline; downloaded reports explicitly distinguish “not searched” from “searched, no matches.”

The blot CNN is experimental, trained on synthetic manipulations, and particularly weak after resizing. A heat overlay shows coarse 32-pixel patches, not a confirmed manipulation boundary. Missing provenance or no flags are not proof of authenticity.

Original scientific figures used here are CC BY and attributed in `data/real_figures/metadata.json`. **Every edit displayed in a demo was made by our team**; no source author is accused. The SynthID observations are manual experiments, with no automated probing; disclosure to Google is drafted but not confirmed sent.

Built by the team with Claude (Anthropic) as a coding assistant. The team created the controlled edits, manually performed SynthID tests, and reviewed the scientific claims.
