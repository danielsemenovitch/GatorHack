# Below the Watermark

**A scientific-image duplication review assistant with internet-assisted published-figure matching and an experimental Western blot AI-edit detector.**

Built for Gator Hack V / UF AI Days 2026, “Fresh off the frontier.”

## What it does now

Upload a PNG, JPEG, or research-paper PDF (up to 25 MB). The FastAPI + React application:

1. Extracts embedded raster figures and records skipped or unsupported content.
2. Finds **whole-image reuse** within the upload using exact pixel checks and perceptual/structural comparisons.
3. Finds **reused regions between different figures** with SIFT keypoints, one-to-one matching, geometric alignment, and pixel agreement; resized, mirrored, cropped or JPEG-compressed copies can be recognized.
4. Compares images with a built-in reference library of **112 published figures from 62 openly licensed papers**. The reference library is a sample, not a comprehensive publication database.
5. On request, **Search Europe PMC** searches related open-access papers, downloads their figures, and compares their image regions. **Gemini plans queries only when `GEMINI_API_KEY` is configured**; otherwise keyword planning is used. The UI and downloaded report show what was searched and what was compared, if the search was run.
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

V4/V4.1 experiments are on separate branches and **not automatically installed in the live app**. Do not quote experimental training-run outcomes as production capabilities.

## Demo

Click **“A manuscript with two hidden duplications.”** We intentionally created a controlled PDF from CC BY figures, with a blot strip reused under another label and a separate figure region sourced from a 2023 paper. The first and last pages disclose our modifications; original scientific authors are not implicated. The app compares the regions directly and against the reference library.

Click **“Search Europe PMC”** for an optional on-demand search. Public keyword fallback passed a live GitHub Actions test, but test the connectivity and Gemini configuration on your actual presentation laptop before relying on it. The public **keyword-planned Europe PMC pathway passed a real live smoke test**: 16 papers, 55 figures compared, and the intentional source PMC10701888 found in 43.9 seconds ([saved result](results/europepmc_live_smoke.json)). Gemini-assisted planning has not been tested live here and still depends on the user's API key and quota.

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

**Optional Gemini planning:** save `GEMINI_API_KEY=...` in the repository-root `.env` file (git-ignored) and restart the backend. Without a key, the agent uses keyword queries. Never commit or display secrets.

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
- `docs/METHODS.md`, `docs/DATA_CARD.md` — reproducible methodology and licensing
- `docs/devpost.md`, `docs/video_script.md`, `docs/pitch_outline.md` — up-to-date submission materials
- `AUDIT_REPORT.md` — **historical** independent audit of earlier versions (not current product specification)
- `HANDOFF.md` — current engineering handoff
- `results/` — measured data; `examples/` — source-attributed team-made demonstrations

## Limitations and ethics

The region matcher misses some small, uniform, or heavily transformed regions, and it does not compare regions inside a single figure. Neither the local reference library nor the Europe PMC agent searches all scientific publications. On-demand search can return zero results or be offline; downloaded reports explicitly distinguish “not searched” from “searched, no matches.”

The blot CNN is experimental, trained on synthetic manipulations, and particularly weak after resizing. A heat overlay shows coarse 32-pixel patches, not a confirmed manipulation boundary. Missing provenance or no flags are not proof of authenticity.

Original scientific figures used here are CC BY and attributed in `data/real_figures/metadata.json`. **Every edit displayed in a demo was made by our team**; no source author is accused. The SynthID observations are manual experiments, with no automated probing; disclosure to Google is drafted but not confirmed sent.

Built by the team with Claude (Anthropic) as a coding assistant. The team created the controlled edits, manually performed SynthID tests, and reviewed the scientific claims.
