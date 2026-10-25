# HANDOFF — Below the Watermark (current, October 25, 2026)

**Source of truth:** latest `main` code, `README.md`, `docs/METHODS.md`, `results/*.json` and passing GitHub Actions. This is the **current** handoff, not an accumulation of old superseded implementation plans. `AUDIT_REPORT.md` retains the older independent audit as historical evidence.

## Shipping product direction

**Scientific-image duplication review is the headline.** The application extracts PDF figures, identifies exact and near-complete image reuse, finds **reused regions between separate figures**, checks against 112 CC BY published reference figures from 62 papers, and offers an on-demand Europe PMC search agent. The agent plans searches from manuscript text with Gemini if `GEMINI_API_KEY` is supplied, or via keyword rules otherwise; the downloaded report includes only search evidence actually obtained.

**Experimental Western blot CNN is an optional research proof of concept** until validated end-to-end results warrant something stronger. It is not a fraud verdict or a general AI detector; do not claim that a lower loss or synthetic-only recall proves real-paper reliability. Keep the v3 checkpoint live unless there is a separately reviewed model promotion.

**Three demonstration examples:** a team-made PDF with two hidden duplications, a selected AI band that the experimental CNN catches, and a whole Gemini-generated blot with C2PA credentials. The ALK2 hero graphic illustrates the manual SynthID finding, **not** a CNN catch.

## Headline evidence (PR #8 merged)

- **110/118 (93.2%)** controlled larger copied blot/micrograph blocks (~25–55% of a source figure) identified across paper figures, with resize, crop, JPEG and mirror changes. `results/region_match_eval_blocks.json`, `python -m scripts.eval_region_match --blocks`.
- **60/120 (50%)** individual small copied panels/strips: separate and harder evaluation. `results/region_match_eval.json`.
- **0/7,140** untouched reference-pair matches on the evaluated library, not population-level specificity.
- V3 experimental CNN (held-out synthetic panel benchmark): AUROC 0.703; 35.4% AI splices caught; 4.1% clean flags on 72 held-out papers. Whole-figure analysis identified an edited panel in 7/22 controlled cases.
- Manual SynthID: 10/10 whole-blot images detected; 4/4 individual pasted AI-band composites not detected. Small case study.
- V4.1 five-versus-eight-epoch work runs on a separate experimental branch with no automatic promotion.

Always report the type of test alongside the number. **93.2% does not apply to tiny strips, arbitrary intra-figure copy-move, real-world misconduct detection, or the internet agent's search recall.**

## Architecture

- `btw/region_match.py`: SIFT content features, one-to-one matching, RANSAC similarity transform and 0.90 content correlation; includes the local reference library and match crops.
- `btw/search_agent.py`: Europe PMC search, open full-text figure retrieval; optional Gemini planner and keyword fallback, with query and paper logs. End-to-end internet availability must be tested on demo machine.
- `btw/report.py`: paper raster extraction, reuse and blot routing, report schemas; `btw/api.py` FastAPI endpoints `/api/health`, `/api/analyze`, `/api/search_web`, `/api/examples`, `/api/results`.
- `web/src/App.jsx`: React review website. `web/src/reportExport.js`: offline HTML evidence export, including local and published region matches and the **most recent agent search result if actually performed**.
- `models/splice_cnn_v3.npz`: current CNN. `data/real_figures/metadata.json`: CC BY attribution and research sources.
- `examples/demo_reuse.pdf`: reproducible team-made disclosure-labeled duplication demo. Rebuild via `python -m scripts.make_demo_pdf`.

## Setup and verification

From repository root, Python 3.11+:

```bash
python -m pip install -r requirements.txt
python -m pip install "jax[cpu]" optax pytest httpx
python -m uvicorn btw.api:app --port 8000
```

Then in another terminal: `cd web && npm ci && npm run dev`, open `http://localhost:5173`. Production: `cd web && npm run build`, open `http://localhost:8000`.

Optional Gemini planning: root `.env` file, one line `GEMINI_API_KEY=...` (git-ignored); never commit or show this key. Keyword fallback does not need the key.

Check `python -m pytest -q tests`; in `web/` run `node --test src/reportExport.test.js && npm run build`. Test the demo PDF, both match pairings, the offline HTML export, and live agent query log on the presentation laptop. CI uses mocked Europe PMC / Gemini responses; it **cannot guarantee a live online run**.

## Non-negotiable scientific claims

- A matching image may be legitimate reuse, and cannot alone establish publication order.
- Empty search results do not rule out publication elsewhere. The local library is a small sample; some figures are vector or skipped due to input limits.
- Region comparison is between extracted figures; patch duplication **within a single figure** is not supported.
- Blot model is research only, weak on resized images and real-paper unknowns; heat is not a pixel-level edit mask.
- C2PA in embedded PDF images is unassessed; absence of metadata is inconclusive; SynthID detection is manual, not automated.
- Published images are CC BY with citations and unchanged originals. All demo alterations are team-made. No implication about original authors.
- Avoid stale cached outputs; example buttons analyze the original input with the current pipeline.

## Submission and last-mile checklist

1. Record `docs/video_script.md` (duplication-first, ~3 min) and attach YouTube/Vimeo link to Devpost.
2. Verify `docs/devpost.md` and deck say **110/118 larger blocks**, **60/120 small strips**, **0/7,140 untouched comparisons**; keep CNN experimental.
3. Run live Europe PMC agent once, with a real query log, before presenting it as successful (Gemini mode only if key works). Otherwise show the locally grounded library match and honestly frame online search as optional.
4. Confirm HTML report includes the region/library evidence, and that the search log appears if requested *before* download.
5. Submit before the published hackathon deadline; confirm team eligibility and completed registration independently.

Historical prototype and audit details are in `AUDIT_REPORT.md`; historical model artifacts and raw metrics remain in `results/` for reproducibility.
