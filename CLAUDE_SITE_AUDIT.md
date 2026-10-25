# Independent Site Audit Brief (for Claude or another reviewer)
**Project:** Below the Watermark / GatorHack
**Updated:** 2026-10-24, after merged PRs #1–#4
**Branch to review:** \`main\` at the actual latest commit. Do not audit stale branch code.

## Start here

1. Read **\`HANDOFF.md\` from the top**, especially the LIVE STATUS, OCT 24 updates, product direction, hard constraints, and test findings. The long document preserves old experimental results; *latest sections supersede older claims*. Read \`README.md\`, \`docs/DATA_CARD.md\`, the API, the React frontend, and tests.
2. Inspect actual source and **verify claims by running the app**, not by trusting this audit brief. Identify the exact commit SHA you audited.
3. Prefer **working code + reproducible tests + truthful scientific claims** over cosmetics. This is a hackathon tool, not a validated forensic instrument.

### What the project actually does

A React/Vite + FastAPI web app accepts JPEG/PNG or research-paper PDF. The backend extracts **embedded raster figures**, compares **whole embedded images within the same uploaded file** for exact or close reuse, reads available standalone-image C2PA information, optionally screens **blot-like panels** with an experimental JAX CNN, applies weak/conservative copied-band matching, and exports a printable HTML evidence report with figures and review leads.

**Do not imply:** confirmed scientific fraud; universal AI-image identification; pixel-accurate edit boundaries; patch-duplicate detection in microscopy; validated C2PA signatures on PDF-embedded images; automated SynthID; absence of findings equals authentic/clean.

## How to run locally

Backend from repository root (Python **3.11** most-tested, JAX on Windows may require WSL2):
\`\`\`bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install "jax[cpu]" optax pytest
.venv/bin/python -m uvicorn btw.api:app --host 127.0.0.1 --port 8000
\`\`\`
On Windows PowerShell the venv Python is \`.\.venv\Scripts\python.exe\`; adjust the invocations.

Frontend in a second terminal:
\`\`\`bash
cd web && npm ci && npm run dev
\`\`\`
Open \`http://localhost:5173\`. Use the actual Vite host/proxy config if it differs. Confirm \`GET /api/health\` and \`POST /api/analyze\` work, and check frontend console/network errors. No live public production URL has been verified.

Run existing tests:
\`\`\`bash
python -m pytest -q tests/test_integrity_mvp.py tests/test_report_regressions.py
cd web && npm ci && npm run build
\`\`\`
Reference CI:
- [Conservative screening and heat run, 17 backend tests + Vite build](https://github.com/danielsemenovitch/GatorHack/actions/runs/40139731885)
- [Live example recompilation build](https://github.com/danielsemenovitch/GatorHack/actions/runs/40140002908)
- [Expanded v3 training](https://github.com/danielsemenovitch/GatorHack/actions/runs/40130694228)

## High-priority audit matrix (try to break it)

| Area | Required verification | Known constraints / red flags |
|---|---|---|
| Full UI | Open each tab/page, upload controls, in-progress/error states, report sections, charts/examples, links, responsive mobile widths, browser console; check a11y focus/contrast/alt labels. | An old cached example report previously showed stale heat; PR #4 changed example buttons to re-run source images. Confirm current output, including hero preview and selected settings. |
| PDF/image upload | Try valid JPG/PNG, paper PDF, wrong MIME/extension, corrupt input, empty file, >25MB file, unusual dimensions, many embedded figures. Confirm errors and never a false authenticity claim. | Extracts embedded raster images, not all vector content; limits still apply (PDF figures and pixels). PDF C2PA is *not assessed*. |
| Whole-image reuse | Upload a PDF with deliberately duplicated figures and a PDF with unrelated figures. Check source page IDs, side-by-side matches, matching hash/SSIM behavior, report exports. | Whole-image comparisons only **within uploaded PDF**. No copy-move inside microscopy and no cross-paper database search. A match can be legitimate. |
| Blot routing vs microscopy | Feed genuine blot strips, fluorescence microscopy, cell micrographs, bar charts, and unusual AI whole-blot photographs. Try automatic, user-confirmed, and skip modes. | A grayscale microscopy crop on the Oncotarget PDF's **page 10** was misclassified. PR #3 added Sobel horizontal/vertical edge screening; confirm it excludes this crop without skipping too many genuine blot strips. This heuristic is **not** a certified modality classifier. |
| CNN v2/v3 | Verify loading both checkpoints; v3 is selected by default. Compare conservative vs research benchmark threshold **on the SAME source image**; ensure reported model/policy/threshold agree with actual inference. | V3 expanded held-out AUROC ~0.703; AI splice recall ~35.4% at original threshold, per-image clean false flags ~4.1%; after 50%-resize AI splice recall ~5.1%. These are synthetic-label benchmarks, not proof of reliable real-paper performance. |
| Conservative flagging | For v3, default review index threshold **0.90**; benchmark mode **0.50**. Run a variety of intact papers and controlled edited samples; measure both missed edits and flags per paper. | Applying threshold to the original Oncotarget v3 scores gives **2/37** review flags instead of **18/37**, before microscopy filter. **Post-hoc on one unlabeled paper only**, *not* validated specificity. V2 legacy scores saturate near 1 and do not use the v3 strict setting. |
| Heat location | Toggle blot heat; compare original figure, panel crop and patch overlay, at various browser widths. Check \`panel.x/y/w/h\`, source dimensions, alpha map size/coordinate scaling, crop orientation, selection hotspots, transformed/zoomed display. | Model uses overlapping **32-pixel patches**, not segmentation. PR #3 fixed threshold-before-blur and excluded unscored non-band patches. The full Gemini blot isn't ordinarily recognized by router; forced heat geometry audit is in the CI artifact. A hotspot near a molecular-weight label does **not** mean the label was edited. |
| Copied bands | Verify both source/matched crop previews, figure/page labels, no spurious "band 0", and noise-correlation wording. Compare untampered panels vs deliberately copied ones. | At strict noise/shape 0.90 policy, synthetic copy detection ~**6.9%** with ~2% untouched-control false flags. Very low recall. Do NOT describe "no match" as absence of copying. |
| C2PA and SynthID | Test missing and invalid credentials; read \`btw/provenance.py\`; ensure explicit "not assessed" for PDF-embedded signatures and "manual" for SynthID. | No automated SynthID uploads. Missing Content Credentials are inconclusive. |
| HTML report | Download, open offline, try printing, inspect figure labels, matched pair crops, score policy, skipped/partial panel counts, styled captions and textual caveats. Feed malicious/odd filename and check HTML escaping; validate no external script dependency. | HTML export is self-contained, not a verified standalone PDF export. |
| Security/performance | Check upload size enforcement, PDF parsing abuse, decompression bombs, server exceptions/logging, CORS, frontend encoding, request latency & memory on many-panels PDFs. | Check \`btw/api.py\` error handling and any claimed data retention policy instead of assuming it. |
| Media/claims | Review intro, badges, example names, "The gap" / SynthID language, labels, dashboard stats, README and pitch claims. Ensure old v1 metrics aren't presented as v3; make experimental disclaimers unmissable. | The entire-Gemini blot is synthetic, but the detector does not reliably label whole images as AI, and the heat map is not a proof map. |

## Grounded test assets

- Repo: \`examples/gemini_whole_blot.jpeg\` (a whole Gemini-generated blot photograph), \`examples/alk2_ai_band.jpg\` (intentionally edited published CC BY figure, clearly attributed), and \`examples/alk2_original.jpg\` (original). *Some old* \`examples/*.report.json\` files are **historical/stale artifacts**; prefer live image analysis.
- Controlled 3-figure repeated-image PDF, plus JSON and HTML results: [Actions run 40130346270 artifacts](https://github.com/danielsemenovitch/GatorHack/actions/runs/40130346270) → \`integrity-pdf-demo\`. Two copies were repeated by the team; never imply the paper authors duplicated them.
- Whole-Gemini heat geometry comparison: [Actions run 40139731885 artifacts](https://github.com/danielsemenovitch/GatorHack/actions/runs/40139731885) → \`conservative-heat-audit\`. Clearly marked debug/forced crop where applicable.
- Oncotarget paper: DOI \`10.18632/oncotarget.23647\` (Nazim et al.); the **exact user-uploaded PDF and its HTML report** may not be committed in GitHub. In the initial v2/v3 reports, both flagged 18 of 37 candidates at their benchmark thresholds, and page 10 microscopy was misrouted. If this artifact isn't present locally, say so rather than reconstructing "the same" paper invisibly. Running original figure JPEGs through a new PDF yields different crops.

## Model limitations that must survive your edits

1. Experimental CNN v3 learned from ~241 additional paper IDs' openly accessible source figures, but most manipulation examples remain **synthetic** and AI splice donor variety is limited.
2. More epochs on the **same** two AI donor blots are not proven to improve generalization; robust v4 should use genuine hard negatives, generator diversity, multiscale augmentations, masks and held-out papers/generators.
3. The strict 0.90 cutoff is a user-experience tradeoff, not a statistically calibrated false-positive operating point.
4. No allegation about a published paper should be made from CNN scores, heat or noise correlations.

## What to deliver as the external auditor

Provide one concise **AUDIT_REPORT.md** that includes:
- Commit SHA, runtime/OS, which commands/tests were actually run, and what could *not* be verified.
- A **P0/P1/P2 bug table** with reproduction steps, expected vs actual behavior, precise file/line, screenshots where relevant, and recommended fix.
- A clear distinction between **bugs**, **known methodological limitations**, **scientific misstatements**, **UX/accessibility problems**, and **speculative enhancements**.
- Before/after evidence for each changed behavior. Use tests rather than silently tweaking production thresholds to get an appealing screenshot.
- Audit of the full click-through and at least one realistic upload-to-export journey, plus explicit checking for stale example reports.
- If making code changes, do them on a feature branch, run tests/build, and provide the PR/commit; do not silently overwrite checkpoints, model metrics or existing evaluation data.
- For the hackathon: a 3-minute reliable demo path and any **submission-blocking P0 risks**, plus truthful copy changes.

**Avoid:** fabricating a hosted site URL, inventing availability of the original PDF file, running SynthID automatically, claiming model scores are probabilities of fraud, or suppressing alert visibility without documenting sensitivity tradeoffs.

## Ready-to-paste instruction to Claude

> Audit the entire current \`main\` branch of \`danielsemenovitch/GatorHack\` as an independent skeptical reviewer. Start with \`HANDOFF.md\` and \`CLAUDE_SITE_AUDIT.md\`. Run the app and walk every UI route and the PDF/image upload-to-HTML-report flow, try the test assets listed, inspect security/performance/accessibility, review scientific truthfulness, and test CNN v2/v3 Conservative versus Research modes, microscopy routing and heat-map alignment. Do not treat existing self-reported CI as sufficient verification. Create a prioritized audit report with actionable bugs and evidence. Make high-confidence fixes in a PR only after tests pass. Keep both model checkpoints, and don't claim authenticity/forgery conclusions from unvalidated scores. Highlight anything that would prevent a credible GatorHack demo and submission.
