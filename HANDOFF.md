# HANDOFF: Below the Watermark (Gator Hack V, UF AI Days 2026)

This is the continuity document for teammates and future assistants. **The current product direction and LIVE STATUS were updated Saturday Oct 24, 2026, around 3:40pm ET.** Historical model and benchmark notes below are retained as evidence; read the new strategy and hard constraints before editing.

---

## LIVE STATUS (read first)

<!-- LIVE -->
- **Updated Sat Oct 24, 2026, approx. 3:40pm ET: PRODUCT DIRECTION CHANGED.** Read **CURRENT PRODUCT DIRECTION** directly below before following any older TODO below. We are building a scientific-image integrity *review assistant* with a WESTERN BLOT specialty, not claiming a reliable general fraud or AI-image detector. No PowerPoint is needed now.
- **Repo:** https://github.com/danielsemenovitch/GatorHack ; `main` still contains the production v2 CNN. The experimental code is only on `experiment/classifier-stress-20261024`, not deployed or merged.
- **New scale stress experiment FINISHED successfully** (GitHub Actions run https://github.com/danielsemenovitch/GatorHack/actions/runs/40124259884). On 103 NEW deterministic test pairs, native: old CNN 37/103 detected, 4/103 clean false flags; scale-augmented logistic head 48/103 detected, 9/103 false flags. At 50% resize: 1/103 versus 3/103 detected, with 0/103 versus 6/103 false flags. **Not a satisfactory improvement; do not deploy the experimental head.** Its test set and operating point differ from the older `results/robustness.json` experiment; never conflate their percentages. Results currently live in the successful Actions log/artifact, NOT in `main/results`.
- **Uploaded bundle 2 (440 figures from 241 new papers) was validated and provided to ChatGPT as `btw_real_blots_bundle2.json` but has NOT been imported into the repo or used to train CNN v3.** A prior chat generated a standalone audit/import patch and a preliminary presentation; neither should be mistaken for production code or a trained model. Keep JSON bundle out of git.
- **Current product pivot priority:** get the existing PDF/image -> panel/report workflow working end-to-end; design and IMPLEMENT universal provenance + cross-figure duplication/near-duplicate checks with explicit confidence/limitations; create a reproducible downloadable review report; add user-friendly unsupported/insufficient-quality states. Do not run blot CNN over non-blot figures. Validate the general detectors before claiming they work.
- **Pending external/manual:** hand-upload SynthID round 4 S0-S4 and enter exact outcomes in `results/synthid_manual.json`; Devpost submission and team roster; video; optional Google disclosure; Monday in-person finalist attendance. SynthID remains manual only, never automate or iterate to evade it.
- **Deadline:** Sun Oct 25 at 5pm ET; submit by 4pm. Working product and honest demo come before uncertain additional model training.
<!-- /LIVE -->

---

## CURRENT PRODUCT DIRECTION — READ BEFORE EDITING CODE (decision made Oct 24)

### A. Team summary / pitch in plain language

**Name:** Below the Watermark — Scientific Image Integrity Review Assistant.

**One-sentence explanation:** A researcher, peer reviewer, or journal editor can upload a scientific figure or research-paper PDF; the app organizes the embedded images, checks provenance and potential image reuse, applies specialized Western-blot forensic screening only where appropriate, and produces a transparent, exportable report of evidence for **human review**.

**Short explanation to tell teammates:** We started by asking whether watermarks like SynthID can detect an AI-generated *single band* pasted into a scientific Western blot. Our experiments found that whole AI blots were detectable in the tested scenarios, while isolated bands were too small to verify. We created a Western-blot forgery benchmark and an experimental CNN to investigate that gap. The CNN is not accurate enough to certify whether a blot is authentic. Rather than promise a detector we do not have, we are building a practical integrity **workflow** around it. For any scientific figure, the workflow can provide input inventory, available provenance and (once implemented and validated) repeated-region/image-reuse evidence; for detected Western blots it additionally offers a clearly labeled, experimental band-splice screen. The final artifact is a review report showing *what was examined, what was found, what cannot be determined, and what to check manually*.

**Keep the frontier hook:** Gemini image tools + publicly accessible SynthID Detector motivated the project; the open, labeled AI-era Western-blot benchmark and measured watermark limitations remain the research contribution. The broader utility is a usable review assistant, NOT a newly validated general-purpose AI-image detector. "AI watermarks catch whole fake images. Science gets faked one band at a time" can remain a hook, but do not suggest every manipulated image is AI-made or watermarked.

**Avoid this pitch:** "We detect fake scientific images and prove research misconduct." **Use:** "We help reviewers triage scientific figures and inspect reproducible integrity signals; flags are leads, not verdicts."

### B. Scope decision: broad INPUT and REVIEW, narrow ML specialization

**Decision:** ACCEPT scientific images and PDFs of many modalities; **only apply the blot-trained CNN to Western blots**. Do not stretch its 0.643 AUROC to microscopy, gels, histology, charts, or fluorescence images. For other modalities the MVP should still offer provenance, quality caveats, and general-purpose duplicate/reuse inspection once that is built and validated. If no applicable detector has run, show "Not assessed" or "No applicable checks" — never "Clean", "Authentic", or a meaningless score.

| Input/category | Intended checks | State as of Oct 24 | Caution |
|---|---|---|---|
| Western blot/immunoblot | C2PA provenance + paper/figure-level reuse checks + blot panel extraction + band-specific CNN + band-duplicate matcher | Blot workflow IMPLEMENTED, experimental/weak | V2 catches 22.1% of AI splices at 2.7% false flags on the benchmark; resize destroys signal |
| Agarose gels / SDS-PAGE and related gels | Same general checks, optionally gel-specific lane/band work after independent validation | GENERAL CHECKS PROPOSED; no gel model | Visual resemblance does not validate blot model transfer |
| Microscopy (light, fluorescence, confocal), histology, cell culture photos | C2PA + exact/near image reuse + localized duplicated patch matching; human comparison view | GENERAL DUPLICATION PROPOSED; NOT implemented | Repeated cell patterns/textures and legitimate reused controls cause false flags |
| Graphs, diagrams, photographic panels, other raster figures | C2PA, duplicates of whole figures/crops, resolution/metadata context | GENERAL CHECKS PROPOSED; input/PDF figure extraction exists | Reused templates, labels and scale bars are not manipulation; graph data cannot be verified from pixels alone |
| Full article PDF | Extract embedded raster figures and associate page/source; run appropriate per-image checks; compile one review report | Embedded-raster extraction IMPLEMENTED; generic per-image checks & export PROPOSED | Scanned/vector-only content may be missed; currently PDF-level C2PA not checked |

**What code really does TODAY (audit, not aspirations):**
- `btw/api.py` exposes POST `/api/analyze`, accepts uploads up to 25 MB; `btw/report.py` handles an image or embedded raster images in a PDF with PyMuPDF, returns report JSON and figure previews.
- `btw/panels.py` uses OpenCV-connected non-white regions and a *blot-likeness heuristic*. It does **not** classify arbitrary scientific modality types.
- `btw/cnn.py`/`btw/detector.py` evaluate **blot patches** (relative intensity and noise residual), make localized heat maps; production model is `models/splice_cnn.npz`. There is no general synthetic-image detector.
- `btw/duplicates.py` detects **copied blot bands** by matching high-pass/noise-residual templates and flips **among blot panels**. It is **not** a generic microscopy/whole-image duplicate engine. `results/duplicates_metrics.json`: on 94 held-out panels, 10.6% detection at 5.3% false flags at native size; ~2.1% detection under JPEG/PDF/half-size. No claim of robust duplication detection is warranted.
- `btw/provenance.py` looks for C2PA and attempts to read/validate manifests with `c2pa-python`; fallback byte strings are **not signed validation**. In `btw/report.py`, PDF upload currently marks C2PA as not checked; generic EXIF metadata extraction is **not implemented**. Absence of a credential does **not** mean tampering.
- `web/src/App.jsx` displays blot analysis and handoff to manual SynthID, but NO user-downloadable integrity PDF/HTML report workflow exists yet.
- Inputs *can* include non-blot images, but current processing may find zero blot panels or apply a blot heuristic inappropriately; the new general-purpose analysis below is still engineering work.

### C. The actual product workflow, end-to-end

1. **Intake and asset inventory (existing partial implementation):** upload JPG/PNG/other supported raster or PDF via React/FastAPI. Parse PDFs with PyMuPDF; extract embedded figures and retain page and image identifiers. Add direct non-blot handling rather than discarding unmatched figures. Guard unsupported formats, file limits, huge PDF page counts, corrupt images, and failures. Keep original bytes when checking provenance.
2. **Per-asset basics (MVP build):** record dimensions, pixel count, color mode, format, downsampling/low-resolution warning, cryptographic SHA-256 of *input bytes*, and available non-sensitive EXIF metadata (if present). EXIF can be missing, edited, or stripped: it is contextual metadata, not authentication. Record processing version and analysis scope.
3. **C2PA provenance check (partly existing):** use `c2pa-python` for manifest/claim/signature validation when supported. Display separate states **valid signed claim**, **present but unverified/invalid**, **not found**, **unsupported/error**. Show issuer and declared actions if verified. Never assume signed claim means the scientific data is genuine. Keep fallback byte-scan strictly "possible marker, unverified". For PDFs explicitly display "embedded-image provenance not assessed" until that support works.
4. **Generic image-reuse checks (NEW, not currently implemented):**
   - **Exact same-image reuse:** SHA-256 of canonical/decoded image pixels (as well as original bytes) to find pixel-identical images across figures/pages of the *same uploaded document*, even when container metadata differs. Make a gallery of both source locations. This only establishes identical pixels, not whether reuse is improper.
   - **Near-duplicate whole-image reuse:** perceptual hash (pHash/dHash, suggested OpenCV/Pillow/imagehash or own DCT) + confirm geometric alignment and image similarity (e.g. SSIM or registered pixel comparison). Can catch resized/recompressed/recolored versions; by itself pHash is only a candidate generator, not reliable proof.
   - **Repeated localized region / copy-move:** for sufficiently detailed raster figures, compare local keypoints/descriptors (OpenCV ORB or SIFT where available) within and across figures, group correspondences via geometric consistency (RANSAC/affine), require spatially separated nontrivial regions, check aligned texture similarity and reject trivial text, scale bars, blank backgrounds and repeated panels. Optionally include noise-residual correlation for confirmation. Return bounding boxes on **both locations** and transformation details. This is a proposed separate generic detector; do NOT call `btw.duplicates.find_duplicates` as though it does this now.
   - **Strong controls for false matches:** exclude overlapping/self matches, borders/lettering/scale bars, identical schematic elements, common grids and repeated biological texture; calibrate thresholds on diverse **genuine** microscopy/gel/photograph figures. Legitimately reused controls or representative microscopy images may be valid if labeled. Every flagged reuse needs contextual scientific review.
   - **Scope explicitly limited:** within the uploaded document or files the user supplies; no global web plagiarism/database matching unless separately implemented, permitted and disclosed.
5. **Optional quality/inconsistency indicators (NEW, exploratory only):** per-figure resolution, heavy compression, local high-pass noise discontinuities, clipping/saturation, unusual JPEG blocks/ELA. These can assist a reviewer but are **not forgery verdicts**; publishing workflows themselves create inconsistencies. If unvalidated, present as diagnostic maps, not red "fraud" flags. Avoid a single opaque "authenticity score".
6. **Blot-specific branch (existing, to be guarded by routing):** if a panel looks like a Western blot (allow manual "this is a blot" override and "not sure"), run `find_panels` + `find_bands` + current `CnnDetector` and existing within-blot duplicate check. Display panel/band overlays, raw operating threshold, source-panel context and explicit "experimental model" warning. Current CNN scores are **not probabilities of misconduct**.
7. **Human review and report (NEW high-priority MVP):** per file/figure/page, list checks run, checks skipped, technical signal and supporting paired crops/overlays, provenance details, quality limitations and a recommendation for human inspection. Let user download a self-contained **HTML review report** first (easier and reproducible); add PDF export via print or generated PyMuPDF after HTML is stable. No unsupported "guilty" label. Ideally offer reviewer note/checklist and "mark explained" for legitimate reuse. Always clearly distinguish a missing check from a negative result.

### D. Integrity signals and interpretation — judge-ready explanation

| Signal | Mechanism / proposed libraries | Where it applies | Interpretation / limitations |
|---|---|---|---|
| C2PA Content Credentials | Existing `btw/provenance.py`, `c2pa-python` signature/manifest reader | Direct raster uploads today; PDFs future | Verified origin/edit assertion **when present**, not proof of research honesty; missing usually inconclusive |
| Exact duplication | NEW cryptographic pixel SHA-256 after canonicalization | All raster figures/pages in a submission | Same underlying pixels reused; could be valid. Bytes hash alone misses resave |
| Near-duplicate images | NEW perceptual hashes + image registration + SSIM/pixel difference | All raster figures | Recompression, resizing, mild color variations; templates and figure copies can yield false positives |
| Duplicated *regions* | NEW ORB/SIFT keypoints, geometric matching and confirmation | Microscopy and other detailed images, some gels | Candidate copy-move/region reuse, not proof; visually repetitive scientific textures generate false matches |
| Copied Western blot bands | Existing `btw/duplicates.py` high-pass residual and template matching (flips) | Blot panels only | Very low recall at web resolution after publishing; show as experimental |
| AI band splice suspicion | Existing JAX CNN, relative pixel/noise texture, heat map | Validated target: Western blots only | Detects some simulated forgeries; other modalities **unsupported** |
| JPEG/noise/ELA irregularity | Some existing features in `btw/features.py` but only inside blot baseline; general viewer NEW | Exploratory across modalities | Heuristics highly sensitive to benign processing; never standalone verdict |
| EXIF metadata / file quality | NEW Pillow/image parser + dimensions/compression warnings | Direct images | Missing/mismatched fields are not proof of deception; helps understand transformations |
| SynthID | Manual external check at synthid.com; NEVER programmatic | Appropriate whole AI images, only manual | Only shows watermark detection/limitations for supported images; a negative is not proof of human origin |

**Recommended report language:** "Potential duplicate region between figure 2 and figure 4 (review images side by side)", "No signed content credential was found (inconclusive)", "Western-blot anomaly score exceeded the experimental review threshold", "Could not evaluate because the band is too small" and "No applicable forensic screen available for this image." Avoid "fake", "AI generated", "fraud" and "authentic" as automatic conclusions.

### E. Quantitative evidence (DO NOT replace with optimistic assumptions)

- Current benchmark in `main`: 4,725 derived panel images, 241 selected panels from 55 CC BY papers; by-paper test split. CNN v2 `results/cnn_metrics.json`: AUROC **0.643**, clean false flags **0.027**, simulated AI splices caught **0.221**, intensify **0.375**, erase **0.023** at the saved threshold. The v1 held-out-method result is copied into the v2 metrics and must not be presented as v2-generated.
- `results/robustness.json`: 103 held-out panel AI-splice pairs, native **0.379 detection / 0.039 false flags**, half-resolution **0.019 / 0.000**, one publishing step at a time.
- **New Oct 24 independent stress test** on branch `experiment/classifier-stress-20261024`: `scripts/stress_classifier.py`, successful [GitHub Actions run](https://github.com/danielsemenovitch/GatorHack/actions/runs/40124259884); 103 test pairs of different deterministic forgeries, original CNN versus a trained scale-aware logistic post-processor:
  - Native: existing **37/103 (35.9%)** detection, **4/103 (3.9%)** clean false flags; experimental **48/103 (46.6%)** detection, **9/103 (8.7%)** false flags.
  - 50% resize: existing **1/103 (1.0%)** detection, **0/103 (0.0%)** false flags; experimental **3/103 (2.9%)** detection, **6/103 (5.8%)** false flags.
  - 33% resize: existing **0/103**, experimental **1/103**, with near-chance AUROC (~0.51).
  - The experimental head provides no trustworthy rescue after resize, and higher false positives at native resolution. **Do not merge or deploy it.** Comparisons above use different operating thresholds/calibration protocols; evaluate detection and FPR together. Logs/artifact, not production files, contain new results.
- `results/duplicates_metrics.json` shows band-copy detector native detect **0.106**, false flags **0.053**, after JPEG 85 detect **0.021**, false flags **0.021**. Do not sell this as a high-performing general duplicate detector.
- **The 440-figure second CC BY bundle has not yet been ingested.** Validation of decodable files does not count as model training. Retrain CNN v3 only when product works and there is time; preserve a v2 checkpoint, use by-paper splits, no test-set tuning.
- **All generic image-reuse and metadata features proposed above have NO measured sensitivity/specificity yet.** Never claim a detection rate for them until a separate non-blot benchmark exists.

### F. MVP roadmap, prioritized for the Oct 25 deadline

**P0 — make the existing tool usable and demo-safe (do first):**
1. Test server + frontend locally: `pip install -r requirements.txt`; install `jax[cpu]` and `optax` as needed; run `uvicorn btw.api:app --port 8000`; `cd web && npm ci && npm run build`. Check file/PDF inputs, errors, elapsed time, no-panel result, credible demo examples, C2PA claims. No result pretense.
2. Adjust routing/UI so every extracted figure can be reviewed, while Western blot heat maps only appear for confident blot panels or explicit manual selection. State "not assessed" for unrecognized modalities instead of "no manipulation". Preserve existing panel functionality and score thresholds.
3. Add an HTML evidence report export of exactly the analyzed figures, source page/figure IDs, matched region crops, methods run/not run, uncertainties and attribution. Verify exported content against JSON, on both successful and no-finding cases.

**P1 — general-purpose integrity markers with demonstrable, bounded utility:**
4. Add whole-image **exact pixel duplicates across embedded figures** with paired thumbnails; very testable, low runtime, minimal false ambiguity. Include benign repeated logo/figure control cases.
5. Add whole-image **near duplicates** (perceptual hash for candidates, registration/SSIM confirmation) and paired evidence. Calibrate on deliberately resized/recompressed duplicates **and** genuine unrelated images. No unilateral accusation.
6. Add a *basic* generic local duplicate check **only if** time and validation allow; otherwise document it as next-step. Require reliable overlays and benign repeated-cell/text controls. This is harder than whole-image duplication.
7. Separate C2PA "validated"/"found unverified"/"absent"/"not checked" from EXIF/quality. Do not conflate them.

**P2 — only after P0/P1 are complete:**
8. Expand CC BY blot benchmark with bundle 2 and train CNN v3; rerun the same held-out and robustness metrics; keep previous model if not genuinely better. Generator-held-out and modality-held-out generalization are future work.
9. Manual SynthID round 4, documentation/README/Devpost updates, demo video and deadline tasks. Never imply the Google disclosure email was sent unless actually sent.

**DEMO narrative (honest and useful):** (a) upload a CC BY paper or own synthetic demo PDF containing two reused/edited images, (b) show source figure extraction, (c) show *independently verified* exact/near duplicate pair markers if implemented, (d) show a Western blot band heat map as **experimental** with its false-positive caveat, (e) export an evidence report, (f) show the limited SynthID case-study matrix. If general duplicate checks aren't ready, omit the corresponding step instead of using a mocked success. No private AKT data, and label every synthetic edit as ours.

### G. Architecture / implementation handoff to the next coder

Keep the FastAPI endpoint and React product intact. Prefer **modular checks** so extending beyond Western blots does not break the current workflow.

**Suggested additions, not existing paths:** `btw/inventory.py` (PDF/image inventory and source IDs), `btw/generic_duplicates.py` (whole/pixel/near/local reuse), `btw/quality.py` (metadata and diagnostic limits), `btw/report_export.py` (export), optional `btw/routing.py` (blot-vs-other supported routing). Add deterministic tests under `tests/`, without making external services a test dependency.

**Suggested unified findings schema:**
~~~json
{
  "method": "exact_pixel_duplicate | near_duplicate | local_duplicate | c2pa | blot_cnn | quality",
  "status": "finding | no_finding | not_assessed | inconclusive | error",
  "figure_ids": ["page-1-img-1", "page-2-img-3"],
  "regions": [{"figure_id": "page-1-img-1", "bbox_xywh": [4, 8, 40, 25]}],
  "measurement": {"name": "pixel_equal", "value": true},
  "limitations": ["Visual reuse may be legitimate"],
  "human_review_required": true
}
~~~
Keep detector-specific values and calibration separate; do **not** convert all measurements into one unsupported numerical authenticity score. For each finding, preserve source coordinates and transforms so overlays/extracted crops are auditable. Persist JSON and add report-download endpoint or frontend HTML export; minimize unnecessary retention of uploaded unpublished data (project repo/demo uses CC BY data only).

**Testing and honest acceptance criteria:**
- Exact pixel duplicate: detects a true positive across two figures (same decoded pixels) and does not flag a different image. Verify page IDs and bounding boxes.
- Near duplicate: finds resized/compressed same image and avoids dissimilar controls; report comparison image and similarity threshold.
- Local duplicates: evaluate moved/flipped/rotated patches AND unrelated repetitive-cell, grid/label, scale-bar negatives; track sensitivity/FPR, don't demo unless stable.
- C2PA: known signed-positive sample, malformed/unverified, absent and PDF "not assessed".
- Routing: never call blot CNN on microscopy or photos just because segmentation returned a rectangle; manual override labeled appropriately.
- Report: reviewer sees original, highlighted evidence, limitations, checks performed/skipped, attribution, and no misconduct accusation. Match fields to underlying analysis JSON.
- Preserve paper-level holdouts and version metrics when/if CNN training resumes.

**Key source files**: `btw/report.py` is the orchestration point; `btw/api.py` upload route; `btw/panels.py` restricted blot finding; `btw/duplicates.py` restricted band matching; `btw/provenance.py` credential reading; `btw/detector.py`/`btw/cnn.py` learned blot screen; `web/src/App.jsx` the current viewer. New modules need explicit calls from `report.analyze` and app display components; creating modules alone changes nothing.

### H. Explicit non-goals and continuity safeguards

- Never present a "negative" screen as proof an image is untouched. Never confuse a *missing watermark* or C2PA with AI-origin verdict.
- No automated submissions to SynthID or optimization against it. Manual check only, per existing project terms.
- No real authors accused, no claim that synthetic edits occurred in published papers; all edited demonstrations clearly labeled as team-made.
- No use of Daniel's unpublished/private AKT/pAKT blot; see HARD CONSTRAINTS below.
- No unvalidated microscopy forgery CNN, no global plagiarism search without data/source permission, no universal "AI generated" claim.
- General original section 1, original code map and older TODOs **below** explain how we got here. When they conflict with this CURRENT PRODUCT DIRECTION and updated LIVE status, **this newer direction wins**. The old numerical experiments remain useful as historical, timestamped evidence, not new model results.
- Keep Devpost/README claims synced with only deployed, tested capabilities. Judge can ask for raw evidence or how an output was calculated: be ready to show the input, method, threshold, and output.
- On a future chat: open `HANDOFF.md` and read this strategy + constraints + experiment notes FIRST, inspect current branch, check whether P0/P1 implementation has been merged, then continue the highest-priority unfinished item. Do not automatically spend an hour retraining the CNN or make slides unless the product and submission items are done.

---

## 0. Hard deadlines and rules (do not miss these)

| When (ET) | What |
|---|---|
| **Sun Oct 25, 5:00 PM** | Devpost submission closes. Late submissions are not accepted. Aim to submit by 4:00 PM. |
| Sun night | Offline judging by UF grad students: they read Devpost, the repo and the video. Top 10 are announced before Monday morning. |
| **Mon Oct 26, 10:15 to 11:15 AM** | Top 10 pitch live at the Reitz Union. **At least one team member must be physically present or the team cannot win.** |
| Mon 12:15 PM | Awards (AI Student Competition Awards). |

**Devpost requirements:** title and description; link to a **public** code repo; a demo video (max 3 min) or slides (work in progress is acceptable); full team roster (2 to 5 people, all registered on Devpost); tracks and sponsor challenges you want to be considered for.

**Event rules that matter:** all project code must be written during the hacking period (Fri Oct 23 5pm to Sun Oct 25 5pm). Open source libraries and public APIs are allowed. MLH Code of Conduct applies.

**Judging criteria:** Impact, Technical Execution, Design & Usability, Novelty, Presentation & Demo (which explicitly asks you to explain limitations).

**Prizes to enter on Devpost:** main awards ($2,000 / $1,000 / $500); Best AI Hack, Best Design, Best Social Impact ($200 each); sponsor tracks **Best Use of ML** (UF DSI) and **Best Use of Gemini** (MLH). One project can win a main award plus track and sponsor prizes.

**Theme released Friday 5pm, "Fresh off the frontier":** build something that uses a technology, dataset or discovery first made public in the past 12 months, central to the project, in a field you care about. Be ready to show what it does, what difficulties you faced, and why the tech is recent.

**People:** the student organizer and a judge is **Matheus Kunzler Maldaner**, the lead of last year's (2025) winning team, whose project "Plato's Cave" verified research paper claims with web agents. Expect him to probe how the detector works and whether anything is mocked. Don't pitch anything that's stubbed.

---

## 1. The project in one paragraph

In the past year, Gemini's image tools made it trivial to fabricate a western blot, or to edit one (an early test asked Gemini to make a drug's effect disappear from a blot, and it did, with no refusal; that test used the private AKT blot, so it must be re-done on a CC BY figure or our own AI blot before it is shown or claimed). On **Oct 7, 2026**, Google opened **SynthID Detector** (synthid.com) to the public as the safety net. We tested it by hand on scientific images. Whole AI images were caught every time, even after JPEG, resizing, cropping and PDF export. Single AI bands were "not enough details." An AI band pasted into a real blot came back "not detected" (that original test used a teammate's private blot, which must not be used, so it is being **re-run on CC BY figures**; see section 4). We then built (1) an **open benchmark** of band level blot forgeries with pixel masks, (2) a **band splice detector** trained on it, and (3) a **web app** that checks a figure or PDF band by band, looks for copied bands, reads C2PA content credentials, and hands the user panel crops to check in SynthID by hand.

**Pitch line:** "AI watermarks catch whole fake images. Science gets faked one band at a time."

**What is novel:** (a) the measured finding about where SynthID stops working for science; (b) an open benchmark that didn't exist (existing datasets like BioFors predate generative AI); (c) a detector aimed specifically at band scale edits, compared head to head with SynthID on the same publishing steps. The detector alone is **not** novel (Proofig and ImageTwin sell blot forensics), so never pitch it as the innovation.

**Why it fits the theme:** SynthID Detector went public Oct 7, 2026, and Gemini's current image tools (Gemini 3 family, including Nano Banana Pro from Nov 2025) are both inside the 12 month window and central. Note: the free Gemini app only offers "Auto (Gemini 3 models)," so we cannot claim which exact image model was used. Say "Gemini's image tools."

---

## 2. HARD CONSTRAINTS (read before doing anything)

1. **Daniel's AKT/pAKT blot (the "NRG-1 / AKT1/2 inhibitor" figure) must NOT be used anywhere in the hackathon:** not in the repo, slides, video, Devpost or demo. It has been removed from the code and data. Never re-add it. Every SynthID row that came from it was deleted from `results/synthid_manual.json`.
2. **SynthID Detector terms of service** forbid (a) automated or programmatic submissions and bulk verification, (b) using it to optimize transformations that break or evade the watermark, and (c) evading quotas. So: upload by hand only, fixed realistic scenarios only, never an iterative "make it sneakier until it passes" loop against SynthID. Any adversarial loop must target **our own** detector. Frame findings as "where the watermark stops working," never as an evasion recipe. Keep exact crop sizes and steps off slides.
3. **CC BY attribution:** all real blot figures come from CC BY papers via Europe PMC. Metadata (PMCID, DOI, title, license, caption) is in `data/real_figures/metadata.json`. Every forgery was made by us. Always say "we made this edit; the published figure is unaltered; the authors are not implicated."
4. **No unpublished lab data** anywhere.
5. **Honesty:** results are modest. State exact numbers from `results/*.json`. Never round up, and never show a demo that implies more than measured. A flag means "look closer," never "proof of misconduct."
6. **AI assistance disclosure:** the rules don't address AI tools, but Sunday's schedule includes a "How to Use Claude" workshop. The README states the project was built with Claude as a coding assistant. Keep that statement, and make sure the team can explain every component (see `docs/METHODS.md`).

---

## 3. Repository map

```
README.md                  public facing readme (results table auto-written by scripts/results_table.py)
HANDOFF.md                 this file
requirements.txt           Python deps (plus jax[cpu] and optax for the CNN)
btw/                       the Python library
  imgio.py                 loading, grayscale, polarity normalization (bands dark on light)
  bands.py                 band finder: lane profile peaks (find_bands_lanes) with a blob fallback
  panels.py                cuts blot panels out of a figure (non white rectangles plus a blot likeness score)
  publish.py               publishing transforms: resave, resize, JPEG, contrast, PDF round trip, random chain
  splice.py                forgery generator: copy_move, cross_splice, ai_splice, intensify, erase, with pixel masks
  sources.py               loads the AI donor blots (data/ai_sources), cropped to panels
  features.py              hand built per window forensic features (relative to the same panel)
  detector.py              Detector (gradient boosting), CnnDetector (JAX CNN), load_best() picks the better one by AUROC
  cnn.py                   pure JAX CNN: patch extraction (2 channels), model, predict, save/load (.npz)
  duplicates.py            copied band detector (noise residual template matching, with flips)
  provenance.py            C2PA reader (c2pa-python), falls back to byte search
  report.py                analyze(bytes, filename) -> report JSON for the app (figures, panels, heat, bands, duplicates, C2PA, SynthID note)
  api.py                   FastAPI: POST /api/analyze, GET /api/examples, GET /api/results, serves web/dist
scripts/
  collect_europepmc.js     browser console script that downloads CC BY blot figures from Europe PMC into one JSON bundle
  build_benchmark.py       bundle(s) -> data/real_figures -> data/real_panels (+ keep.json filter) -> data/benchmark
  train_detector.py        gradient boosting baseline -> results/detector_metrics.json, models/splice_hgb.joblib
  train_cnn.py             CNN -> results/cnn_metrics.json, models/splice_cnn.npz (SKIP_HELDOUT=1 to reuse the v1 held out number)
  eval_robustness.py       one publishing step at a time on AI splices -> results/robustness.json
  eval_duplicates.py       copied band detector on exact copies -> results/duplicates_metrics.json
  make_examples.py         precomputes demo reports into examples/*.report.json and examples/examples.json
  results_table.py         writes the README results table, the README benchmark size and the DATA_CARD table from results/*.json and data/benchmark/index.json (never edit those numbers by hand)
laptop/
  gemini_blot_library.py   generates more AI blots via the Gemini API (needs GEMINI_API_KEY; model name configurable)
  claim_link.py            given a PDF and a figure number, lists the sentences that rely on it (Gemini API)
web/                       React + Vite frontend (src/App.jsx, src/styles.css)
data/
  real_figures/            CC BY figures (*.jpg) + metadata.json (120 figures from bundle 1)
  real_panels/             panels cut from figures + panels.json + keep.json (241 kept panels, 55 papers)
  ai_sources/              gemini_clean_blot.jpeg, gemini_film_blot.jpeg (whole AI blots we generated)
  benchmark/               GENERATED, not committed (rebuild with build_benchmark; fixed seed)
models/                    splice_cnn.npz (current), splice_cnn_v1.npz (round 1), splice_hgb.joblib (baseline)
results/                   every reported number (json) + logs
examples/                  demo inputs: alk2_ai_band.jpg (CC BY figure with an AI band WE added), alk2_original.jpg, gemini_whole_blot.jpeg
docs/                      METHODS.md (explainer for the pitch), DATA_CARD.md, devpost.md (draft), video_script.md, disclosure_email.md, pitch_outline.md (slide by slide content, speaker notes, likely judge questions)
```

---

## 4. The SynthID experiment (manual, by hand)

File: `results/synthid_manual.json` (rendered on the app's "The gap" page).

Done (all on images we generated ourselves with Gemini, no private data):

| Scenario | SynthID | C2PA credential |
|---|---|---|
| Whole AI blot, original download | detected | present ("Created by Google Generative AI" + "Applied imperceptible SynthID watermark", signed by Google) |
| Whole AI blot, re-saved / resized 50% / JPEG 60 / 70% crop / contrast / as 1 panel of a 2x2 figure / PDF round trip | detected (all) | stripped (all) |
| Whole clean style AI blot, re-saved / shrunk to figure panel size 344x192 JPEG | detected | stripped |
| Single AI band crops (155x65, 85x120, 195x80, 180x75) | "not enough details" | stripped |
| A flat synthetic control drawn in Python | "not enough details" (inconclusive) | none |

**PENDING (Daniel must do this by hand):** files in `C:\Users\danie\Downloads\gatorhack_synthid\round4\` (copies are in `examples/` for S1 and S0):
- `S1_ccby_alk2_ai_band.jpg`: AI band pasted into the ALK2 figure (PMC10376983, CC BY)
- `S2_ccby_PMC10674287_ai_band.jpg`, `S3_ccby_PMC10033651_ai_band.jpg`, `S4_ccby_PMC10243194_ai_band.jpg`: AI bands pasted into other CC BY figures
- `S0_ccby_alk2_original_control.jpg`: untouched original, the negative control

Upload each at synthid.com by hand and record the exact verdict wording ("detected", "not detected", "not enough details") in `results/synthid_manual.json` (rows S0 to S4, field `synthid`). Then rerun `python -m scripts.make_examples` so the hero shows the real verdict. The app shows "check pending" until then and never claims an unverified result. If more evidence is wanted, make more composites (see section 7) and upload them by hand; that is allowed (fixed scenarios), iterating to evade is not.

**Disclosure:** `docs/disclosure_email.md` is a draft note to Google (via the synthid.com feedback icon, or bughunters.google.com). Update it with the CC BY results, then Daniel sends it from his account. It's good to say "we reported this to Google" in the pitch, but only once it's sent.

---

## 5. Data and benchmark (current state)

- **Collection:** Europe PMC REST API, query `(METHODS:"western blot" OR METHODS:"immunoblot") AND OPEN_ACCESS:y AND LICENSE:"cc by" AND SRC:MED AND HAS_FT:y`, figures whose caption mentions western blot or immunoblot, up to 2 per paper. Images come from `https://europepmc.org/api/fulltextRepo?pmcId=..&type=FILE&fileName=<graphic href>.jpg&mimeType=image/jpeg` (web resolution, about 750 px wide). Our build machine could not reach these hosts, so collection ran as a browser console script (`scripts/collect_europepmc.js`) that downloads one JSON bundle. Chrome blocks repeated automatic downloads per site, so run it on a site that hasn't downloaded yet (https://example.com works).
- **Bundle 1** (in repo, unpacked): 120 figures, 79 papers, years mostly 2023. After panel extraction: 360 panels, **241 kept** (from 55 papers) by `filter_panels()` (blot score >= 0.72, grey, at least 2 bands, not text heavy, not tiny or huge).
- **Bundle 2: SAVED but not yet used** (see LIVE STATUS): `C:\Users\danie\Downloads\btw_real_blots_bundle2.json`, 440 figures from 241 new papers (2015 to 2022). If the file exists, run `python -m scripts.build_benchmark --bundle path/to/btw_real_blots_bundle2.json --n-clean 6 --n-forged 3`. That merges metadata, re-extracts and filters all panels, and rebuilds the benchmark. Then retrain (section 6). More papers should help most with the false flag problem, since the clean false flag rate rises on unseen papers. If it didn't save, rerun the script in a browser.
- **Current benchmark** (built from bundle 1, `--n-clean 6 --n-forged 3`): 4,725 images (3,279 forged, 1,446 clean); train 2,654 / test 2,071; split by paper with an md5 hash (about 30% test). Each forgery has a mask in `data/benchmark/masks`. Median panel is only about 7,200 px, and bands are often about 13x6 px. **That's the core difficulty.**
- `docs/DATA_CARD.md` numbers are now written automatically by `python -m scripts.results_table` from `data/benchmark/index.json`.

---

## 6. Detectors: what was tried and the honest numbers

All numbers are on held out test papers (never seen in training). The flag line is set so 5% of clean calibration or train panels are flagged.

1. **Gradient boosting on hand built features** (`train_detector.py`). Image AUROC **0.54 to 0.55**, close to chance. It learned "this window is a band" (a shortcut). Hard negatives and band relative features didn't fix it. Window localization within forged images was decent (0.84 to 0.87). Kept as the baseline. Its metrics file is from the older benchmark and needs a rerun on the final benchmark.
2. **CNN v1** (`train_cnn.py`, 24 px patches, 2 channels: intensity relative to panel background + noise residual normalized by panel noise; about 40k params; hard negatives = real band patches), on the 3,161 image benchmark (`results/cnn_metrics_v1.json`):
   - image AUROC **0.658** (small panels 0.664, medium 0.70, large 0.41 with n=51)
   - caught at the flag line: AI splice **43%**, intensify 48%, cross splice 35%, copy move 29%, erase 18%
   - clean test panels flagged: **12.7%** (the threshold doesn't transfer perfectly to new papers)
   - **held out method** (trained with NO AI splices, tested on AI splices): AUROC **0.717**, 38% caught at 7.5% false flags
3. **CNN v2 = CURRENT MODEL** (32 px patches, augmentation, threshold and score rule chosen on 6 separate calibration papers; 4,725 image benchmark; `results/cnn_metrics.json`, `models/splice_cnn.npz`):
   - image AUROC **0.643** (small 0.645, medium 0.70, large 0.52 with n=77), essentially tied with v1
   - **the flag line now transfers honestly: clean test panels flagged 2.7%** (v1: 12.7%)
   - caught at that stricter line: AI splice **22%**, intensify 37.5%, cross splice 19%, copy move 16%, erase 2%
   - held out method number in the file is copied from v1 (marked with a note); rerun without SKIP_HELDOUT for a v2 value
   - **Recommendation:** keep v2 because its false flag rate is honest. Quote v1 as an alternative operating point only if clearly labeled. To restore v1: `cp models/splice_cnn_v1.npz models/splice_cnn.npz` and `cp results/cnn_metrics_v1.json results/cnn_metrics.json` (v1 used P=24; the model is fully convolutional, so it runs at P=32 too).
   - **How to say it in the pitch:** "At a 3% false flag rate on papers it never saw, it catches about 1 in 5 pasted AI bands, a case SynthID cannot check at all. It's a first baseline, and the benchmark is open so others can beat it."
   - **Robustness of v2** (`results/robustness.json`, 103 held out test panels, one Gemini lane pasted into each, then ONE publishing step): caught 38% with no step, 38% re-saved, 39% JPEG 85, 36% JPEG 60, 38% contrast, 39% PDF round trip, but only **2% after resizing to 50%**; clean panels flagged 2% to 6%. Honest reading: on whole lane AI splices it gives a real signal that survives compression and PDF export, which SynthID cannot do on a single band, but downscaling destroys it.
   - **Demo caveat (v2):** on the hero composite (`examples/alk2_ai_band.jpg`) v2 does NOT flag the panel: the pasted bands score high (ranks 2 to 4 of the panel's bands) but a real, very dark BMP4 band scores highest, and the untouched original gets flagged for that same band. So v2 has a "very dark band" bias. That's why v3 tries panel relative rules (below). Never fake the hero.
5. **CNN v3 (code ready, training pending on the bigger benchmark):** scores in logit space (v2's probabilities saturated at 0.997, which made thresholds meaningless), picks one of three panel rules on calibration papers (`top`, `outlier` = minus the panel median, `z` = divided by the panel spread), saves `space="logit"` and a `scale` in the model file so the app shows every score on a 0 to 1 display scale with the flag line at exactly 0.5, stores patches as float16 to fit 5x more data in RAM, and reports `top_patch_on_edit` (how often the most suspicious patch in a forged panel lies on the edit). `btw/detector.py` loads both v2 (prob space) and v3 (logit space) models. Test a code change quickly with `SMOKE=60 EPOCHS=1 python -m scripts.train_cnn` (prints, writes nothing).
6. **Copied band detector** (`eval_duplicates.py`): finds 31 of 35 exact copies when nothing is resized, but at a calibrated threshold on web resolution panels it catches only about 11% (5% false flags) with no publishing step, and about 2% after JPEG, PDF or resize. **Present it as a supporting check with this limitation, not a feature that works.** A widened self exclusion zone (ignores a band's own mirrored halves) was added after this measurement; rerun to update.

**Ideas that may improve the detector (if time):** use bundle 2 (more papers); more epochs with early stopping on calibration AUROC; ensemble CNN + GBM scores; band level classification (crop each detected band plus a ring, compare to the other bands of the same panel); higher resolution figures (publishers' full size images are not easily reachable).

---

## 7. How to run everything

```bash
pip install -r requirements.txt
pip install "jax[cpu]" optax          # CNN
# data (only if you have new bundles):
python -m scripts.build_benchmark --bundle btw_real_blots_bundle2.json --n-clean 6 --n-forged 3
# or rebuild the benchmark from the committed figures/panels:
python -m scripts.build_benchmark --skip-panels --n-clean 6 --n-forged 3
python -m scripts.train_cnn            # about 50 min on 2 CPU cores; SKIP_HELDOUT=1 skips the held out retrain
python -m scripts.train_detector       # baseline, about 10 min
python -m scripts.eval_robustness      # robustness table with the best detector
python -m scripts.eval_duplicates
python -m scripts.make_examples        # rebuild the demo reports after any model change
python -m scripts.results_table        # rewrite README results from results/*.json
# app
uvicorn btw.api:app --port 8000        # serves web/dist if built
cd web && npm install && npm run build # or npm run dev (port 5173, proxies /api)
```

Making more SynthID composites from CC BY figures: see the code pattern in section 4's files. In short, `btw.splice.forge(panel, "ai_splice", rng, ai_panels=btw.sources.ai_panels())` on a panel cropped from a figure in `data/real_figures`, paste it back into the figure, then `btw.publish.jpeg(fig, 88)`. Whole lane crops from `data/ai_sources/gemini_clean_blot.jpeg` look most realistic (the hero used rows 215:350, cols 690:910).

---

## 8. TODO list (in priority order)

1. **Daniel:** upload round 4 to SynthID (section 4) and record the verdicts. This is the headline finding; without it the pitch must say "single bands are too small to check," which is weaker but true.
2. Check the CNN v2 results; keep whichever of v1 and v2 is better (by test AUROC and AI splice detection at the flag line).
3. DONE for v2: `python -m scripts.eval_robustness`. Rerun it after any new model.
4. Rerun `train_detector.py` on the final benchmark so the baseline comparison is fair (same data).
5. Optional: add bundle 2 and retrain (more papers, should reduce false flags).
6. `python -m scripts.make_examples` then `python -m scripts.results_table`. Update `docs/DATA_CARD.md` numbers, `docs/devpost.md`, `docs/video_script.md` placeholders, and the "Results" in README.
7. Look at the app (`npm run build` then `uvicorn btw.api:app`): hero, report, Gap page. Heat now shows only above the flag line (Ponceau pink). Check the hero example result after the final model; if the detector does not flag the hero composite, do not fake it. Say so, or pick a different honest example.
8. Deploy (optional): Hugging Face Spaces (Docker) or Render. The FastAPI app serves the built frontend. A public link helps judges.
9. Slides for Monday (about 5 min pitch): title; threat (Gemini edit screenshot of OUR AI blot); SynthID test matrix; why band scale matters (Bik et al. 2016: 3.8% of 20,621 papers had inappropriate image duplication); what we built; live demo; honest results vs baseline; limits and ethics (terms, CC BY, disclosure); next steps.
10. Record the 3 min video (`docs/video_script.md`), write the Devpost entry (`docs/devpost.md`), enter tracks: Best AI Hack, Best Use of ML (DSI), Best Use of Gemini (MLH), Best Social Impact, Best Design. Make sure the repo is public. Submit by Sunday 4pm.
11. Send the disclosure email (after the CC BY results).

---

## 9. Design system (frontend)

Concept: a **film lightbox** (blots are read on a backlit viewer). One family, Archivo (variable width: expanded and heavy for headlines, normal for the UI, tabular numbers). Palette: lightbox white `#F6F8F9`, membrane slate ink `#1F2B33`, X ray film blue grey `#6E8796`, clean wash `#DCE7EC`, **Ponceau S pink `#C8285F` = suspicion** (from the red protein stain used on membranes). Left aligned, two columns: figure on the lightbox left, a rail of checks right. Avoid: all caps eyebrows, monospace labels, identical card grids, arrows on buttons.

Pages: **Check a blot** (hero with the CC BY example, drop zone, examples, report), **The gap** (SynthID matrix + detector stats + robustness table), **How it works** (three layers + limits + benchmark).

---

## 10. Sources worth citing

- Gator Hack V: https://gatorhack2026.devpost.com/ , https://ufgatorhack.com/
- SynthID Detector opened to the public Oct 7, 2026: https://www.ithome.com.tw/news/179517
- Nano Banana Pro (Gemini 3 Pro Image), Nov 20, 2025: https://techbuzzireland.com/2025/11/20/google-introduces-nano-banana-pro/
- Bik, Casadevall, Fang 2016, "The Prevalence of Inappropriate Image Duplication in Biomedical Research Publications" (mBio): 782 of 20,621 papers (3.8%): https://www.biorxiv.org/content/10.1101/049452.full.pdf
- BioFors (ICCV 2021), biomedical image forensics benchmark, predates generative AI: https://arxiv.org/abs/2108.12961v1
- Proofig AI generated image detection (commercial, closed): https://www.proofig.com/ai-generated-image-detection/
- "Blind spots on western blots" (PLoS Biol 2022), PMC9518894: background on western blot reporting problems
- Last year's winner (Plato's Cave): https://github.com/matheusmaldaner/PlatosCave , paper https://arxiv.org/abs/2603.23526
