# Independent audit: Below the Watermark (GatorHack)

**Audited commit:** `f1371fe` on `main` (Oct 24, 2026, 8:17pm ET). **Fixes:** branch `audit/claude-2026-10-24`, [PR #5](https://github.com/danielsemenovitch/GatorHack/pull/5).
**Auditor:** Claude (Anthropic), as an independent reviewer at Daniel's request. Every claim below was checked by running code, not by reading the handoff.
**Environment:** Linux container, Python 3.13.17 (the handoff targets 3.11), JAX 0.12, Node 22, Chromium through Playwright, FastAPI serving the built React app. Windows was not tested.

## Round 3 (Oct 24, late night): submission pass

- SynthID round 4 is recorded from Daniel's hand uploads: all 4 Gemini bands pasted into published blots came back "not detected", and so did the unedited control (`results/synthid_manual.json`).
- The reuse demo no longer depends on a file someone has to make: the "A demo PDF with one figure used twice" button runs `examples/demo_reuse.pdf` (built by `scripts/make_demo_pdf.py` from two CC BY figures, with the reuse stated on its title page). The pair now appears at the top of the report, and each figure in the pair is tagged.
- The landing page quotes the headline benchmark (72 held-out papers: 35% of pasted AI bands caught at 4% clean flags), the same numbers as the deck and Devpost. The strict line figures still come from the 20 paper resolution audit.
- Known limit found while building the PDF: when a figure is shrunk to 90% size, two blot strips can merge into one framed box that the glyph guard then skips as "text heavy". The demo copy uses 95%; a real fix needs the panel finder to split stacked strips.
- 29 tests pass; build clean; no console errors on any page or example at 1440, 1280 and 390 px.

## Round 2 (Oct 24, about 11pm ET): team decisions applied, open items fixed

| Item | What changed | Evidence |
|---|---|---|
| Default threshold | Research threshold (0.50) is the default in the API, `report.analyze` and the UI; the 0.90 line is now the optional "strict" setting, labeled with its measured 3% catch rate | `test_research_threshold_is_the_default` |
| P2-6 single bands as panels | A blob inside an elongated framed strip is part of that strip; the strip itself becomes the panel | Oncotarget: 11 single band crops before, 0 after; `test_framed_strip_bands_are_not_separate_panels` |
| P2-7 frame lines in crops | Frames thicker than 2 px are trimmed fully (plus halo); 1 to 2 px frames keep the old 2 px margin so web resolution crops stay identical to the training panels; arrows and labels touching a frame are cut off; text heavy framed boxes (tables) are not routed | Bundle 1 figures: 240 of 241 reference panels still found, automatic routing of the reference panels 130 of 241 (same as before); the ALK2 example scores are unchanged to 3 decimals |
| P2-8 CORS | Only the Vite dev origin, or origins in `BTW_CORS_ORIGINS` | `test_cors_is_not_open_to_every_origin` |
| Resource limit | 150 MP total decode budget per upload, overflow listed as skipped | `test_total_pixel_budget_limits_one_request` |
| Offline demo | Archivo font self-hosted (no Google Fonts request); page description no longer claims it "finds AI edits that watermark detectors miss" | UI walk shows zero console errors |
| Stale examples | `examples/*.report.json` regenerated with current code | `python -m scripts.make_examples` |
| Second demo example | `examples/pmc10315040_fig5_ai_band.jpg`, labeled as **selected** from the end-to-end test below, with its ground truth box | The CNN flags the edited IL-6 panel and nothing else |
| Chance flags | The report now states how many flags to expect by chance (about 5% of screened panels at the research threshold) | UI and HTML export |

**New end-to-end result (`scripts/eval_figure_level.py`, `results/figure_level_eval.json`):** one Gemini band pasted into one routed panel of each of 22 held-out figures from 15 test papers, saved as JPEG 90, analyzed exactly like an upload.

| | Research threshold (default) | Strict |
|---|---|---|
| Edited panel flagged | 7 of 22 | 0 of 22 |
| Edited figures with an untouched panel flagged | 0 of 22 | 0 of 22 |
| Unedited originals with any flag | 1 of 22 | 0 of 22 |

**Final Oncotarget numbers (default settings):** 30 panels screened, 2 flagged, both at 0.512, just over the 0.50 line (one is the p62 strip, now screened as a whole strip for the first time). That matches the chance rate of about 1.5 flags for 30 unedited panels; it is not evidence of editing.

**Methodological note found in round 2:** the automatic forgery generator sometimes pastes shapes that do not look like bands (a grey smear, a black wedge). Those easy cases are in the benchmark and inflate its recall. Filter or regenerate them before training a v4.

## Bottom line

The app runs, the tests pass, and the HTML export is safe. But at `f1371fe` it had one crash bug that any judge could hit with a different PDF, a resolution mismatch that produced most of the false Western blot flags on real papers, and several displays that said more than the evidence supports. All of those are fixed on the branch with regression tests. The detector itself is still weak, the default "conservative" setting catches almost nothing, and the demo example shows a miss plus a false positive. The pitch has to be built around that honestly (see the demo path at the end).

## What was actually run

| Check | How | Result |
|---|---|---|
| Backend tests | `python -m pytest -q tests` | main: 17 passed. Branch: 25 passed (8 new). The 8 new tests all **fail on main** and pass on the branch. |
| Frontend build | `cd web && npm ci && npm run build` | passes |
| Full UI walk | Playwright: every page at 390, 768 and 1440 px; three example buttons; file upload; HTML download | no horizontal overflow, every control labeled, every image has alt text. The only console error is Google Fonts being unreachable from the sandbox (the font falls back offline). |
| Oncotarget PDF (`oncotarget-09-4161.pdf`, the exact file from Daniel's Downloads, 8,197,373 bytes) | API and UI, v3 conservative, v3 research, v2, automatic and user-confirmed blot modes | see "Oncotarget results" |
| Malformed inputs | empty file, corrupt JPEG, text renamed .png, PDF with a grayscale image, PDF with no images, PDF with a logo on every page, PDF with a 36 MP image, 27 MB file | see P0/P1 table |
| HTML export | downloaded from the UI, opened offline, injected a hostile filename and labels | no script, event handler, `javascript:` or `data:text/html` survives escaping |
| Resolution experiment | `python -m scripts.audit_resolution --pdf oncotarget-09-4161.pdf` | `results/audit_resolution.json` |
| Heat geometry | held-out forged panels with known masks, pasted into small and large figures, run through `report.analyze` | overlays line up with panel boxes to the pixel; see "Heat maps" |

**Could not verify:** a hosted site (none exists); Windows setup; SynthID round 4 (manual only, still pending); any real-world false positive rate (no labeled real papers exist); v3's own expanded benchmark (not in the repo), so held-out numbers below use the bundle 1 test papers, which v3 never trained on.

## Bugs, by priority

| ID | Problem | Reproduce | Expected vs actual (main) | Where (main) | Status |
|---|---|---|---|---|---|
| **P0-1** | Any PDF with a grayscale embedded image fails completely | Upload a PDF whose figure is stored as grayscale (common for blots) | Expected: analysis. Actual: HTTP 422 "not enough image data", nothing analyzed | `btw/report.py:62-66` (`Image.frombytes("RGB")` on a 1 channel pixmap) | **Fixed**: every pixmap converted to RGB (`_pixmap_rgb`); one unreadable image is listed and skipped instead of failing the PDF |
| **P1-1** | CNN scored PDF figures far above training resolution | Oncotarget figures are 1,038 to 2,028 px wide; training figures are 682 to 791 px (10th to 90th percentile) | Expected: benchmark like false flag rate. Actual: 18 of 36 panels flagged at the research threshold (benchmark clean rate is about 4%) | `btw/report.py:177` | **Fixed**: panels from figures over 937 px wide are scored on a copy resized to a 750 px figure; heat and band boxes mapped back (`_score_at_training_scale`). Evidence below. |
| **P1-2** | Journal logo on every page reported as image reuse | 4 page PDF with the same logo | Expected: no finding. Actual: 6 "identical image" pairs and a "suspicious" verdict (12 pages would give 66) | `btw/report.py:57-67` | **Fixed**: one object drawn at the same spot on 3+ pages is reported once as a page decoration; the same image in two different figures is still flagged |
| **P1-3** | Spoofable C2PA result | A file containing the bytes `c2pa`, `jumb`, `Google`, `Created by Google Generative AI` | Expected: unverified. Actual: "Content Credentials found, signed by Google LLC. They declare this image AI generated." | `btw/provenance.py:36-45` | **Fixed**: byte matches are reported as "could not be read or verified"; real manifests state integrity and that the signer is not checked against a trust list |
| **P1-4** | Pink heat and hot band boxes on panels the report says are not flagged | Oncotarget, v3 conservative | Expected: highlights only where flagged. Actual: pink boxes on β-actin bands in unflagged panels, contradicting "2 review flags" | `web/src/App.jsx:56-60` | **Fixed**: heat and hot outlines only on flagged panels; clicking a locally high band in an unflagged panel says so |
| **P1-5** | Images skipped silently | PDF with a 36 MP figure; PDFs with more than 48 figures | Expected: "not analyzed" notice. Actual: decoded the whole image (memory), then dropped it; verdict "no_blots" | `btw/report.py:58-62` | **Fixed**: size checked before decoding; every skipped image is listed with a reason in JSON, UI and export |
| **P1-6** | Gap page showed v2 numbers while the app runs v3, plus a v1 result as if current | Open "The gap" | Expected: numbers for the default model. Actual: v2 AUROC 0.643 and 22%, and "AI splice with no AI splices in training 38%" copied from v1 | `web/src/App.jsx:364, 428` | **Fixed**: v3 leads, v2 shown as legacy, copied v1 row hidden, robustness table uses v3 |
| **P1-7** | Demo example loses its attribution and hides the truth | Click "See this example", download the report | Expected: CC BY credit and "we made this edit". Actual: neither in the report or export, and no way to see where the edit is | `web/src/App.jsx:274` | **Fixed**: credit and the known edit outline (from our own mask) carried into the report and export, with an exact sentence saying whether the edited panel was flagged |
| **P1-8** | AI declaring credential not treated as a finding | Click "A whole blot made by Gemini" | Expected: headline. Actual: "Figure inventory and available checks", credential buried as "Unverified credential" | `btw/report.py:220` | **Fixed**: counts as a review finding; headline "Content Credentials declare this image AI generated" |
| P2-1 | Analysis blocked the server | Request `/api/health` while the Oncotarget PDF is analyzing | main: health answered after 30.7 s; branch: 28 ms | `btw/api.py:30` (`async def` doing CPU work) | **Fixed**: plain `def` endpoint runs in the threadpool; upload read is capped at 25 MB + 1 byte |
| P2-2 | Internal error text shown to users | Upload `text.png` | "cannot identify image file <_io.BytesIO object at 0x...>" | `btw/api.py:37` | **Fixed**: readable messages; internals go to the server log |
| P2-3 | HTML export showed a transparent heat PNG beside each crop | Download any report with panels | Heat could not be related to the crop | `web/src/reportExport.js:48` | **Fixed**: heat laid over the crop for flagged panels; panel boxes and the known edit drawn on each figure |
| P2-4 | Upload controls wrapped apart from their labels | Desktop, 1440 px | "Alert sensitivity:" on one line, its menu on the next | `web/src/App.jsx` controls row | **Fixed** |
| P2-5 | "Download for SynthID" on crops of real published blots | Any flagged panel | SynthID cannot say anything useful about a 2017 journal figure | `web/src/App.jsx:181` | **Fixed**: "Download panel crop" |
| P2-6 | Single bands routed as whole "panels" | Oncotarget p4, p8, p9 | Tiny crops (99×22 to 129×57 px) of one band are screened as blots and boxed in the UI | `btw/report.py:_blot_candidates`, `btw/panels.py` | **Fixed in round 2** (frame-aware panels) |
| P2-7 | Published frame lines inside crops | Oncotarget strips at 2,000 px | The fixed 2 px trim leaves 4 to 6 px black frames; heat clustered on frame and band junctions before P1-1 | `btw/panels.py:56` | **Fixed in round 2** |
| P2-8 | CORS allows any origin | n/a | Fine for a local demo; restrict before any public deployment | `btw/api.py:21` | **Fixed in round 2** |

## Scientific statements that were not supported (now corrected)

- "Catches pasted bands, AI or not, and bands darkened or erased in place" (How it works). v3 catches 20% of erased bands and about a third of the rest at the research line. Rewritten to state it catches a minority.
- "We are reporting what we saw to Google" (How it works). The disclosure email has not been sent. Removed.
- "found in 3.8% of 20,621 papers" attached to band copy and paste. Bik et al. measured inappropriate duplication across all image types. Reworded.
- README: "Scientific image fraud happens one band at a time" softened; the results table was v2 under a v3 default and claimed the flag line was "set on training data". Regenerated from `results/*.json` by `scripts/results_table.py`, v3 first.
- README and handoff quoted "18/37 to 2/37" as the effect of the conservative mode. That reduction came from cutting the tail of a resolution shifted score distribution, and the conservative line costs almost all sensitivity (next section).

- README and handoff cited `results/duplicates_conservative_sanity.json` for the copied band numbers (6.86% caught, 1.96% false flags), but the file was never committed. Regenerated with `python -m scripts.eval_precision_band_duplicates`; it reproduces the same 204 pairs and rates exactly, and is now in `results/`.

## Oncotarget results (exact PDF, before and after)

8 embedded figures (all RGB, 1,038 to 2,028 px wide); 36 blot panels routed in automatic mode, 38 in user-confirmed mode.

| Setting | main `f1371fe` | branch |
|---|---|---|
| v3, conservative (0.90) | 2 of 36 flagged | 0 of 36 |
| v3, research (0.50) | 18 of 36 flagged | 0 of 36 |
| v2 (0.9966) | 18 of 36 flagged | 0 of 36 |
| median v3 panel score | 0.512 | 0.248 |
| page 10 grayscale microscopy | excluded | excluded (both blot modes) |
| phase contrast, crystal violet, immunofluorescence, TEM figures | not routed | not routed |

This paper has no ground truth labels, so "0 flags" does not mean it is clean. It shows the old flags were driven by image scale, which the controlled experiment below confirms.

## The resolution experiment (`results/audit_resolution.json`)

Held-out bundle 1 test papers (never seen by v3), every second image: 318 clean, 138 AI splice, 156 intensify, 131 cross splice.

| Input to v3 | AUROC | Clean flagged @0.50 | AI splice caught @0.50 | AI splice caught @0.90 |
|---|---|---|---|---|
| Web resolution, as trained | 0.740 | 6.6% | 38.4% | **2.9%** |
| Same images enlarged 2.5x (like a publisher PDF figure) | 0.622 | 18.2% | 32.6% | 0% |
| Enlarged, then resized back (the fix) | 0.701 | 3.1% | 21.0% | 0% |

Two conclusions. First, high resolution input nearly triples clean false flags, and the fix removes that while restoring most of the AUROC; it costs some recall, because a resize round trip blurs the evidence. Second, **the default conservative line (0.90) caught 2.9% of AI splices and 8.3% of intensified bands even at native resolution.** It is close to blind. The UI now prints these measured rates under the setting menus. I did not change the default, because that is a product decision, but I recommend making the research threshold the default now that the resolution bug is fixed.

## Heat maps

- **Geometry is correct.** Overlay boxes match panel boxes exactly at every tested width (Playwright compared bounding boxes); the heat PNG always has the panel's pixel size, including through the new scaling path (unit test).
- **Localization is coarse but real.** Held-out AI splice panels with known masks, run through the whole pipeline: the hottest heat pixel fell within 16 native pixels of the true edit in 7 of 8 flagged panels at web size, and 5 of 6 when the panel was enlarged 2.5x inside a 2,000 px figure.
- **What misled before:** on high resolution PDFs, heat sat on band edges, gaps between saturated bands and published frame lines (contact sheet in the audit notes), and appeared on unflagged panels. Both are fixed (P1-1, P1-4).

## Demo example (ALK2, our own edit)

v3 does **not** flag the panel we edited (score 0.424). At the research threshold it **does** flag the untouched loading control below it (0.620), in both the edited and the original figure. The conservative setting flags nothing. The example now draws a dashed "known edit" box and states this in a sentence. Do not present this example as a detection.

## What works

- Whole image reuse within one PDF (exact and near duplicates) with side by side evidence; tested on controlled PDFs.
- Microscopy routing on the test paper, in both blot modes.
- C2PA reading on the Gemini example (manifest integrity passes; signer certificate issued to Google LLC; AI generation declared).
- HTML export: self contained, printable, escaped, now with overlays and attribution.
- Clear "not assessed" states, panel limit accounting, skipped image accounting.

## Recommended 3 minute demo path

1. **The finding (Gap page):** whole AI blots detected by SynthID through every publishing step; single bands "not enough details"; bands pasted into published blots "not detected", 4 of 4.
2. **Provenance:** click "A whole blot made by Gemini". The headline reads that the credential declares AI generation, with the honest caveat that any re-save removes it.
3. **Whole image reuse:** click "A demo PDF with one figure used twice"; the pair is at the top of the report. Download the HTML report.
4. **The detector, honestly:** click "Another AI band the model catches (selected)": the dashed box is our edit and the flagged panel is on it. Then click "Run our checks on this blot" (ALK2): a miss. Then the Gap page sentence: end to end on 22 held-out figures it flagged 7 edited panels and 1 of 22 unedited figures.
5. Optional: the Oncotarget PDF, to show a real paper running end to end in about 20 seconds, with the chance flag note explaining why 2 flags at 0.512 are expected noise.

## Improve next (in order)

1. Fill SynthID round 4 and the pasted band rows before submitting.
2. Regenerate the forgery benchmark without the implausible pastes (see round 2 note), then train v4 on figures at several resolutions instead of resizing at inference, with more AI generators and real hard negatives.
3. Calibrate the research threshold per paper on 10 to 20 unedited published papers, so the chance flag estimate is measured rather than assumed from panels.
4. Before any public deployment: put the app behind HTTPS, set `BTW_CORS_ORIGINS`, and add a per-request time limit (the 150 MP budget bounds memory, not time).
