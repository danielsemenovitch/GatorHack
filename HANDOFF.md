# HANDOFF: Below the Watermark (Gator Hack V, UF AI Days 2026)

This file is the complete state of the project as of **Saturday Oct 24, 2026, about 2:25pm ET** (it is updated after every work step; the "Live status" section right below says what is running right now), written so a teammate or another AI assistant can continue without any other context. Read it top to bottom once before touching anything.

---

## LIVE STATUS (read first)

<!-- LIVE -->
- **Stopped at 2:25pm Sat** (Claude usage ran out). Everything below is pushed.
- **Bundle 2 is SAVED** at `C:\Users\danie\Downloads\btw_real_blots_bundle2.json` (72 MB: 440 CC BY figures from 241 NEW papers, 2015 to 2022; PMCIDs from bundle 1 were skipped). Bundle 1 is also there as `btw_real_blots_bundle.json`. **Next step, top priority:** copy bundle 2 into the repo folder (do NOT commit the json, it is big; the unpacked figures in `data/real_figures` are fine to commit) and run
  `python -m scripts.build_benchmark --bundle btw_real_blots_bundle2.json --n-clean 3 --n-forged 1`
  (fewer variants per panel because there are about 5x more panels; keeps CNN training near 1 hour on 2 CPU cores). Back up v2 first (`cp results/cnn_metrics.json results/cnn_metrics_v2.json; cp models/splice_cnn.npz models/splice_cnn_v2.npz`), then `EPOCHS=6 python -m scripts.train_cnn` (CNN v3), `python -m scripts.train_detector`, `python -m scripts.eval_robustness`, `python -m scripts.eval_duplicates`, `python -m scripts.make_examples`, `python -m scripts.results_table`, `cd web && npm run build`. Compare v3 with v2 (test AUROC, AI splice caught at the flag line, clean false flags) and keep the better one honestly.
- **Daniel said yes** to generating about 5 more AI blots in Gemini (X ray film, digital imager white background, fluorescent greyscale, faint low contrast, doublet bands; no labels) for `data/ai_sources/`. Not done. Do it BEFORE rebuilding the benchmark if there is time; it diversifies the AI splice class.
- The Gap page sentence under the side by side table ("keeps its signal through compression... loses it when downscaled") is hard coded in `web/src/App.jsx` to match v2's robustness numbers. Recheck after v3.
- Docs: `docs/pitch_outline.md` (new) has slide by slide content and judge Q&A. The video hook and the disclosure email no longer reference the private AKT blot; both have [placeholders] for the round 4 SynthID results. Slides not built yet (a Claude "Slides" artifact type is available on Daniel's account).
- Shell tip: `pkill -f` kills its own shell here; kill by PID instead.
<!-- /LIVE -->

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
