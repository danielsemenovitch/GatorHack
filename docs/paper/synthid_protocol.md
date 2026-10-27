# Protocol: do provenance signals survive partial AI edits of scientific figures?

Status: written before any upload in this study (October 26, 2026). The pilot (results/synthid_manual.json, 14 uploads on October 23) motivated it and is reported separately as a pilot.

## Question

Google's SynthID Detector (public since October 7, 2026) identifies images made by Gemini. Gemini also attaches C2PA Content Credentials. When only part of a published scientific figure is AI generated, how much of the figure has to be AI before SynthID reports it, and do C2PA credentials survive ordinary handling?

## Stimuli

- **AI donors:** Western blot images generated with the Gemini API by `scripts/generate_ai_blots.py`, saved exactly as returned, with prompt, model, time and SHA-256 logged in `data/ai_donors_v6/gemini/manifest.json`. Donors are taken in manifest order. Selection never uses any detector output.
- **Hosts:** published figures from openly licensed (CC BY) papers in `data/real_figures`, cited in `key.csv`. Hosts are tried in a fixed shuffled order (seed 7); a donor is paired with the first unused host on which the code can place every condition, and a donor with no such host is skipped.
- **Conditions per donor** (`scripts/synthid_study_kit.py`):

| Condition | What is AI | Approximate AI share of the uploaded image |
|---|---|---|
| whole_original | entire image, untouched | 100% |
| whole_jpeg85 | entire image, re-saved as JPEG 85 | 100% |
| panel_in_crop | one blot panel, figure cropped around it | about 25% |
| panel | one blot panel in the full figure | about 0.5 to 3% |
| row | one row of bands replaced | about 0.1 to 1% |
| bands3 | three adjacent bands replaced | about 0.1 to 0.5% |
| band1 | one band replaced | under 0.2% |
| host_unedited, host_crop_unedited | nothing (negative controls) | 0% |

Every composite is saved as PNG and as JPEG 85. AI pixels are converted to grayscale, resized and background matched as a person assembling a figure would do; the exact changed pixels are recorded.

## Quality control (before any upload)

*Amended October 26, 2026, before any upload:* candidate pairs come from two builds of the compositing code (`synthid_kit_pool2`, first 30 donors; `synthid_kit_pool3`, all 60 donors, after a fix for panels that straddle strips). An earlier 20-donor build was discarded unreviewed after its contact sheets showed systematic background halos; that code was then fixed. Each donor and host pair has one contact sheet in `qc/`. One reviewer rates every pair as pass or fail on visual plausibility alone (pasted text, a smeared or solid rectangle, bands misaligned with lanes or rows, a visible box edge), and records the rating and reason in `qc_decisions.csv`. The first six passing pairs in donor manifest order, with no donor or host figure used twice, form the study set (20 pairs reviewed, 6 passed). No file is dropped after uploading starts.

## Procedure

1. Files carry random 5-character codes. The person uploading does not open `key.csv` until every verdict is logged.
2. Each file is uploaded once, by hand, through the public SynthID Detector, in code order. There is no automated, scripted or repeated querying, and no adjustment of a file based on a verdict. This keeps the study within the service's terms and keeps the design fixed in advance.
3. The verdict is copied verbatim into `log.csv` with the date, and a screenshot is saved as `<code>.png` in `synthid_kit/screens/`.
4. C2PA is read locally with `c2pa-python` for every file; no upload is needed for that.

## Analysis

- **Primary outcome:** the proportion of files SynthID reports as made with Google AI, by condition. Reported as counts with exact (Clopper-Pearson) 95% intervals; there is no hypothesis test across conditions because the counts are small.
- **Secondary:** the share of AI pixels at which detection changes, shown as detection against AI share on a log axis per donor; the effect of JPEG 85; C2PA presence by condition.
- **Controls:** any detection on an unedited host is reported as a false positive.
- Everything is reported, including conditions that come out uninteresting.

## Limits stated in advance

One generator family (Gemini) is tested for SynthID. Six donors and roughly 80 uploads are a measured case study, not a population estimate. Composites are made by our code, not by people trying to evade detection; real fraud could be easier or harder to catch.
