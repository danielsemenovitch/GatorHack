# Protocol: do provenance signals survive partial AI edits of scientific figures?

Status: version 2, fixed October 26, 2026, before any upload in this study. The pilot (results/synthid_manual.json, 14 uploads on October 23) motivated it and is reported separately as a pilot. History of changes is at the end.

## Questions

Google's SynthID Detector (public since October 7, 2026) identifies images made by Gemini. Gemini and ChatGPT also attach C2PA Content Credentials.

1. **Area:** how much of a published figure has to be Gemini pixels, untouched, before SynthID reports it?
2. **Processing:** do the ordinary steps used to put AI content into a blot figure (grayscale, resizing, re-rendering onto a membrane background) erase the watermark on their own, even at 100% AI?
3. **Realistic edits:** does SynthID report figures in which real bands were replaced by AI bands, as a person editing a figure would do it?
4. **C2PA:** do credentials survive the steps a figure goes through on its way into a paper?

## Stimuli

- **AI donors:** Western blots generated with the Gemini API by `scripts/generate_ai_blots.py` (6 Gemini image models, 60 images), saved exactly as returned, with prompt, model, time and SHA-256 in `data/ai_donors_v6/gemini/manifest.json`. ChatGPT blots (20) for the C2PA question only. Donor selection never uses any detector output.
- **Hosts:** published Western blot figures from openly licensed (CC BY) papers in `data/real_figures`, each cited in the key.

### Arm A: area, untouched pixels (first 10 Gemini donors in manifest order)

An untouched crop of the donor (no resizing, no colour change), centred on its bands, is pasted over the largest blot panel of a host figure so that it covers 50, 20, 10, 5, 2, 1 and 0.5% of the uploaded image; each host is used for one donor only and is not used in arm C. Plus, per donor: the original file as generated, the original re-saved as JPEG 85, and the unedited host. Lossless PNG except the JPEG control.

### Arm B: processing at 100% AI (same 10 donors)

The whole donor (i) converted to grayscale only, and (ii) put through the same steps as the arm C band edits: grayscale, downscaling to 420 px width, darkness map, re-rendered on a flat membrane background.

### Arm C: realistic edits (6 donor and host pairs)

Real bands are replaced by AI bands aligned to the host's lanes and painted onto the host's own membrane: one panel in a figure crop (25 to 42% AI), one panel in the full figure (1.5 to 8%), one row, three bands, one band (0.03 to 1.8%), plus each unedited host and host crop. PNG only.

Quality control for arm C (before any upload): candidate pairs come from two builds of the compositing code (`synthid_kit_pool2`, first 30 donors; `synthid_kit_pool3`, all 60 donors, after a fix for panels that straddle strips). An earlier 20-donor build (`synthid_kit_pool`) was reviewed and discarded whole because most composites showed background halos or misplaced rows; the code was then fixed. One reviewer rated every pair as pass or fail on visual plausibility alone (pasted text, a smeared or solid rectangle, bands misaligned with lanes or rows, a visible box edge), with reasons in `results/synthid_qc_decisions.csv`. The first six passing pairs in donor manifest order, no donor or host used twice, form arm C (20 pairs reviewed, 6 passed). A second rater, blind to the first, will rate the same contact sheets; agreement is reported, and arm C results are reported separately for pairs both raters passed.

Arm A and B files need no quality control: they test detection, not plausibility.

## Procedure

1. All 164 files carry random 5-character codes and are uploaded in alphabetical order. The person uploading does not open `sealed/key.csv` until every verdict is logged.
2. Each file is uploaded once, by hand, through the public SynthID Detector. There is no automated, scripted or repeated querying, and no change to a file because of a verdict. This keeps the study within the service's terms and keeps the design fixed in advance. Uploads may be split across sessions; the dates are logged.
3. The verdict is copied verbatim into `log.csv`, and a screenshot is saved as `screens/<code>.png`.
4. C2PA is read locally with `c2pa-python`; no upload is needed (`scripts/c2pa_pipeline.py` for question 4).

## Analysis

- **Primary outcome:** proportion of files SynthID reports as made with Google AI, by condition, with exact (Clopper-Pearson) 95% intervals.
- **Area (arm A):** detection against AI share of the image on a log axis, one line per donor; the smallest share detected for each donor; overall proportion detected per share.
- **Processing (arm B):** detection of grayscale and fully processed whole images, compared with the untouched originals of the same donors (paired, 10 donors).
- **Realistic edits (arm C):** detection by edit type; interpreted against arms A and B (an arm C miss at a share that arm A detects points to processing, not area).
- **Controls:** any detection on an unedited host is reported as a false positive.
- Everything is reported, including conditions that come out uninteresting.

## Limits stated in advance

One watermark (SynthID) from one company; 10 donors for area and processing, 6 pairs for realistic edits: a measured case study, not a population estimate. Composites are made by our code, not by people trying to evade detection. SynthID can change at any time; results describe the detector as it was on the upload dates.

## History

- Version 1 (October 26, 2026): arm C only, 6 pairs, PNG and JPEG copies (84 files). Never uploaded.
- Version 2 (October 26, 2026, before any upload): arms A and B added to separate area from processing; JPEG copies of arm C dropped; second QC rater added; C2PA pipeline question added.
