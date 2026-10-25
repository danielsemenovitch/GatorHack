# Pitch outline (Monday, top 10, about 5 minutes)

The deck is a Claude Slides artifact titled "Below the Watermark pitch" (11 slides, speaker notes on every slide; it downloads as PowerPoint or PDF). Order and the one line each slide must land:

1. **Cover:** a published blot with one Gemini band pasted in; SynthID Detector: not detected.
2. **Stakes:** 3.8% of 20,621 papers had image duplication before generative AI (Bik et al. 2016); Gemini makes a blot in seconds.
3. **Safety net:** SynthID (public October 7, 2026) and the C2PA credential.
4. **Finding:** whole AI blot detected 10 of 10; one band alone "not enough details" 4 of 4; one band pasted into a published blot not detected 4 of 4; credential stripped on the first re save.
5. **Statement:** watermarks protect whole images; science gets edited one band at a time.
6. **What we built:** five steps (list figures, read credentials, find reused images, screen blots, write the report) plus the open benchmark.
7. **Live demo:** Gemini blot credential; the PDF with a figure used twice; the pasted AI band the model catches; download the report. The slide's screenshot is the fallback if the live demo fails.
8. **Honest numbers:** 72 held out papers, AUROC 0.703, 35% of AI splices at 4% clean flags; 22 whole figures end to end, 7 edited panels flagged, 1 of 22 originals flagged; about 5% after a 50% resize.
9. **Lessons:** resolution (false flags 7% to 18%, fixed to 3%); chance flags; show the misses.
10. **Limits and ethics.**
11. **Next and close.**

## Likely judge questions

- **"Is anything mocked?"** No. Every number comes from `results/*.json`, produced by scripts in `scripts/`. The SynthID results are hand uploads recorded in `results/synthid_manual.json`.
- **"Why not just use SynthID?"** It works on whole images. It can't read a single band, and a pasted band is the realistic attack.
- **"Isn't your detector weak?"** Yes, and we say so: on 22 held out whole figures it flagged 7 of the edited panels. The contribution is the measured gap plus the open benchmark plus a working review tool; the detector is the first baseline.
- **"What does Strict do?"** It's a second flag line set on calibration papers to flag at most 1% of unedited panels. On 20 held out papers it caught 24% of pasted AI bands with 2.5% of unedited panels flagged, against 33% and 5.5% for Standard. On the 22 whole figures it flagged 4 edited panels and none of the originals. The old strict line caught 4%, so we replaced it.
- **"Why does the Oncotarget paper get 2 flags?"** At the research threshold about 1 in 20 unedited panels is flagged; with 30 panels, 1 or 2 are expected by chance, and both scores (0.512) sit just over the 0.50 line. The report says this.
- **"Did you just tune it until the demo worked?"** No. Both flag lines come from calibration papers, never the test set. The example we demo is labeled as one of the third it catches, and the Gap page shows the full rates. The figure on our landing page, the one SynthID missed, is a miss for our model too.
- **"How is this different from Proofig or ImageTwin?"** Those are closed commercial tools for duplication. We measured where the new public watermark stops, and we released a benchmark for AI era band edits that anyone can test against.
- **"What's recent?"** SynthID Detector (public Oct 7, 2026) and Gemini's current image tools are both central: one is the threat, one is the defense we measured.
