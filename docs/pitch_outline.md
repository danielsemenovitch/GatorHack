# Pitch outline (Monday, top 10, about 5 minutes)

The deck is a Claude Slides artifact titled "Below the Watermark pitch" (11 slides, speaker notes on every slide; it downloads as PowerPoint or PDF). Order and the one line each slide must land:

1. **Cover:** Below the Watermark — detecting duplicated scientific figures within and across papers.
2. **Stakes:** 3.8% of 20,621 papers had image duplication before generative AI (Bik et al. 2016); Gemini makes a blot in seconds.
3. **Safety net:** SynthID (public October 7, 2026) and the C2PA credential.
4. **Finding:** whole AI blot detected 10 of 10; one band alone "not enough details" 4 of 4; one band pasted into a published blot not detected 4 of 4; credential stripped on the first re save.
5. **Statement:** provenance alone does not reveal copied regions in scientific figures; reviewers need image-level evidence.
6. **What we built:** PDF inventory, in-paper region matching, published-reference library, optional Europe PMC search, downloadable HTML evidence, and experimental CNN.
7. **Live demo:** Gemini blot credential; the manuscript with two hidden duplications (a blot reused inside the paper, a figure lifted from a 2023 paper), then "Search Europe PMC" for the agent; the pasted AI band the model catches; download the report. The slide's screenshot is the fallback if the live demo fails.
8. **Honest numbers:** 110/118 (93.2%) **larger block reuse**, 60/120 (50%) **smaller panels/strips**, 0/7,140 matched untouched reference pairs. CNN proof of concept only: synthetic AUROC 0.703, 35.4% splice recall; 7/22 whole edited figures flagged.
9. **Lessons:** resolution (false flags 7% to 18%, fixed to 3%); chance flags; show the misses.
10. **Limits and ethics.**
11. **Next and close.**

## Likely judge questions

- **"Is anything mocked?"** The demo duplications were deliberately created by us using attributed CC BY material; the quantitative benchmarks are controlled synthetic experiments, and the agent CI tests mock external APIs. SynthID tests were manually uploaded and recorded.
- **"Why not just use SynthID?"** It works on whole images. It can't read a single band, and a pasted band is the realistic attack.
- **"Isn't your detector weak?"** Yes, and we say so: on 22 held out whole figures it flagged 7 of the edited panels. The contribution is the measured gap plus the open benchmark plus a working review tool; the detector is the first baseline.
- **"What does Strict do?"** It's a second flag line set on calibration papers to flag at most 1% of unedited panels. On 20 held out papers it caught 24% of pasted AI bands with 2.5% of unedited panels flagged, against 33% and 5.5% for Standard. On the 22 whole figures it flagged 4 edited panels and none of the originals. The old strict line caught 4%, so we replaced it.
- **"Why does the Oncotarget paper get 2 flags?"** At the research threshold about 1 in 20 unedited panels is flagged; with 30 panels, 1 or 2 are expected by chance, and both scores (0.512) sit just over the 0.50 line. The report says this.
- **"Did you just tune it until the demo worked?"** No. Both flag lines come from calibration papers, never the test set. The example we demo is labeled as one of the third it catches, and the Gap page shows the full rates. The figure on our landing page, the one SynthID missed, is a miss for our model too.
- **"How does it find duplication across papers?"** Two ways. A library of 112 published figures from 62 papers is checked on every upload. On request, an agent goes further: Gemini reads the manuscript (or the labels on a figure) and plans Europe PMC searches for the same authors and topic, the agent downloads the figures of the open access papers it finds, and the same matcher checks each one. It lists every search it ran.
- **"How good is the matcher?"** It found 110/118 (93.2%) **larger artificial reused regions (about 25–55% of their source figure)**, but only 60/120 (50%) **small panels or strips**. It matched zero of 7,140 untouched pairs in the evaluated reference set. It is not tested as a universal real-world fraud detector.
- **"How is this different from Proofig or ImageTwin?"** Those are closed commercial tools for duplication. We measured where the new public watermark stops, and we released a benchmark for AI era band edits that anyone can test against.
- **"What's recent?"** SynthID Detector (public Oct 7, 2026) and Gemini's current image tools are both central: one is the threat, one is the defense we measured.
