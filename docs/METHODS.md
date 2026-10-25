# How Below the Watermark works (read this before pitching)

This is written so you can explain every part in your own words and answer judges' questions. Each section ends with the questions you're likely to get.

## 1. The finding: where SynthID stops working

**What we did.** Using Gemini's image tools (free account, model "Auto," Gemini 3 family) we made western blots: whole fakes, and single bands pasted into published blots. We uploaded each one by hand to SynthID Detector (public since October 7, 2026). We tested one fixed scenario at a time, with no automation and no iterating to beat the detector, because the terms forbid both.

**What we saw.**
- Whole AI images were caught every time: re-saved, resized to 50%, JPEG at quality 60, cropped to 70%, contrast changed, placed as one panel of a figure, PDF round trip.
- The visible C2PA content credential ("Created by Google Generative AI") was stripped by any re-save at all.
- Single AI bands on their own came back "not enough details."
- AI bands pasted into published CC BY blots: see `results/synthid_manual.json` (being re-run on open data).

**Why that matters.** Fraud in blots happens at band scale: one lane made darker or emptier changes the conclusion. A watermark needs enough watermarked pixels to be read, and a single band doesn't have them.

**Likely questions**
- *Isn't this just a handful of uploads?* Yes. It's a case study, and we say so. Each composite takes about two minutes to make, and the pattern was consistent.
- *Did you try to break SynthID?* No. Fixed, realistic publishing steps only, uploaded by hand. We reported what we saw to Google.
- *Is it fair to call it a miss?* For lone bands it honestly declines ("not enough details"). For the four pasted band composites it answered "SynthID was not detected", so yes, those are misses; the untouched control was also not detected, as it should be.

## 2. The benchmark

See `docs/DATA_CARD.md` for the exact counts (written automatically from the data). In one sentence: thousands of blot images from CC BY papers, clean or forged five ways (copy move, cross splice, AI splice, intensify, erase), each pushed through a random publishing chain, each with a pixel mask, split by paper.

**Likely questions**
- *Why generated forgeries and not real fraud cases?* Real cases are few, legally sensitive, and don't come with pixel masks. Generated forgeries let us measure detection exactly. The tradeoff is realism, which is why we hold out a whole forgery type in testing.
- *Why web resolution?* That's what reviewers and readers see, and what Europe PMC and PubMed Central serve. It's also the hard case.

## 3. The band splice detector

**Idea.** An edited region was made under different conditions from the rest of the blot (another camera, another blot, a generator, a resampling step), so its pixel statistics disagree with the rest of the same panel. We never ask "is this band dark?" We ask "does this band behave like its neighbors?"

**Two models**
- **Hand built features + gradient boosting.** Each panel is cut into 24 × 24 pixel windows. For each window we measure:
  - noise level (residual after a median filter)
  - noise shape (kurtosis)
  - error level analysis (how much the window changes when recompressed)
  - JPEG 8 × 8 block alignment
  - edge sharpness
  - background brightness and grain

  Each measure is compared with the panel's other bands, or with its background. A gradient boosted tree model scores each window.
- **Small convolutional network (JAX, CPU).** Same windows, but the network sees the pixels directly: intensity relative to the panel background, plus a noise residual normalized by the panel's typical noise. It's about 40,000 parameters, trained with real band windows as hard negatives so it can't cheat by flagging "anything that is a band."

**Panel score.** The mean of the 3 most suspicious band windows (for the CNN, on the raw logit scale, optionally relative to the panel's own median band window; the rule is picked on calibration papers). The flag line is set on clean panels from separate CALIBRATION papers (held out from training) so that 5% of them would be flagged. Test papers never touch the threshold or the rule. In the app every score is shown on a 0 to 100 scale where the flag line is exactly 50.

**Explanations.** When a band is flagged, the app lists the hand built measures that deviate most at that spot, in plain language ("its noise level differs from the rest of the blot").

**Likely questions**
- *What's the accuracy?* See `results/detector_metrics.json` and `results/cnn_metrics.json`. Quote the AUROC and the AI splice detection rate at 5% false flags. Do not round up.
- *Why not a big pretrained forensic model?* We had a CPU and 48 hours. The benchmark is public so anyone can try one.
- *Does it generalize to AI bands it never saw?* That's the held out method row: trained without any AI splices, tested on AI splices.

## 4. Copied bands

**Idea.** Two real bands can share a shape but never share noise. For each band we take the high pass residual, remove line structure (band edges and frame lines, which look alike everywhere), and search the blot for the same grain pattern, including flipped copies.

**Honest status.** It finds exact copies in unresized images. At web resolution after resizing or JPEG it rarely does (see `results/duplicates_metrics.json`). Present it as a supporting check, not a headline.

## 5. Content credentials

We read C2PA manifests with the official `c2pa-python` library. A raw Gemini download shows "Created by Google Generative AI" and "Applied imperceptible SynthID watermark." Any re-save removes it, so for published figures this check almost always says "none." It's still worth running, because when it is present it is signed evidence.

## 6. What we do not claim

- A flag is a reason to look closer, never proof of misconduct.
- Our forgeries are simulated; a careful forger can get past us.
- The SynthID results are a small hand run case study.
