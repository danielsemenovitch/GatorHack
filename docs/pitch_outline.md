# Pitch outline (Monday, top 10, about 5 minutes)

Slide by slide content plus what to say. Numbers in [brackets] must be copied from `results/*.json` at the end (the README results table is generated from the same files, so you can copy from there). Never round up.

The slide deck itself is being built as a Claude Slides artifact; if that isn't available, this outline is enough to build it in Google Slides in about an hour.

---

**1. Title**
- Below the Watermark
- "AI watermarks catch whole fake images. Science gets faked one band at a time."
- Team names, Gator Hack V, "Fresh off the frontier"

Say: "I work in a wet lab. Western blots are how we show a protein went up or down, and they sit behind a huge share of biomedical papers."

**2. The problem got easier this year**
- Picture: a whole western blot Gemini made for us from one sentence (from `data/ai_sources/`)
- Bik et al. 2016: 782 of 20,621 papers (3.8%) had inappropriate image duplication, before generative AI
- Gemini's image tools (Gemini 3 family, including Nano Banana Pro, Nov 2025) now make a convincing blot in seconds

Say: "Image fraud in papers was already common. This year it got a lot cheaper."

**3. The frontier safety net**
- SynthID: Google's invisible watermark in every Gemini image
- SynthID Detector opened to the public October 7, 2026, 2 weeks before this hackathon
- Plus a C2PA content credential ("Created by Google Generative AI") attached to downloads

Say: "Google's answer is SynthID. We wanted to know whether it protects science."

**4. We tested it by hand (the finding)**
- Matrix (from the app's Gap page):
  - whole AI blot, after re-save, 50% resize, JPEG 60, crop, contrast, as a figure panel, PDF round trip: detected every time
  - content credential: gone after the first re-save
  - a single AI band on its own: "not enough details" (4 of 4)
  - an AI band pasted into a published CC BY blot: [S1 to S4 verdicts from results/synthid_manual.json]
- Footnote: done by hand, one fixed scenario at a time, within the detector's terms

Say: "The watermark is genuinely robust on whole images. But that's not how blots get faked."

**5. Fraud happens one band at a time**
- Three tiny pictures: a band copied from another lane, a band darkened, a band pasted in
- The watermark works at image scale; the fraud is at band scale (a band can be 13 by 6 pixels in a web figure)

Say: "If someone pastes one AI band into a real blot, the watermark has almost nothing to work with."

**6. What we built**
- An open benchmark of band level forgeries with pixel masks
- A band splice detector trained on it
- A web app: figure or PDF in, panels and bands found, every band scored, copied bands checked, content credentials read, panel handed to SynthID

**7. The benchmark**
- [N images] from [P] CC BY papers via Europe PMC (see the DATA_CARD table)
- Five forgeries: copy move, cross splice, AI splice (Gemini bands), intensify, erase
- Every image pushed through a random publishing chain (contrast, resize, JPEG, PDF), clean ones too
- Split by paper so test papers are never seen in training
- Picture: one forged panel next to its mask

Say: "Existing datasets like BioFors predate generative AI. This is the dataset the problem was missing."

**8. Live demo** (the app, about 60 seconds)
- Upload or click the ALK2 example, show the heat and click a band for the explanation
- Show the untouched figure
- Say exactly what the model did on the example, even if it is not flagged (see HANDOFF section 6 for the current behavior)

**9. Honest results**
- On [T] papers it never saw: AUROC [x] (hand built baseline [b])
- At a [z]% false flag rate: catches [y]% of pasted AI bands
- Trained with no AI splices at all: still catches [h]% (it generalizes to edits it never saw)
- Side by side table: our detector keeps its signal through JPEG, contrast and PDF, loses it when the figure is downscaled

Say: "This is a first baseline on a hard problem, at a scale where the watermark can't run at all. The benchmark is open so others can beat it."

**10. Limits and ethics**
- A flag means look closer. It is never proof of misconduct.
- Forgeries are simulated; web resolution figures are the hardest case
- SynthID tests by hand, within its terms; no evasion loop against Google's tool
- Every figure CC BY with attribution; every edit made by us; no private lab data
- We reported the gap to Google [only say this once the email is sent]

**11. What's next**
- Full resolution figures from publishers (far more signal)
- A bigger and more varied AI donor pool, more generators
- A public leaderboard on the benchmark
- Signing images at capture on lab imagers

Close: "Watermarks protect whole images. Science needs protection one band at a time."

---

## Likely judge questions

- **"Is anything mocked?"** No. Every number comes from `results/*.json`, produced by scripts in `scripts/`. The SynthID results are hand uploads recorded in `results/synthid_manual.json`.
- **"Why not just use SynthID?"** It works on whole images. It can't read a single band, and a pasted band is the realistic attack.
- **"Isn't your detector weak?"** Yes, and we say so. The contribution is the measured gap plus the open benchmark, and the detector is the first baseline on it.
- **"How is this different from Proofig or ImageTwin?"** Those are closed commercial tools for duplication. We measured where the new public watermark stops, and we released a benchmark for AI era band edits that anyone can test against.
- **"What's recent?"** SynthID Detector (public Oct 7, 2026) and Gemini's current image tools are both central: one is the threat, one is the defense we measured.
