# 3-minute submission video — duplication-first (October 25, 2026)

**Current pitch choice, updated after PR #8:** Lead with **scientific image reuse within and across papers**. Keep the Western blot CNN as an explicitly experimental proof of concept until a new model independently demonstrates a meaningful improvement. The completed eight-epoch V4.1 experiment did **not** meet our ~40% target (23.8% synthetic AI-splice recall on its newer test). Keep the CNN explicitly a work-in-progress and the main demo focused on verified duplication matches.

**Before recording:** pull latest `main`, restart both servers, and run the demo search once (this builds the reference library and wakes Gemini, so it answers quickly on camera). Gemini planning works with `GEMINI_API_KEY` in the git-ignored `.env` (it uses `gemini-flash-latest`); never show the key. Spoken narration is about 2:45; if you run over, cut "Spotting that can take hours of comparing figures by eye."

Use the full browser window, around 1440 pixels wide, and record at 1080p. **All duplications shown in the demo manuscript were intentionally created by our team from CC BY images. Published authors did not make our edits.**

---

## 0:00–0:23 — The problem
**Screen:** Hero on “Review images,” then a close-up of the paper figure.

> In science, figures are evidence. But sometimes the same image is reused under a different label—or appears in another paper. Spotting that can take hours of comparing figures by eye. Now AI image editing makes the problem even harder.

## 0:23–0:44 — Why provenance isn't enough
**Screen:** Click “The gap”; show the whole-image versus pasted-band results.

> Google just opened its SynthID Detector to the public, on October 7. We tested it by hand. It identified all ten whole AI blots in our small test, but found none of four AI-generated bands we pasted into real, published blots. Watermarks are valuable, but they don't tell the whole story. We wanted to check the image evidence itself.

## 0:44–1:03 — What we built
**Screen:** Click “How it works,” then return to “Review images.”

> So we built Below the Watermark: a scientific-image review assistant. Upload an image or a paper PDF. It inventories the figures, checks for duplicated images and regions, compares them with published references, and shows the evidence side by side for a human reviewer.

## 1:03–1:46 — The working duplication demo
**Screen:** Click “A manuscript with two hidden duplications.” Let it analyze; show both match panels at the top of the report.

> Here's our test manuscript. We assembled it from openly licensed figures and deliberately hid two duplications. First, a blot labeled Vinculin in Figure One actually comes from the HO-1 image in Figure Two. We resized the strip; the matcher still finds the shared pixels. Second, Figure Three contains part of an image from a 2023 paper. Our reference library finds the published source and shows the matching regions alongside each other. Every example here is our own controlled edit—not an allegation about the researchers.

## 1:46–2:14 — Search beyond the local library
**Screen:** Scroll to “Search published papers online” and click “Search Europe PMC.” Keep the result on screen (“1 match found… searches planned by Gemini”), then open “What the agent searched.”

> The built-in library is only a starting point, so we built a search agent. Gemini reads the manuscript's title and captions and plans searches on Europe PMC. The agent downloads the figures of every open-access paper it finds and runs the same matcher. Here it found the 2023 source after checking just a few papers, and it lists every search it ran.

*(Say "Gemini" and "a few papers" only if the run on screen shows them; if it fell back to keywords, say "it plans searches".)*

## 2:14–2:38 — Honest evidence and limitations
**Screen:** Return to the matched regions; briefly show methodology or a simple metric card.

> In a controlled benchmark, our matcher found 110 of 118 larger reused blot or microscopy blocks after resizing, cropping and other edits. Tiny single-panel copies were harder: 60 of 120 were found. It matched none of 7,140 untouched reference-image pairs. These are controlled tests, not proof of fraud. A reviewer makes the final call.

## 2:38–2:51 — Experimental frontier
**Screen:** Click “A pasted AI band the model catches”; point to the dashed known-edit box and heat.

> We're also training a neural network for altered Western blot bands. It's a research proof of concept: it catches about a third of pasted bands in our tests. With more data and HiPerGator compute, that's our next step.

## 2:51–3:00 — Close
**Screen:** Return to the hero and project title.

> Below the Watermark helps reviewers find the evidence worth checking. Not automated accusations—transparent, reproducible starting points for scientific integrity.

---

### Recording safeguards

- If the online search is slow, offline, or returns no matches, **do not manufacture a result**: leave the library evidence onscreen and say the agent can search further. Show the query log only if a real run produced one.
- Say **110/118 (93.2%) only for larger artificial copies, about 25–55% of the source figure**, and **60/120 (50%) for small-strip/panel tests**, not for all duplication. The 0/7,140 figure is only the evaluated reference library, not universal specificity.
- Do not promote V4/V4.1 or claim “40%” unless completed comparable hold-out metrics substantiate it. Even if a new CNN exceeds 40% on synthetic tests, end-to-end real-paper validation remains future work.
- The built-in source demo's first page and final page explicitly disclose team-created reuse. Keep the disclosure visible if showing the manuscript.
- Show region matches in the website and the HTML report; for online agent evidence, download **after** a real completed search.
- Have a pre-recorded local fallback of the **actual** successful demo for spotty Wi-Fi; do not use fabricated agent responses.
- Aim for roughly 2:50 of spoken narration with a few seconds for clicks, and upload the video to an unlisted YouTube or Vimeo URL for Devpost.
