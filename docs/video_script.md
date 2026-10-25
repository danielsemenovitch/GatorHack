# Demo video script (under 3 minutes)

**Setup (5 minutes before recording):** start the app (README "Run it", then open http://localhost:8000), browser window about 1440 px wide, zoom 100%, close other tabs. Record the screen at 1080p with your voice (OBS, or Windows Game Bar with Win+Alt+R). Read the lines below at a calm pace; they are about 400 words, which fits in 2:45.

Never show the old AKT blot or any unpublished lab data. Every edited figure on screen is one we made from a CC BY figure, and the site says so.

---

**0:00 to 0:20. The hook.** Screen: the landing page, hero visible.

> Western blots are the evidence behind a huge share of biomedical papers. This year, Gemini made a convincing fake blot a twenty second job. Google's safety net is SynthID, an invisible watermark in every Gemini image, with a public detector that opened on October 7. We tested it on science.

**0:20 to 0:55. The finding.** Screen: click "The gap". Hold on the three part summary, then scroll slowly through the table.

> We uploaded every test by hand. A whole AI blot: detected ten times out of ten, even after JPEG, resizing, cropping and PDF export. One AI band on its own: not enough details. And the case that matters for science: one Gemini band pasted into a real published blot. SynthID did not detect it, four times out of four. The content credential was gone after the first re save. So the watermark protects whole images, and science gets edited one band at a time.

**0:55 to 1:15. What we built.** Screen: click "How it works", scroll through the five steps.

> So we built Below the Watermark, a review assistant for the figures a watermark can't vouch for. Drop in a figure or a whole paper PDF. It lists every figure, reads content credentials, finds images used twice, screens Western blots band by band with a model we trained, and writes a report a reviewer can check.

**1:15 to 1:35. Provenance.** Screen: click "Review images", scroll to the examples, click "A whole blot made by Gemini".

> When the credential is still there, we read it. This blot came straight from Gemini, and the report says so up front.

**1:35 to 2:05. The detector, honestly.** Screen: click "A pasted AI band the model catches". Point at the dashed box, then at the flagged panel and its heat.

> This is a published figure where we pasted one AI band into the IL 6 strip. The dashed box is our edit, and the model flagged the panel that contains it. It doesn't catch everything: on twenty two held out figures it flagged the edited panel in seven, and it flagged one of the twenty two unedited originals. So this is one of the good ones, and the site says that.

**2:05 to 2:40. Duplication, inside and across papers.** Screen: click "A manuscript with two hidden duplications". Point at the two boxes at the top, then scroll to "Search published papers online" and click "Search Europe PMC". (Start the search before you record this part if your connection is slow; it takes about a minute.)

> Watermarks can't see the oldest trick either: reusing an image. This demo manuscript hides two. The Vinculin control in Figure 1 is the HO 1 blot from Figure 2, resized, and the app lines them up. Figure 3 was lifted from a 2023 paper, and the app finds it in our library of published figures. For everything else, an agent takes over: Gemini reads the title and captions, plans searches on Europe PMC, and the agent checks the figures of every open access paper it finds. Every search it ran is listed, so a reviewer knows what was and wasn't checked.

**2:40 to 2:55. Close.** Screen: back to the landing page.

> Our benchmark of AI era blot edits is open, so others can beat our model. Watermarks protect whole images. Science needs protection one band at a time.

---

**If the online search finds nothing or can't connect:** say "the agent lists everything it searched" and show the list under "What the agent searched"; the library match above is the guaranteed part of the demo.
**If you run over:** cut the provenance section (1:15 to 1:35) first.
**Before recording:** put your Gemini key in a file named `.env` in the GatorHack folder (one line, `GEMINI_API_KEY=...`) and restart the backend, so the agent plans with Gemini. The file is ignored by git; never commit the key.
**Upload:** YouTube (unlisted is fine) or Vimeo, then paste the link into Devpost.
