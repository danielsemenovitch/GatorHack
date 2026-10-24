# Draft: note to the SynthID team

**To:** the SynthID feedback channel (the feedback icon at the top right of synthid.com), and Google's Vulnerability Reward Program intake if you want a tracked report: https://bughunters.google.com
**Subject:** SynthID Detector and band level edits in scientific images (student hackathon finding)

Hi SynthID team,

I'm an undergraduate at the University of Florida. This weekend at Gator Hack V (UF AI Days) my team tested the public SynthID Detector on western blot images, the kind used as evidence in biomedical papers. Everything was uploaded by hand, one fixed scenario at a time, following the detector's terms.

What we saw:

1. Whole images from Gemini's image tools were detected every time, including after re-saving, resizing to 50%, JPEG at quality 60, cropping, contrast changes, placement as one panel of a figure, and a PDF round trip. Impressive robustness.
2. Tight crops of a single band (roughly 80 to 200 pixels wide) returned "not enough details."
3. A band edited by Gemini and then pasted into an otherwise real blot returned "SynthID was not detected," both as PNG and as JPEG.

Point 3 matters for research integrity, because image manipulation in papers usually happens at the scale of a single band. We are not publishing a step by step recipe. Our public materials describe the gap only at the level above, and we are releasing a band level forensics tool and benchmark to help cover it.

Happy to share the exact files privately if useful.

Thanks,
Daniel Semenovitch
University of Florida
