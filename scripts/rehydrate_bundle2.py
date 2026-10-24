"""Rehydrate the uploaded bundle-2 paper set from PUBLIC Europe PMC source figures.

The user's 69-MiB JSON bundle contains 440 figures from 241 CC-BY-labeled
papers. Large binary inputs are intentionally NOT committed. This script
uses its 241 exact PMCIDs, looks up western-blot figures in the published
article XML, and downloads up to two figures per paper, like the original
browser collector. Rehydration is a reconstruction, not a byte-exact import:
save the manifest and report the actual counts.

Only run on consenting/public CC BY sources; attribution is preserved.
No automated SynthID submissions are involved.
"""
from __future__ import annotations

import concurrent.futures as cf
import io
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data"
FIGS = BASE / "real_figures"
PAPERS = BASE / "bundle2_pmcids.txt"
WESTERN = re.compile(r"western blot|immunoblot", re.I)
HEADERS = {"User-Agent": "BelowTheWatermark-Hackathon/1.0 (CC BY figure reproducibility)"}
XLINK = "{http://www.w3.org/1999/xlink}href"


def read(url, timeout=22):
    error = None
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status}")
                return response.read()
        except Exception as exc:
            error = exc
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"{url}: {error}")


def get_paper(pmcid):
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
    root = ET.fromstring(read(url))
    title = "".join(root.find(".//article-title").itertext()).strip() if root.find(".//article-title") is not None else ""
    doi = next(("".join(e.itertext()).strip() for e in root.findall(".//article-id") if e.get("pub-id-type") == "doi"), "")
    year = next(("".join(e.itertext()).strip() for e in root.findall(".//pub-date/year")), "")
    # Paper IDs originated from the user's CC-BY-filtered bundle; metadata below
    # must still be checked against each article's actual license for redistribution.
    # Some JATS files express this in license text, others only in ext links.
    license_text = " ".join("".join(x.itertext()) for x in root.findall(".//license")).lower()[:1200]
    figures = []
    for fig in root.findall(".//fig"):
        caption = " ".join(fig.find("caption").itertext()).strip() if fig.find("caption") is not None else ""
        if not WESTERN.search(caption):
            continue
        graphic = fig.find(".//graphic")
        if graphic is None:
            continue
        href = graphic.get(XLINK) or graphic.get("href")
        if not href:
            continue
        fname = href if re.search(r"\.(?:jpg|jpeg|png)$", href, re.I) else (
            re.sub(r"\.(?:gif|tif|tiff)$", ".jpg", href, flags=re.I)
            if re.search(r"\.(?:gif|tif|tiff)$", href, re.I) else href + ".jpg")
        encoded = urllib.parse.urlencode({
            "pmcId": pmcid, "type": "FILE", "fileName": fname, "mimeType": "image/jpeg"})
        url = "https://europepmc.org/api/fulltextRepo?" + encoded
        try:
            payload = read(url)
            if len(payload) < 30000:
                continue
            with Image.open(io.BytesIO(payload)) as im:
                im.verify()
            with Image.open(io.BytesIO(payload)) as im:
                jpg = io.BytesIO()
                im.convert("RGB").save(jpg, format="JPEG", quality=92)
                payload = jpg.getvalue()
            label = re.sub(r"[^A-Za-z0-9]", "", fig.get("id") or "fig")
            if not label:
                continue
            figures.append({
                "file": f"{pmcid}_{label}.jpg",
                "pmcid": pmcid, "title": title, "doi": doi, "year": year,
                "journal": "", "license": "cc by", "figLabel": "".join(fig.find("label").itertext()).strip()
                    if fig.find("label") is not None else "",
                "caption": caption[:600],
                "_payload": payload,
            })
        except Exception as exc:
            print(f"WARN download {pmcid} {fname}: {exc}", flush=True)
        if len(figures) >= 2:
            break
    return pmcid, figures, license_text


def main():
    ids = PAPERS.read_text().split()
    assert len(ids) == 241 and len(set(ids)) == 241, f"Unexpected PMCID list size: {len(ids)}"
    FIGS.mkdir(parents=True, exist_ok=True)
    mf = FIGS / "metadata.json"
    metadata = {r["file"]: r for r in json.loads(mf.read_text())}
    old_count = len(metadata)
    success = []
    empty, failures, unusual_license = [], [], []
    started = time.monotonic()
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {pool.submit(get_paper, pmcid): pmcid for pmcid in ids}
        for num, job in enumerate(cf.as_completed(jobs), 1):
            pmcid = jobs[job]
            try:
                _, figures, license_text = job.result()
                if not figures:
                    empty.append(pmcid)
                if license_text and not ("creative commons attribution" in license_text or
                                         "creativecommons.org/licenses/by/" in license_text or
                                         "cc-by" in license_text or "cc by" in license_text):
                    unusual_license.append(pmcid)
                    # Preserve original user-specified CC-BY list but report this
                    # uncertainty instead of silently asserting verification.
                for f in figures:
                    data = f.pop("_payload")
                    (FIGS / f["file"]).write_bytes(data)
                    metadata[f["file"]] = f
                    success.append(f["file"])
            except Exception as exc:
                failures.append({"pmcid": pmcid, "error": str(exc)[:160]})
            if num % 25 == 0:
                print(f"PROGRESS papers {num}/{len(ids)}, new figures {len(success)}, "
                      f"errors {len(failures)}, elapsed {time.monotonic()-started:.1f}s", flush=True)
    mf.write_text(json.dumps(list(metadata.values()), indent=1))
    report = {
        "requested_papers": len(ids), "new_figures": len(success),
        "total_figures": len(metadata), "old_figures": old_count,
        "papers_with_no_images": empty, "failed_papers": failures,
        "license_text_not_recognized": unusual_license,
        "note": "Rehydration from original Europe PMC figure URLs; may not exactly match uploaded bundle image bytes.",
        "seconds": round(time.monotonic() - started, 2),
    }
    out = ROOT / "results" / "bundle2_rehydration.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print("RESULT", json.dumps({k: v for k, v in report.items() if not isinstance(v, list)}), flush=True)
    print(f"NOTICE skipped/no figures: {len(empty)}, fetch failures: {len(failures)}, ambiguous article licenses: {len(unusual_license)}")
    # Require a meaningful number of new independent source papers before training.
    if len(success) < 220:
        raise RuntimeError("Not enough downloaded figures for an expanded training benchmark (minimum 220)")
    return report


if __name__ == "__main__":
    main()
