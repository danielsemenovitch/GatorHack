"""Build a NEW, attributed CC-BY Western-blot dataset from Europe PMC OA APIs.

Rehydrates the 241 source PMCID list supplied for v3 AND discovers genuinely
additional 2023–26 source papers. Each article's license is checked again.
Never silently claim publication authors edited an image: our edits are synthetic.
Output: data/real_figures/{PMCID}_{figure}.jpg and metadata.json, manifest
results/v4_sources.json. Do NOT commit harvested image binaries.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import random
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"
HEADERS = {"User-Agent": "GatorHackBelowTheWatermark/2026 CC-BY-dataset-research"}
XLINK = "{http://www.w3.org/1999/xlink}href"
CAPTION = re.compile(r"\b(?:western\s*blot(?:ting)?|immunoblot(?:ting)?|western[- ]blots?)\b", re.I)


def get(url, timeout=15):
    error = None
    for attempt in range(2):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=timeout) as r:
                if r.status != 200:
                    raise RuntimeError(f"HTTP {r.status}")
                return r.read(3_000_000)
        except Exception as exc:
            error = exc
            time.sleep(0.15 * (attempt + 1))
    raise RuntimeError(f"{type(error).__name__}: {error}")


def parse_license(root):
    license_nodes = root.findall(".//license")
    for node in license_nodes:
        urls = [e.attrib.get(XLINK, "") + " " + e.attrib.get("href", "")
                for e in node.iter()]
        links = " ".join(urls).lower()
        words = " ".join(node.itertext()).lower()
        if re.search(r"creativecommons\.org/licenses/by/(?:[234]\.0|[234]\.0/)", links):
            return "CC BY", "jats_license_link"
        if ("creative commons attribution" in words or re.search(r"\bcc[\s-]by\b", words)) and not any(
                bad in words for bad in ("noncommercial", "non-commercial", "no derivatives", "sharealike",
                                          "share-alike", "cc-by-nc", "cc-by-nd", "cc-by-sa")):
            return "CC BY", "jats_license_text"
    return None, "unverified"


def discover(existing, limit, years=(2023, 2024, 2025, 2026)):
    result = []
    seen = set(existing)
    for year in years:
        query = f'(METHODS:"western blot" OR METHODS:"immunoblot") AND OPEN_ACCESS:y AND LICENSE:"cc by" AND PUB_YEAR:{year} AND SRC:MED AND HAS_FT:y'
        for page in (1, 2, 3):
            uri = BASE + "/search?" + urllib.parse.urlencode({
                "query": query, "format": "json", "pageSize": "200",
                "resultType": "core", "page": str(page)})
            try:
                j = json.loads(get(uri, timeout=22))
            except Exception as exc:
                print(f"WARN discovery {year}/{page}: {exc}", flush=True)
                continue
            hits = j.get("resultList", {}).get("result", [])
            for row in hits:
                pid = row.get("pmcid")
                if pid and pid not in seen and str(row.get("license", "")).strip().lower() == "cc by":
                    seen.add(pid)
                    result.append(pid)
            print(f"SEARCH {year} page {page}: {len(hits)} hits, {len(result)} candidate new papers", flush=True)
            if len(hits) < 200 or len(result) >= max(3 * limit, 600):
                break
        if len(result) >= max(3 * limit, 600):
            break
    return result


def one_paper(pid):
    root = ET.fromstring(get(f"{BASE}/{pid}/fullTextXML"))
    lic, lic_method = parse_license(root)
    if lic != "CC BY":
        return pid, [], f"license_{lic_method}"
    title = " ".join(root.find(".//article-title").itertext()) if root.find(".//article-title") is not None else ""
    doi = next(("".join(e.itertext()).strip() for e in root.findall(".//article-id")
                if e.get("pub-id-type") == "doi"), "")
    year = next(("".join(e.itertext()).strip() for e in root.findall(".//pub-date/year")), "")
    journal = " ".join(root.find(".//journal-title").itertext()) if root.find(".//journal-title") is not None else ""
    items = []
    for fig in root.findall(".//fig"):
        caption = " ".join(fig.find("caption").itertext()).strip() if fig.find("caption") is not None else ""
        if not CAPTION.search(caption):
            continue
        graphic = fig.find(".//graphic")
        href = (graphic.get(XLINK) or graphic.get("href", "")) if graphic is not None else ""
        if not href:
            continue
        fname = href if re.search(r"\.(?:jpe?g|png)$", href, re.I) else (
            re.sub(r"\.(?:tif|tiff|gif)$", ".jpg", href, flags=re.I)
            if re.search(r"\.(?:tif|tiff|gif)$", href, re.I) else href + ".jpg")
        url = "https://europepmc.org/api/fulltextRepo?" + urllib.parse.urlencode({
            "pmcId": pid, "type": "FILE", "fileName": fname, "mimeType": "image/jpeg"})
        try:
            raw = get(url)
            if len(raw) < 20_000:
                continue
            im = Image.open(io.BytesIO(raw)).convert("RGB")
            if min(im.size) < 90 or im.width * im.height > 15_000_000:
                continue
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=92)
            raw = buf.getvalue()
        except Exception:
            continue
        fid = re.sub(r"[^A-Za-z0-9]", "", fig.get("id") or fname)[:64]
        item = {
            "file": f"{pid}_{fid}.jpg",
            "pmcid": pid, "journal": journal, "year": year,
            "title": title, "doi": doi, "license": "cc by",
            "license_verification": lic_method,
            "figLabel": " ".join(fig.find("label").itertext()) if fig.find("label") is not None else "",
            "caption": caption[:800],
            "source_api": url,
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        items.append((item, raw))
        if len(items) >= 2:
            break
    return pid, items, ("ok" if items else "no_qualifying_figures")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-papers", type=int, default=240)
    ap.add_argument("--workers", type=int, default=7)
    args = ap.parse_args()
    start = time.monotonic()
    figures = ROOT / "data" / "real_figures"
    mf = figures / "metadata.json"
    figures.mkdir(parents=True, exist_ok=True)
    metadata = {x["file"]: x for x in json.loads(mf.read_text())}
    original = len(metadata)
    prior = (ROOT / "data/v4_prior_pmcids.txt").read_text().split()
    assert len(prior) == 241 and len(set(prior)) == 241
    existing_papers = {r["pmcid"] for r in metadata.values()}
    # The entire old published corpus is already on disk; fetch only prior
    # v3 sources missing from this checkout, and new never-before-used papers.
    requested_prior = [pid for pid in prior if pid not in existing_papers]
    candidates = discover(existing_papers | set(prior), args.new_papers)
    random.Random(2026).shuffle(candidates)
    collected = {"prior": set(), "new": set()}
    rejects = {}
    successes = 0
    jobs = [(p, "prior") for p in requested_prior] + [(p, "new") for p in candidates]
    # Don't fetch thousands of unnecessary new articles; candidate failures
    # typically come from no figure captions or missing figure image URLs.
    jobs = jobs[:len(requested_prior) + min(len(candidates), max(3 * args.new_papers, 600))]
    print(f"Starting {len(jobs)} CC-BY paper inspections: {len(requested_prior)} prior, "
          f"{len(jobs)-len(requested_prior)} genuinely additional candidates", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(one_paper, pid): (pid, category) for pid, category in jobs}
        for i, task in enumerate(concurrent.futures.as_completed(futures), 1):
            pid, category = futures[task]
            try:
                _, images, status = task.result()
            except Exception as exc:
                rejects[pid] = str(exc)[:100]
                continue
            if images:
                collected[category].add(pid)
                for record, data in images:
                    dst = figures / record["file"]
                    dst.write_bytes(data)
                    metadata[record["file"]] = record
                    successes += 1
            else:
                rejects[pid] = status
            if i % 70 == 0:
                print(f"COLLECT {i}/{len(jobs)}: prior={len(collected['prior'])}, "
                      f"new={len(collected['new'])}, new images={successes}, "
                      f"elapsed={round(time.monotonic()-start)}s", flush=True)
    mf.write_text(json.dumps(list(metadata.values()), indent=1))
    out = ROOT / "results" / "v4_sources.json"
    out.parent.mkdir(exist_ok=True)
    report = {
        "original_figures": original,
        "added_figures": successes,
        "total_figures": len(metadata),
        "prior_papers_rehydrated": len(collected["prior"]),
        "genuinely_new_papers": len(collected["new"]),
        "total_papers": len({x["pmcid"] for x in metadata.values()}),
        "verification_policy": "exact CC-BY license from article JATS, with DOI and source URL",
        "rejected_papers": len(rejects),
        "rejection_examples": dict(list(rejects.items())[:30]),
        "elapsed_seconds": round(time.monotonic()-start),
        "new_article_ids": sorted(collected["new"]),
    }
    out.write_text(json.dumps(report, indent=2))
    print("V4_SOURCES", json.dumps({k:v for k,v in report.items() if not isinstance(v,(list,dict))}), flush=True)
    if len(collected["new"]) < 70 or len(collected["prior"]) < 130:
        raise RuntimeError("Insufficient verified independent source diversity; do not run v4 train")
    return report


if __name__ == "__main__":
    main()
