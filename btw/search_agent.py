"""Interpaper duplication agent: look for the uploaded figures in other published papers.

The agent
  1. reads the upload: PDF title, authors, abstract and figure captions, or for
     a single image, the figure itself (Gemini reads its labels);
  2. plans Europe PMC searches: the same authors' other papers (self reuse) and
     papers on the same proteins, cells and treatments (paper mill templates).
     Gemini plans when GEMINI_API_KEY is set; otherwise keyword rules do;
  3. pulls the figures of the open access papers it finds;
  4. runs the region matcher (btw.region_match) between every uploaded figure
     and every downloaded figure, and reports matches with links.

Everything it searched is returned in a log, so a reviewer can see what was and
was not checked. A match means the same pixels appear in both papers; it does
not say which came first or whether the reuse was permitted.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from . import region_match as rm

EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
FILES = "https://europepmc.org/api/fulltextRepo"
UA = {"User-Agent": "BelowTheWatermark/1.0 (Gator Hack V research demo)"}
XLINK = "{http://www.w3.org/1999/xlink}href"

MAX_QUERIES = 5
PER_QUERY = 10
MAX_PAPERS = 16
MAX_FIGS_PER_PAPER = 6
TIME_BUDGET = 75.0  # seconds


# ------------------------------------------------------------------ http ----

def _get(url: str, timeout: float = 20.0) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _plain(text: str | None) -> str:
    """Europe PMC titles carry HTML (italics, entities such as &lt;i&gt;); show plain text."""
    import html
    t = html.unescape(html.unescape(text or ""))
    return " ".join(re.sub(r"<[^>]+>", "", t).split())


def search(query: str, n: int = PER_QUERY) -> list[dict]:
    """Open access, full text papers matching a Europe PMC query."""
    q = f"({query}) AND OPEN_ACCESS:y AND HAS_FT:y"
    url = f"{EPMC}/search?" + urllib.parse.urlencode({"query": q, "format": "json", "pageSize": n, "resultType": "core"})
    data = json.loads(_get(url))
    out = []
    for x in data.get("resultList", {}).get("result", []):
        if x.get("pmcid"):
            out.append({"pmcid": x["pmcid"], "title": _plain(x.get("title")), "year": x.get("pubYear"),
                        "doi": x.get("doi"), "authors": x.get("authorString", ""), "journal": _plain(x.get("journalTitle"))})
    return out


def figure_list(pmcid: str) -> list[dict]:
    """Figures of one article from its JATS full text: label, caption, file name."""
    root = ET.fromstring(_get(f"{EPMC}/{pmcid}/fullTextXML"))
    figs = []
    for f in root.iter("fig"):
        g = f.find(".//graphic")
        href = g.get(XLINK) if g is not None else None
        if not href:
            continue
        fn = re.sub(r"\.(tif|tiff|gif)$", ".jpg", href, flags=re.I) if re.search(r"\.(jpe?g|gif|tiff?|png)$", href, re.I) else href + ".jpg"
        label = "".join(f.find("label").itertext()).strip() if f.find("label") is not None else ""
        cap = " ".join("".join(f.find("caption").itertext()).split())[:300] if f.find("caption") is not None else ""
        figs.append({"label": label, "caption": cap, "file": fn})
    return figs


def figure_image(pmcid: str, file: str) -> Image.Image:
    url = FILES + "?" + urllib.parse.urlencode({"pmcId": pmcid, "type": "FILE", "fileName": file, "mimeType": "image/jpeg"})
    return Image.open(io.BytesIO(_get(url))).convert("RGB")


# --------------------------------------------------------------- context ----

def upload_context(data: bytes, filename: str) -> dict:
    """Title, authors, DOI and text the planner can use. Empty for plain images."""
    ctx = {"title": "", "authors": "", "doi": "", "text": ""}
    if not (filename.lower().endswith(".pdf") or data[:5] == b"%PDF-"):
        return ctx
    try:
        import fitz
        doc = fitz.open(stream=data, filetype="pdf")
        meta = doc.metadata or {}
        ctx["title"] = (meta.get("title") or "").strip()
        ctx["authors"] = (meta.get("author") or "").strip()
        text = "\n".join(doc[i].get_text() for i in range(min(3, len(doc))))
        caps = []
        for page in doc:
            for line in page.get_text().splitlines():
                if re.match(r"\s*(Fig(ure)?\.?\s*\d+)", line, re.I):
                    caps.append(line.strip())
        ctx["text"] = (text[:4000] + "\n" + "\n".join(caps[:30]))[:6000]
        m = re.search(r"\b(10\.\d{4,9}/[^\s\"<>]+)", text)
        ctx["doi"] = m.group(1).rstrip(".,;)") if m else ""
        if not ctx["title"]:
            first = [l.strip() for l in text.splitlines() if len(l.strip()) > 25]
            ctx["title"] = first[0] if first else ""
    except Exception:
        pass
    return ctx


# --------------------------------------------------------------- planning ---

STOP = set("""a an the of in on and or for with to from by via into at as is are was were be been this that these those
using use based study analysis effect effects role new novel its their our we human mouse cells cell expression
induce induces induced activates activation response responses demo manuscript paper after during""".split())


def keyword_queries(ctx: dict) -> list[dict]:
    qs = []
    for name in [a.strip() for a in re.split(r"[;,]| and ", ctx.get("authors", "")) if a.strip()][:2]:
        last = name.split()[-1]
        if len(last) > 1:
            qs.append({"query": f'AUTH:"{last}"', "why": f"other papers by {name}"})
    words = [w for w in re.findall(r"[A-Za-z0-9\-]{4,}|\b[A-Z0-9]{3}\b", ctx.get("title", "")) if w.lower() not in STOP]
    if words:
        qs.append({"query": "TITLE_ABS:(" + " AND ".join(f'"{w}"' for w in words[:2]) + ')', "why": "papers on the same topic"})
        qs.append({"query": "TITLE_ABS:(" + " AND ".join(f'"{w}"' for w in words[:3]) + ') AND METHODS:"western blot"', "why": "same topic, with western blots"})
        if len(words) > 3:
            qs.append({"query": "TITLE_ABS:(" + " AND ".join(f'"{w}"' for w in words[2:5]) + ')', "why": "same proteins or treatments"})
    return qs[:MAX_QUERIES]


PLAN_PROMPT = """You help screen a scientific manuscript for image duplication across papers
(the same blot or micrograph appearing in an earlier paper, by the same authors or from a paper mill).
Propose up to {n} Europe PMC search queries that would find papers likely to contain the same figures:
earlier papers by the same authors, and papers studying the same proteins, cell lines and treatments.
Use Europe PMC syntax (AUTH:"Surname", TITLE_ABS:(...), METHODS:"western blot", quotes, AND/OR).
Do not add OPEN_ACCESS or date filters. Reply with JSON only:
{{"queries": [{{"query": "...", "why": "short reason"}}]}}

Manuscript title: {title}
Authors: {authors}
Text excerpt:
{text}
"""

# Stable model first; preview models are often rate limited (429) or overloaded
# (503) on the free tier. GEMINI_TEXT_MODEL in .env overrides the order.
MODELS = [m for m in dict.fromkeys([os.environ.get("GEMINI_TEXT_MODEL"), "gemini-3-flash-preview",
                                    "gemini-flash-latest", "gemini-2.5-flash"]) if m]
LAST_MODEL: str | None = None


def _why(e: Exception) -> str:
    code = getattr(e, "code", None)
    return f"HTTP {code}" if code else type(e).__name__


def _gemini(parts: list[dict], key: str) -> str:
    """Try each model; retry once on rate limit or overload. Raises with every model's reason."""
    global LAST_MODEL
    reasons = []
    base = {"temperature": 0.2, "responseMimeType": "application/json"}
    # Planning queries needs little reasoning; ask thinking models to keep it short.
    quick = json.dumps({"contents": [{"parts": parts}], "generationConfig": {**base, "thinkingConfig": {"thinkingLevel": "low"}}}).encode()
    plain = json.dumps({"contents": [{"parts": parts}], "generationConfig": base}).encode()
    for model in MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        body = quick
        for attempt in range(3):
            req = urllib.request.Request(url, data=body, headers={**UA, "Content-Type": "application/json", "x-goog-api-key": key})
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    out = json.loads(r.read())
                text = "".join(p.get("text", "") for p in out["candidates"][0]["content"]["parts"])
                LAST_MODEL = model
                return text
            except Exception as e:  # unknown model, quota, overload, network, blocked reply
                code = getattr(e, "code", None)
                if code == 400 and body is quick:   # model does not take thinkingLevel
                    body = plain
                    continue
                if code in (429, 500, 503) and attempt < 2:
                    time.sleep(2.0)
                    continue
                reasons.append(f"{model}: {_why(e)}")
                break
    raise RuntimeError("; ".join(reasons))


def gemini_queries(ctx: dict, images: list[Image.Image], key: str) -> list[dict]:
    parts = [{"text": PLAN_PROMPT.format(n=MAX_QUERIES, title=ctx.get("title") or "(none; read the figure)",
                                          authors=ctx.get("authors") or "(unknown)", text=ctx.get("text") or "(none)")}]
    if not ctx.get("text"):
        # No manuscript text: let Gemini read the protein labels off the figure itself.
        im = images[0].copy(); im.thumbnail((1200, 1200))
        buf = io.BytesIO(); im.convert("RGB").save(buf, "JPEG", quality=85)
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(buf.getvalue()).decode()}})
    raw = _gemini(parts, key)
    raw = raw[raw.find("{"): raw.rfind("}") + 1]
    qs = [q for q in json.loads(raw).get("queries", []) if isinstance(q, dict) and str(q.get("query", "")).strip()]
    return [{"query": str(q["query"])[:300], "why": str(q.get("why", ""))[:120]} for q in qs][:MAX_QUERIES]


def plan(ctx: dict, images: list[Image.Image]) -> tuple[list[dict], str, str | None]:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    note = None
    if key:
        try:
            qs = gemini_queries(ctx, images, key)
            if qs:
                return qs, "Gemini", None
            note = "Gemini returned no queries; used keyword search instead."
        except Exception as e:
            note = f"Gemini could not plan the search ({e}); used keyword search instead."
    return keyword_queries(ctx), "keywords", note


# -------------------------------------------------------------------- run ---

def _doi_key(d: str | None) -> str:
    return (d or "").lower().strip()


def run(data: bytes, filename: str, images: list[Image.Image], labels: list[str]) -> dict:
    t0 = time.time()
    ctx = upload_context(data, filename)
    queries, planner, note = plan(ctx, images)
    ts = time.time()  # the search budget starts after planning (Gemini can take a while)
    log, papers, seen = [], [], set()
    if not queries:
        return {"status": "no_queries", "planner": planner, "note": note or
                "There was nothing to search on: no title, authors or text in the upload, and no Gemini key to read the figure.",
                "queries": [], "papers": [], "figures_compared": 0, "matches": [], "seconds": round(time.time() - t0, 1)}
    for q in queries:
        try:
            hits = search(q["query"])
            q["results"] = len(hits)
            for h in hits:
                if h["pmcid"] in seen or (ctx["doi"] and _doi_key(h["doi"]) == _doi_key(ctx["doi"])):
                    continue
                seen.add(h["pmcid"]); papers.append({**h, "why": q["why"]})
        except Exception as e:
            q["error"] = type(e).__name__
        if len(papers) >= MAX_PAPERS or time.time() - ts > TIME_BUDGET / 3:
            break
    papers = papers[:MAX_PAPERS]
    if not papers and all("error" in q for q in queries):
        return {"status": "offline", "planner": planner, "note": "Could not reach Europe PMC from this computer.",
                "queries": queries, "papers": [], "figures_compared": 0, "matches": [], "seconds": round(time.time() - t0, 1)}

    upload_feats = [rm.features(im, with_flip=True) for im in images]

    def one_paper(p):
        found, n = [], 0
        try:
            figs = figure_list(p["pmcid"])[:MAX_FIGS_PER_PAPER]
        except Exception as e:
            return p, 0, [], type(e).__name__
        for f in figs:
            if time.time() - ts > TIME_BUDGET:
                break
            try:
                im = figure_image(p["pmcid"], f["file"])
            except Exception:
                continue
            n += 1
            rf = rm.features(im)
            for i, qf in enumerate(upload_feats):
                m = rm.match(qf, rf)
                if m:
                    found.append({"figure": i, "figure_label": labels[i], "box": rm._pad(m["query_box"], qf.size),
                                  "source": {"pmcid": p["pmcid"], "title": p["title"], "year": p["year"], "doi": p["doi"],
                                             "authors": p["authors"], "figLabel": f["label"], "caption": f["caption"],
                                             "url": f"https://europepmc.org/article/PMC/{p['pmcid'].removeprefix('PMC')}"},
                                  "source_box": rm._pad(m["ref_box"], rf.size), "source_image": im,
                                  "inliers": m["inliers"], "scale": m["scale"], "flipped": m["flipped"]})
        return p, n, found, None

    matches, compared = [], 0
    with ThreadPoolExecutor(max_workers=6) as ex:
        for p, n, found, err in ex.map(one_paper, papers):
            p["figures"] = n
            if err:
                p["error"] = err
            compared += n
            matches.extend(found)
    return {"status": "done", "planner": planner, "note": note, "queries": queries,
            "papers": [{k: p.get(k) for k in ("pmcid", "title", "year", "doi", "why", "figures", "error")} for p in papers],
            "figures_compared": compared, "matches": matches, "seconds": round(time.time() - t0, 1),
            "context": {"title": ctx["title"][:200], "doi": ctx["doi"]}}
