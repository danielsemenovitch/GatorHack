"""The interpaper agent, with Europe PMC and Gemini replaced by local fixtures.

The real services cannot be reached from CI; these tests pin the parsing of
Europe PMC's search JSON and JATS full text, the planning fallbacks, and that a
figure lifted from a published paper is reported with its source."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from btw import search_agent as sa
from btw.report import _figures_from_upload

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples/demo_reuse.pdf"
SOURCE = ROOT / "data/real_figures/PMC10701888_f1.jpg"

SEARCH_JSON = {"hitCount": 2, "resultList": {"result": [
    {"id": "38000000", "pmcid": "PMC10701888", "title": "Cleaved caspase study", "pubYear": "2023",
     "doi": "10.3389/fonc.2023.1287444", "authorString": "Doe J, Roe R.", "journalTitle": "Front Oncol"},
    {"id": "1", "title": "No full text"}]}}
JATS = b"""<?xml version="1.0"?><article xmlns:xlink="http://www.w3.org/1999/xlink"><body>
<fig id="f1"><label>Figure 1</label><caption><p>Western blots of patient samples.</p></caption>
<graphic xlink:href="fonc-13-1287444-g001"/></fig></body></article>"""


def fake_get(url, timeout=20.0):
    if "/search?" in url:
        return json.dumps(SEARCH_JSON).encode()
    if url.endswith("/fullTextXML"):
        return JATS
    if "fulltextRepo" in url:
        assert "fileName=fonc-13-1287444-g001.jpg" in url and "pmcId=PMC10701888" in url
        return SOURCE.read_bytes()
    raise AssertionError(url)


def demo():
    data = DEMO.read_bytes()
    figs, _, _ = _figures_from_upload(data, "demo.pdf")
    return data, [im for _, im in figs], [lb for lb, _ in figs]


def test_parses_search_results_and_figures(monkeypatch):
    monkeypatch.setattr(sa, "_get", fake_get)
    hits = sa.search('AUTH:"Doe"')
    assert [h["pmcid"] for h in hits] == ["PMC10701888"]          # entries without a PMCID are skipped
    figs = sa.figure_list("PMC10701888")
    assert figs == [{"label": "Figure 1", "caption": "Western blots of patient samples.", "file": "fonc-13-1287444-g001.jpg"}]


def test_finds_a_figure_lifted_from_a_published_paper(monkeypatch):
    monkeypatch.setattr(sa, "_get", fake_get)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    data, images, labels = demo()
    out = sa.run(data, "demo.pdf", images, labels)
    assert out["status"] == "done" and out["planner"] == "keywords"
    assert out["queries"], "keyword planner must produce queries from the PDF title"
    hits = [m for m in out["matches"] if m["source"]["pmcid"] == "PMC10701888"]
    assert hits and hits[0]["figure_label"] == "page 4, image 1"
    assert hits[0]["source"]["url"] == "https://europepmc.org/article/PMC/10701888"


def test_gemini_plan_is_used_and_falls_back(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(sa, "_gemini", lambda parts, key: 'Sure: {"queries": [{"query": "AUTH:\\"Doe\\"", "why": "same author"}]}')
    qs, planner, note = sa.plan({"title": "A study", "authors": "Doe J", "text": "x"}, [])
    assert planner == "Gemini" and qs[0]["query"] == 'AUTH:"Doe"' and note is None

    def boom(parts, key):
        raise RuntimeError("quota")
    monkeypatch.setattr(sa, "_gemini", boom)
    qs, planner, note = sa.plan({"title": "Caspase cleavage in leukemia cells", "authors": "Doe J", "text": "x"}, [])
    assert planner == "keywords" and qs and "keyword" in note


def test_reports_offline_cleanly(monkeypatch):
    def offline(url, timeout=20.0):
        raise OSError("no network")
    monkeypatch.setattr(sa, "_get", offline)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    data, images, labels = demo()
    out = sa.run(data, "demo.pdf", images, labels)
    assert out["status"] == "offline" and out["matches"] == []


def test_titles_are_plain_text():
    assert sa._plain("Effects of &lt;i&gt;Eubacterium limosum&lt;/i&gt; in cancer.") == "Effects of Eubacterium limosum in cancer."
    assert sa._plain("Role of <i>GRN</i> &amp; miR-29b") == "Role of GRN & miR-29b"
