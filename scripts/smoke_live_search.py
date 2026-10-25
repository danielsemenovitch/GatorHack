"""Live, opt-in Europe PMC smoke test of the REAL public keyword-planned agent.

Run locally: python -m scripts.smoke_live_search
No Gemini key or private credentials required. This must not be run in routine
unit tests because internet availability and search results are external.
"""
from __future__ import annotations
import json
import os
import time
from pathlib import Path

from btw import search_agent
from btw.report import _figures_from_upload

ROOT=Path(__file__).resolve().parents[1]
PDF=ROOT/"examples/demo_reuse.pdf"
OUT=ROOT/"results/europepmc_live_smoke.json"

def main():
    # This particular smoke test should exercise the public no-key path,
    # independent of credentials and provider quota.
    os.environ.pop("GEMINI_API_KEY",None)
    data=PDF.read_bytes()
    figures, _, _=_figures_from_upload(data,"demo_reuse.pdf")
    images=[im for _,im in figures][:30]
    labels=[label for label,_ in figures][:30]
    start=time.monotonic()
    result=search_agent.run(data,"demo_reuse.pdf",images,labels)
    trimmed={
        "input":"examples/demo_reuse.pdf (two intentionally created duplications)",
        "type":"live_public_europe_pmc_no_gemini",
        "status":result.get("status"),
        "planner":result.get("planner"),
        "figures_compared":result.get("figures_compared"),
        "papers_checked":len(result.get("papers",[])),
        "queries":result.get("queries",[]),
        "papers":result.get("papers",[]),
        "matching_source_pmcids":sorted({m.get("source",{}).get("pmcid") for m in result.get("matches",[]) if m.get("source",{}).get("pmcid")}),
        "matched_regions":len(result.get("matches",[])),
        "duration_seconds":round(time.monotonic()-start,1),
        "note":"A live API call, not a mocked test. No result does not mean no reuse elsewhere; a match does not mean misconduct."
    }
    OUT.write_text(json.dumps(trimmed,indent=2)+"\n")
    print("LIVE_AGENT_SMOKE_RESULT",json.dumps(trimmed,indent=2),flush=True)
    if result.get("status") != "done" or not result.get("papers") or not result.get("figures_compared"):
        raise SystemExit("Live Europe PMC path did not retrieve and check published figures; examine JSON and logs.")
if __name__=="__main__":
    main()
