"""Assemble the final SynthID upload set from the QC'd candidate pool.

Reads <pool>/key.csv and <pool>/qc_decisions.csv (columns: figure, donor, decision, reason),
takes the first N passing donor-host pairs in donor manifest order, copies their files
unchanged under fresh random codes, and writes the study's key.csv (unblinding key, keep
closed) and log.csv (to fill during uploads). See docs/paper/synthid_protocol.md.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="synthid_kit_pool")
    ap.add_argument("--out", default="synthid_kit")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    pool, out = ROOT / a.pool, ROOT / a.out
    rows = list(csv.DictReader(open(pool / "key.csv")))
    qc = {r["figure"]: r for r in csv.DictReader(open(pool / "qc_decisions.csv"))}
    order = {r["file"]: r["prompt_index"] for r in json.loads((ROOT / "data/ai_donors_v6/gemini/manifest.json").read_text()) if r.get("file")}
    # donor -> host figure, from the composite rows
    pairs = {}
    for r in rows:
        if r["donor"] and r["host"]:
            pairs[r["donor"]] = r["host"].split()[-1]
    passing = [d for d in sorted(pairs, key=lambda d: order.get(d, 1e9))
               if qc.get(pairs[d], {}).get("decision") == "pass"]
    chosen = passing[: a.n]
    figs = {pairs[d] for d in chosen}
    keep = [r for r in rows if (r["donor"] in chosen) or (not r["donor"] and r["host"].split()[-1] in figs)]
    if out.exists():
        shutil.rmtree(out)
    (out / "uploads").mkdir(parents=True)
    (out / "screens").mkdir()
    rng = random.Random(a.seed)
    used, key = set(), []
    for r in keep:
        while True:
            c = "".join(rng.choice("ABCDEFGHJKMNPQRSTUVWXYZ23456789") for _ in range(5))
            if c not in used:
                used.add(c)
                break
        ext = r["file"].rsplit(".", 1)[1]
        dst = out / "uploads" / f"{c}.{ext}"
        shutil.copyfile(pool / "uploads" / r["file"], dst)
        assert hashlib.sha256(dst.read_bytes()).hexdigest() == r["sha256"]
        key.append({**r, "code": c, "file": dst.name, "pool_code": r["code"]})
    rng.shuffle(key)
    with open(out / "key.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(key[0].keys()))
        w.writeheader()
        w.writerows(key)
    with open(out / "log.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["code", "synthid_verdict_verbatim", "date", "uploader", "notes"])
        for c in sorted(k["code"] for k in key):
            w.writerow([c, "", "", "", ""])
    shutil.copyfile(pool / "qc_decisions.csv", out / "qc_decisions.csv")
    print(f"{len(chosen)} donor-host pairs, {len(key)} files in {out / 'uploads'}")


if __name__ == "__main__":
    main()
