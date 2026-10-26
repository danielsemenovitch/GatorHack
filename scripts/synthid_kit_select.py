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
    ap.add_argument("--pool", nargs="+", default=["synthid_kit_pool"])
    ap.add_argument("--out", default="synthid_kit")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    out = ROOT / a.out
    order = {r["file"]: r["prompt_index"] for r in json.loads((ROOT / "data/ai_donors_v6/gemini/manifest.json").read_text()) if r.get("file")}
    cands = []  # (donor, figure, pool dir, rows of that pair)
    for name in a.pool:
        pool = ROOT / name
        rows = list(csv.DictReader(open(pool / "key.csv")))
        qc = {r["figure"]: r for r in csv.DictReader(open(pool / "qc_decisions.csv"))}
        pairs = {r["donor"]: r["host"].split()[-1] for r in rows if r["donor"] and r["host"]}
        for donor, fig in pairs.items():
            if qc.get(fig, {}).get("decision") != "pass":
                continue
            pr = [r for r in rows if r["donor"] == donor or (not r["donor"] and r["host"].split()[-1] == fig)]
            cands.append((donor, fig, pool, pr))
    chosen, seen_d, seen_f = [], set(), set()
    for donor, fig, pool, pr in sorted(cands, key=lambda c: order.get(c[0], 1e9)):
        if donor in seen_d or fig in seen_f:
            continue
        chosen.append((donor, fig, pool, pr)); seen_d.add(donor); seen_f.add(fig)
        if len(chosen) == a.n:
            break
    if out.exists():
        shutil.rmtree(out)
    (out / "uploads").mkdir(parents=True)
    (out / "screens").mkdir()
    rng = random.Random(a.seed)
    used, key = set(), []
    for donor, fig, pool, pr in chosen:
        for r in pr:
            while True:
                c = "".join(rng.choice("ABCDEFGHJKMNPQRSTUVWXYZ23456789") for _ in range(5))
                if c not in used:
                    used.add(c)
                    break
            ext = r["file"].rsplit(".", 1)[1]
            dst = out / "uploads" / f"{c}.{ext}"
            shutil.copyfile(pool / "uploads" / r["file"], dst)
            assert hashlib.sha256(dst.read_bytes()).hexdigest() == r["sha256"]
            key.append({**r, "code": c, "file": dst.name, "pool": pool.name, "pool_code": r["code"]})
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
    with open(out / "qc_decisions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pool", "figure", "decision", "reason", "reviewer", "date"])
        for name in a.pool:
            for r in csv.DictReader(open(ROOT / name / "qc_decisions.csv")):
                w.writerow([name, r["figure"], r["decision"], r["reason"], r.get("reviewer", ""), r.get("date", "")])
    print(f"{len(chosen)} donor-host pairs, {len(key)} files in {out / 'uploads'}")


if __name__ == "__main__":
    main()
