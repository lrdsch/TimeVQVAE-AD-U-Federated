#!/usr/bin/env python3
"""Assemble the client x model x metric matrix for wsd_fed across every tree.

Writes artifacts/floor/wsd_fed_all_models.csv (tidy) and prints aggregates.
Threshold-free metrics (vus_pr, auprc, auroc, pate_f1) are comparable across
FLOOR and the deep arms; f1 / affiliation_f1 are NOT (different threshold rule).
"""
import json, glob, collections, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
METRICS = ["vus_pr", "auprc", "auroc", "pate_f1", "affiliation_f1", "f1"]
FREE = {"vus_pr", "auprc", "auroc", "pate_f1"}       # threshold-free
TREES = {"converged_all": "artifacts/converged_all/wsd_*.json",
         "converge60":    "artifacts/converge60/**/*.json",
         "fed_eval":      "artifacts/fed_eval/wsd_fed/**/*.json"}
TRUST = {"converged_all": "trusted", "converge60": "trusted", "fed_eval": "PRE-PURGE"}

rows = []
# FLOOR
for line in (REPO / "artifacts/floor/records_wsd_fed.jsonl").read_text().splitlines():
    if line.strip():
        r = json.loads(line)
        rows.append({"tree": "floor", "trust": "trusted", "arm": r["_arm"],
                     "cluster": r["_cluster"], "entity": r["_entity"], "seed": r.get("_seed", 0),
                     **{m: r.get(m) for m in METRICS}})
# deep
for tree, pat in TREES.items():
    for f in glob.glob(str(REPO / pat), recursive=True):
        try: d = json.load(open(f))
        except Exception: continue
        if not isinstance(d, dict): continue
        for r in d.get("records", []) or []:
            if not str(r.get("_entity", "")).startswith("kpi"): continue
            rows.append({"tree": tree, "trust": TRUST[tree], "arm": r.get("_arm"),
                         "cluster": r.get("_cluster"), "entity": r["_entity"],
                         "seed": r.get("_seed", 0),
                         **{m: r.get(m) for m in METRICS}})

# de-duplicate (tree, arm, entity, seed) keeping the first
seen, ded = set(), []
for r in rows:
    k = (r["tree"], r["arm"], r["entity"], r["seed"])
    if k in seen: continue
    seen.add(k); ded.append(r)
rows = ded

# collapse seeds -> mean per (tree, arm, entity)
by = collections.defaultdict(list)
for r in rows:
    by[(r["tree"], r["trust"], r["arm"], r["cluster"], r["entity"])].append(r)
flat = []
for (tree, trust, arm, cl, ent), rs in by.items():
    o = {"tree": tree, "trust": trust, "arm": arm, "cluster": cl, "entity": ent,
         "n_seeds": len(rs)}
    for m in METRICS:
        v = [x[m] for x in rs if isinstance(x.get(m), (int, float)) and np.isfinite(x[m])]
        o[m] = float(np.mean(v)) if v else None
    flat.append(o)

out = REPO / "artifacts/floor/wsd_fed_all_models.csv"
cols = ["tree", "trust", "arm", "cluster", "entity", "n_seeds"] + METRICS
with out.open("w") as fh:
    fh.write(",".join(cols) + "\n")
    for r in sorted(flat, key=lambda r: (r["tree"], r["arm"], r["cluster"], r["entity"])):
        fh.write(",".join("" if r[c] is None else
                          (f"{r[c]:.6f}" if isinstance(r[c], float) else str(r[c]))
                          for c in cols) + "\n")

models = sorted({(r["tree"], r["trust"], r["arm"]) for r in flat})
print(f"{len(flat)} rows · {len(models)} models · -> {out.relative_to(REPO)}\n")
hdr = f'{"tree":14s} {"arm":26s} {"n":>3s} ' + " ".join(f"{m[:9]:>9s}" for m in METRICS)
print(hdr); print("-" * len(hdr))
for tree, trust, arm in models:
    sub = [r for r in flat if (r["tree"], r["arm"]) == (tree, arm)]
    cells = []
    for m in METRICS:
        v = [r[m] for r in sub if r[m] is not None]
        cells.append(f"{np.median(v):9.4f}" if v else f'{"-":>9s}')
    mark = "" if trust == "trusted" else "  <PRE-PURGE>"
    print(f"{tree:14s} {arm:26s} {len(sub):3d} " + " ".join(cells) + mark)
print("\n(mediane su 31 client; f1 / affiliation_f1 NON comparabili tra floor e deep:"
      " regola di soglia diversa)")
