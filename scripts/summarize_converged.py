#!/usr/bin/env python3
"""summarize_converged.py — the matched `local` vs `centralized` table for the sweep.

Reads every `artifacts/converged_all/*.json` written by `federated_eval --out-json` and
pairs the two arms PER CLIENT. The pairing is the point: both arms produce one score per
client (centralized reuses one pooled model for every client in its cluster), so the
honest unit is the per-client DIFFERENCE, not two independently-averaged columns.

Reported per dataset:
  * per-cluster mean paired delta on the primary metric
  * pooled delta over all clients, with a Wilcoxon signed-rank test when scipy is
    present (n is small and the deltas are not remotely normal)
  * the delta placed against the honest MDE band for this benchmark

The MDE band comes from the 2026-07-15 audit: ~0.11-0.15 VUS-PR at n=31 real clients.
A delta inside the band is NOT a result, however clean the sign looks.

    python scripts/summarize_converged.py
"""
from __future__ import annotations

import argparse
import json
import statistics as stats
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

PRIMARY = "vus_pr"
METRICS = ["vus_pr", "auprc", "pate_f1", "auroc", "affiliation_f1", "f1"]
MDE_LO, MDE_HI = 0.11, 0.15          # audit 2026-07-15, n=31; NOT the ledger's 0.087


def load(resdir: Path) -> dict[str, list[dict]]:
    """{dataset: [record, ...]} with cluster/arm attached to each record."""
    out: dict[str, list[dict]] = defaultdict(list)
    for f in sorted(resdir.glob("*.json")):
        try:
            blob = json.loads(f.read_text())
        except Exception as e:
            print(f"  !! {f.name}: unreadable ({e})")
            continue
        meta = blob.get("meta", {})
        ds, cl = meta.get("dataset", f.stem), meta.get("cluster")
        for rec in blob.get("records", []):
            rec = dict(rec)
            rec["_cluster"], rec["_file"] = cl, f.name
            out[ds].append(rec)
    return out


def paired(records: list[dict], metric: str) -> list[tuple[str, str, float, float]]:
    """[(cluster, entity, local, centralized)] for clients scored under BOTH arms."""
    by: dict[tuple, dict[str, float]] = defaultdict(dict)
    for r in records:
        arm, ent, seed = r.get("_arm"), r.get("_entity"), r.get("_seed")
        v = r.get(metric)
        if arm and ent and v is not None:
            by[(r.get("_cluster"), ent, seed)][arm] = float(v)
    rows = []
    for (cl, ent, _seed), arms in sorted(by.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        if "local" in arms and "centralized" in arms:
            rows.append((cl, ent, arms["local"], arms["centralized"]))
    return rows


def wilcoxon(deltas: list[float]) -> str:
    nz = [d for d in deltas if d != 0]
    if len(nz) < 6:
        return "n<6, no test"
    try:
        from scipy.stats import wilcoxon as w
        stat, p = w(nz)
        return f"Wilcoxon p={p:.4f} (n={len(nz)})"
    except Exception:
        pos = sum(1 for d in nz if d > 0)
        return f"sign test {pos}/{len(nz)} positive (scipy absent)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resdir", default=str(REPO / "artifacts" / "converged_all"))
    args = ap.parse_args()

    resdir = Path(args.resdir)
    data = load(resdir)
    if not data:
        print(f"no result JSON under {resdir}")
        return 1

    for ds, records in sorted(data.items()):
        arms = sorted({r.get("_arm") for r in records if r.get("_arm")})
        files = sorted({r["_file"] for r in records})
        print("\n" + "=" * 78)
        print(f"{ds}   ({len(records)} records, arms: {', '.join(arms)}, "
              f"{len(files)} job file(s))")
        print("=" * 78)

        rows = paired(records, PRIMARY)
        if not rows:
            print(f"  no client scored under BOTH arms — cannot pair on {PRIMARY}.")
            # still show whatever single-arm means exist
            for arm in arms:
                vals = [float(r[PRIMARY]) for r in records
                        if r.get("_arm") == arm and r.get(PRIMARY) is not None]
                if vals:
                    print(f"    {arm:12s} {PRIMARY}: mean={stats.fmean(vals):.3f} (n={len(vals)})")
            continue

        print(f"\n  per-client {PRIMARY}  (delta = centralized - local)")
        print(f"  {'cluster':12s} {'client':12s} {'local':>8s} {'central':>8s} {'delta':>8s}")
        by_cluster: dict[str, list[float]] = defaultdict(list)
        deltas = []
        for cl, ent, lo, ce in rows:
            d = ce - lo
            deltas.append(d)
            by_cluster[str(cl)].append(d)
            print(f"  {str(cl):12s} {ent:12s} {lo:8.3f} {ce:8.3f} {d:+8.3f}")

        print(f"\n  per-cluster mean delta:")
        for cl, ds_ in sorted(by_cluster.items()):
            print(f"    {cl:12s} {stats.fmean(ds_):+.3f}  (n={len(ds_)})")

        m = stats.fmean(deltas)
        sd = stats.stdev(deltas) if len(deltas) > 1 else 0.0
        wins = sum(1 for d in deltas if d > 0)
        print(f"\n  POOLED  delta={m:+.4f}  sd={sd:.3f}  n={len(deltas)}  "
              f"centralized wins {wins}/{len(deltas)}")
        print(f"  {wilcoxon(deltas)}")
        if abs(m) < MDE_LO:
            print(f"  => |delta| < MDE ({MDE_LO}-{MDE_HI}): NOT a result, whatever the sign.")
        elif abs(m) < MDE_HI:
            print(f"  => |delta| inside the MDE band ({MDE_LO}-{MDE_HI}): borderline, "
                  f"needs more seeds before it is claimable.")
        else:
            print(f"  => |delta| above the MDE band ({MDE_LO}-{MDE_HI}): a real effect.")

        print(f"\n  secondary metrics (mean delta, centralized - local):")
        for k in METRICS:
            if k == PRIMARY:
                continue
            rr = paired(records, k)
            if rr:
                dd = [c - l for _, _, l, c in rr]
                print(f"    {k:18s} {stats.fmean(dd):+.4f}  (n={len(dd)})")

    print("\n" + "=" * 78)
    print("NOTE: federated arms are NOT in this table. They still train on a fixed round")
    print("budget, so comparing them against these converged baselines would read a")
    print("budget artifact as a federation effect.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
