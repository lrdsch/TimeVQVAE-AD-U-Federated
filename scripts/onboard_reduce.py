"""Reduce onboard_clean records: is the cold-start number inflated by codebook
leakage? Compares LEAKY (codebook saw the held-out) vs CLEAN (retrained on
donors only) at the same budget, plus the local ceiling and prevalence floor."""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REC = ROOT / "artifacts/fed_eval/onboard_clean"


def load(ds, rounds=None):
    f = REC / f"records_{ds}.jsonl"
    rows = [json.loads(l) for l in open(f) if l.strip()]
    if rounds is not None:
        rows = [r for r in rows if r.get("rounds") == rounds]
    return rows


def agg(v):
    v = [x for x in v if isinstance(x, (int, float)) and np.isfinite(x)]
    return (np.mean(v), np.std(v, ddof=1)/len(v)**0.5*1.96 if len(v) > 1 else 0.0, len(v)) if v else (float('nan'), 0, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--rounds", type=int, default=None)
    args = ap.parse_args()
    rows = load(args.dataset, args.rounds)
    if not rows:
        print("no records yet"); return
    lk = [r["vus_leaky"] for r in rows]
    cl = [r["vus_clean"] for r in rows]
    lo = [r["local_vus"] for r in rows if isinstance(r.get("local_vus"), (int, float))]
    fl = [r["floor_prevalence"] for r in rows if isinstance(r.get("floor_prevalence"), (int, float))]
    dleak = [r["vus_leaky"] - r["vus_clean"] for r in rows]

    print(f"\n{'='*66}\n  COLD-START LEAKAGE CHECK — {args.dataset} "
          f"(rounds={args.rounds or 'all'}, n={len(rows)} held-outs)\n{'='*66}")
    print(f"  floor (nothing/prevalence) : {agg(fl)[0]:.3f}")
    print(f"  CLEAN onboard (no leakage) : {agg(cl)[0]:.3f} ±{agg(cl)[1]:.3f}")
    print(f"  LEAKY onboard (contaminated): {agg(lk)[0]:.3f} ±{agg(lk)[1]:.3f}")
    print(f"  local-trained (ceiling)    : {agg(lo)[0]:.3f} ±{agg(lo)[1]:.3f}")
    dm, dci, dn = agg(dleak)
    inflate = "LEAKAGE INFLATES" if (dm > dci and dm > 0) else ("clean≥leaky (no inflation)" if dm <= 0 else "within noise")
    print(f"\n  leakage effect (leaky − clean): {dm:+.3f} ±{dci:.3f}  → {inflate}")
    print(f"  leaky>clean in {np.mean([d>0 for d in dleak])*100:.0f}% of held-outs")
    cm = agg(cl)[0]; lom = agg(lo)[0]; flm = agg(fl)[0]
    if np.isfinite(lom) and np.isfinite(flm) and lom > flm:
        print(f"  CLEAN recovers {(cm-flm)/(lom-flm)*100:.0f}% of (local − floor)  [clean/local = {cm/lom*100:.0f}%]")

    print(f"\n  per cluster (leaky | clean | Δleak | local):")
    byc = defaultdict(list)
    for r in rows:
        byc[r["cluster"]].append(r)
    for c in sorted(byc):
        rs = byc[c]
        L = np.mean([x["vus_leaky"] for x in rs]); C = np.mean([x["vus_clean"] for x in rs])
        lv = [x["local_vus"] for x in rs if isinstance(x.get("local_vus"), (int, float))]
        Lo = np.mean(lv) if lv else float('nan')
        print(f"    {c:14s} {L:6.3f} | {C:6.3f} | {L-C:+.3f} | {Lo:6.3f}  (n={len(rs)})")
    print()


if __name__ == "__main__":
    main()
