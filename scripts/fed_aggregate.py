#!/usr/bin/env python3
"""Correct two-level aggregation + honest paired significance for federated_eval.

`pipeline/federated_eval.py`'s in-process `_aggregate` flattens every
(client, seed) report into ONE list, so its reported ``std`` confounds client
and seed variance and its ``worst`` is a single min-of-min observation. Fine for
a quick console glance; WRONG for a camera-ready error bar.

For each metric this script computes, the only defensible way:
  * per SEED — the macro-mean over the clients AND the worst-client value
    (min for higher-is-better metrics, max for the cf_* lower-is-better ones);
  * across SEEDS — mean +/- SAMPLE std (ddof=1) of those per-seed macro-means,
    and of the per-seed worst-client values.

Significance is reported at the SEED level: the SEED is the independent
replication unit. The 6 clients within a seed are evaluations of ONE global
model on sibling silos and are NOT independent, so pooling client x seed into a
single Wilcoxon (n=18/24) is pseudo-replication that inflates significance. We
therefore pair the per-seed macro-means (n = #seeds) and ALSO report descriptive
sign-consistency (on how many seeds / how many client-units the reference arm
wins), which is the honest headline at the handful of seeds a synthetic study
runs (the two-sided exact Wilcoxon floor at n=4 is p=0.125).

It consumes the ``records`` block written by federated_eval.py (--out-json).

    python scripts/fed_aggregate.py artifacts/fed_eval/toy_fed_s0123.json \
        artifacts/fed_eval/toy_fed_t256_s0123.json --primary vus_pr

    # also dump the per-(seed, client) primary-metric matrix for the figure:
    python scripts/fed_aggregate.py <json...> --csv artifacts/fed_eval/pairs.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import force_utf8_stdout   # noqa: E402

force_utf8_stdout()     # _fmt() prints '±' — see utils.force_utf8_stdout

# Metrics where LOWER is better → the "worst client" is the MAX, not the min.
LOWER_BETTER = {"cf_repair_ratio", "cf_disturb_mae"}
DEFAULT_METRICS = ["vus_pr", "auprc", "auroc", "affiliation_f1", "f1",
                   "cf_repair_improvement", "cf_disturb_mae"]


def _worst(vals: list[float], metric: str) -> float:
    return max(vals) if metric in LOWER_BETTER else min(vals)


def _load(path: str) -> tuple[str, list[dict], dict]:
    d = json.loads(Path(path).read_text())
    recs = d.get("records")
    if not recs:
        raise SystemExit(
            f"{path}: no 'records' block. Re-run pipeline/federated_eval.py "
            f"(it now persists per-(client,seed) records in --out-json).")
    meta = d.get("meta", {})
    dataset = meta.get("dataset", Path(path).stem)
    cluster = meta.get("cluster") or ""
    # Per-cluster sweeps emit one json PER CLUSTER of the SAME dataset, so labelling by
    # dataset alone would print N identical headers and collapse the clusters together
    # in the CSV. Qualify with the cluster whenever federated_eval recorded one.
    label = f"{dataset}/{cluster}" if cluster else dataset
    return label, dataset, cluster, recs, meta


def _two_level(recs: list[dict], metric: str) -> dict[str, dict]:
    """{arm: per-seed-then-across-seed stats} for one metric (sample std, ddof=1)."""
    by_arm: dict[str, dict[object, list[float]]] = {}
    for r in recs:
        if metric not in r:
            continue
        by_arm.setdefault(r["_arm"], {}).setdefault(r["_seed"], []).append(float(r[metric]))
    out = {}
    for arm, seeds in by_arm.items():
        ordered = sorted(seeds, key=lambda x: (x is None, x))
        per_seed_macro = [float(np.mean(seeds[s])) for s in ordered]
        per_seed_worst = [float(_worst(seeds[s], metric)) for s in ordered]
        n = len(ordered)
        out[arm] = {
            "macro_mean": float(np.mean(per_seed_macro)),
            "macro_std": float(np.std(per_seed_macro, ddof=1)) if n > 1 else 0.0,
            "worst_mean": float(np.mean(per_seed_worst)),
            "worst_std": float(np.std(per_seed_worst, ddof=1)) if n > 1 else 0.0,
            "n_seeds": n,
            "n_clients_per_seed": [len(seeds[s]) for s in ordered],
            "per_seed_macro": per_seed_macro,
            "per_seed_worst": per_seed_worst,
        }
    return out


def _seed_macros(recs, metric, arm):
    by_seed: dict[object, list[float]] = {}
    for r in recs:
        if r["_arm"] == arm and metric in r:
            by_seed.setdefault(r["_seed"], []).append(float(r[metric]))
    return {s: float(np.mean(v)) for s, v in by_seed.items()}


def _unit_pairs(recs, metric, ref, other):
    def idx(arm):
        return {(r["_seed"], r["_entity"]): float(r[metric])
                for r in recs if r["_arm"] == arm and metric in r}
    A, B = idx(ref), idx(other)
    keys = sorted(set(A) & set(B), key=lambda k: (k[0] is None, k))
    return np.array([A[k] for k in keys]), np.array([B[k] for k in keys])


def _contrast(recs, metric, ref, other) -> dict:
    """Seed-level paired contrast (SEED = independent unit) + sign-consistency."""
    lower = metric in LOWER_BETTER
    ra, oa = _seed_macros(recs, metric, ref), _seed_macros(recs, metric, other)
    seeds = sorted(set(ra) & set(oa), key=lambda x: (x is None, x))
    d = np.array([ra[s] - oa[s] for s in seeds])
    if lower:
        d = -d                                       # positive d ⇒ ref is BETTER
    a_u, b_u = _unit_pairs(recs, metric, ref, other)
    du = (a_u - b_u)
    if lower:
        du = -du
    p = None
    try:
        from scipy.stats import wilcoxon
        if len(d) >= 1 and not np.allclose(d, 0):
            _, p = wilcoxon(d)                       # two-sided seed-level signed-rank
    except ImportError:
        pass
    return {
        "n_seeds": len(seeds),
        "delta_mean": float(d.mean()) if len(d) else float("nan"),
        "delta_std": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
        "wins_seed": int((d > 0).sum()),
        "wins_unit": int((du > 0).sum()),
        "n_units": int(len(du)),
        "p_seedlevel": p,
    }


def _fmt(stats: dict) -> str:
    return (f"{stats['macro_mean']:.3f}±{stats['macro_std']:.3f} "
            f"(worst {stats['worst_mean']:.3f}±{stats['worst_std']:.3f})")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("jsons", nargs="+",
                   help="federated_eval.py --out-json file(s), one per dataset OR per cluster.")
    p.add_argument("--metrics", type=str, default=",".join(DEFAULT_METRICS))
    p.add_argument("--primary", type=str, default="vus_pr", help="endpoint for the paired contrast.")
    p.add_argument("--ref-arm", type=str, default="federated", help="reference arm for contrasts.")
    p.add_argument("--primary-contrast", type=str, default="federated_shared",
                   help="the single pre-registered primary contrast (isolates partial personalization).")
    p.add_argument("--csv", type=str, default=None,
                   help="write the per-(dataset,cluster,arm,seed,entity) primary-metric matrix here.")
    args = p.parse_args()
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    csv_rows = []
    for path in args.jsons:
        label, dataset, cluster, recs, meta = _load(path)
        arms_seen = sorted({r["_arm"] for r in recs},
                           key=lambda a: ({"local": 0, "centralized": 1, "federated": 2,
                                           "federated_shared": 3}.get(a, 9), a))
        seeds = sorted({r["_seed"] for r in recs}, key=lambda x: (x is None, x))
        prov = f"  [commit {str(meta.get('commit'))[:8]}{'-dirty' if meta.get('dirty') else ''}]" \
               if meta.get("commit") else ""
        print(f"\n================ {label}  (seeds={seeds}, n_arms={len(arms_seen)})"
              f"{prov} ================")

        # Two-level table. Width the arm column to the longest arm name actually
        # present — a fixed ljust(18) shifted long names out of alignment with the
        # header (the offender was 22 chars; today's longest,
        # `federated_fedavg_cb_sharedprior`, is 31, so the dynamic width is load-bearing).
        w = max(18, max(len(a) for a in arms_seen) + 2)
        header = "arm".ljust(w) + "".join(f"{m:>33}" for m in metrics)
        print(header)
        for arm in arms_seen:
            row = arm.ljust(w)
            for m in metrics:
                st = _two_level(recs, m).get(arm)
                row += (_fmt(st).rjust(33)) if st else " " * 33
            print(row)
        print("(value = across-seed mean±std[ddof=1] of the per-seed macro-mean; "
              "worst = across-seed mean±std of the per-seed worst client)")

        # Balance check — average-of-averages is only unbiased with equal #clients/seed.
        for arm in arms_seen:
            counts = _two_level(recs, args.primary).get(arm, {}).get("n_clients_per_seed", [])
            if counts and len(set(counts)) > 1:
                print(f"  [warn] {arm}: unequal clients across seeds {counts} "
                      f"(NaN/missing metric dropped) — its average-of-averages is unbalanced.")

        # Seed-level paired contrast on the primary endpoint.
        if any(r["_arm"] == args.ref_arm for r in recs):
            print(f"\n  seed-level paired contrast on '{args.primary}'  (ref = {args.ref_arm}; "
                  f"SEED is the unit — clients within a seed are NOT independent):")
            for other in arms_seen:
                if other == args.ref_arm:
                    continue
                c = _contrast(recs, args.primary, args.ref_arm, other)
                tag = "  ◀ PRIMARY (isolates partial-personalization B)" if other == args.primary_contrast else ""
                pstr = (f"Wilcoxon(n={c['n_seeds']}) p={c['p_seedlevel']:.3f}"
                        if c["p_seedlevel"] is not None else "Wilcoxon p=n/a")
                print(f"    {args.ref_arm} vs {other:<17}: Δ={c['delta_mean']:+.3f}±{c['delta_std']:.3f} "
                      f"| wins {c['wins_seed']}/{c['n_seeds']} seeds, {c['wins_unit']}/{c['n_units']} units "
                      f"| {pstr}{tag}")
            print("    (sign-consistency is the honest headline at this n; two-sided exact Wilcoxon "
                  "floor at n=4 is p=0.125. Treat the PRIMARY contrast as confirmatory, the rest as "
                  "descriptive — no multiple-comparison correction is then needed.)")
        else:
            print(f"\n  [warn] ref arm '{args.ref_arm}' not in records — skipping contrasts.")

        if args.csv is not None:
            for r in recs:
                if args.primary in r:
                    csv_rows.append((dataset, cluster, r["_arm"], r["_seed"],
                                     r["_entity"], r[args.primary]))

    if args.csv is not None and csv_rows:
        out = Path(args.csv); out.parent.mkdir(parents=True, exist_ok=True)
        lines = ["dataset,cluster,arm,seed,entity,%s" % args.primary]
        lines += [f"{d},{c},{a},{s},{e},{v}" for (d, c, a, s, e, v) in csv_rows]
        out.write_text("\n".join(lines) + "\n")
        print(f"\n[csv] wrote {len(csv_rows)} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
