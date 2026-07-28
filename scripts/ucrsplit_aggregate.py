#!/usr/bin/env python3
"""ucrsplit_aggregate.py — the ucr_split results table, in metrics that are actually comparable.

`scripts/fed_aggregate.py` reports VUS-PR. On `ucr_split` that is the WRONG headline and the
mistake is not subtle, so this script exists rather than a flag on that one.

## Why not VUS-PR across clusters

Two independent mechanisms, both measured:

1. **Threshold-grid cap (implementation).** `metrics/vus_local.py` sweeps only `thre=250`
   thresholds, spaced over the RANKS of the score. The smallest resolvable prediction set is
   `N/250`, so when positives `P << N/250` the high-precision region of the PR curve cannot be
   represented and VUS-PR is capped at roughly `P*250/N` no matter how good the detector is.
   Proof on identical scores: ucr_191 scores AUPRC **0.994** (sklearn, all thresholds) and
   VUS-PR **0.211** (250-point grid) — a 4.7x gap that is pure quantisation.
2. **Rate-dependent baseline (intrinsic).** PR metrics baseline at the positive rate; on
   ucr_split that spans 0.0005-0.0335, a 60x range. Measured corr(VUS-PR, rate) = **+0.766**
   vs corr(AUROC, rate) = +0.264.

Net effect: ordering clusters by VUS-PR INVERTS detector quality — ucr_139 (AUROC 0.912)
outranks ucr_191 (AUROC 1.000). A median of VUS-PR over clusters describes the anomaly-rate
distribution, not the model.

**Within a cluster VUS-PR is fine** (all arms share N, P and the grid), so paired
within-cluster differences are reported and are valid. Across clusters we use
`paper_top1/3/5` — rate-free, grid-free, and the currency TimeVQVAE-AD's Table 1 reports in.

## Unit of analysis

The **cluster**, never the client. The 5 clients of a cluster are scored on the SAME test
series with the SAME labels, so their records are 5 models on one event. Every cross-cluster
statistic here is computed on the per-cluster macro-mean; treating the 1130 clients as
independent inflates significance by ~sqrt(5).

## Exclusion rule — declared HERE, before the results are read

A cluster is dropped when the SKYLINE fails, i.e. `centralized` cannot solve it: no arm's
number is interpretable on a series the pooled model cannot do either. Threshold is
`--skyline-min` on the primary metric (default 0: drop only outright zeros). Dropped clusters
are always listed and counted — never silently.
Known degenerate: `ucr_189`, whose archive label is **2 samples** in a 145k test (AUROC 0.002).

    python scripts/ucrsplit_aggregate.py                       # artifacts/ucrsplit
    python scripts/ucrsplit_aggregate.py --res artifacts/ucrsplit_w2p_cb64 --csv out.csv
    python scripts/ucrsplit_aggregate.py --compare artifacts/ucrsplit artifacts/ucrsplit_w2p_cb64
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
PRIMARY = "paper_top1_acc_at_64"
TOPK = ["paper_top1_acc_at_64", "paper_top3_acc_at_64", "paper_top5_acc_at_64"]
# Table 1 of arXiv 2311.12550v5, full 250-series archive. Context only: our clients see
# 10-30% of a train and even `centralized` sees 90% of one cluster, so this is NOT a
# like-for-like target — it is the scale the currency is denominated in.
PAPER = {"TimeVQVAE-AD (published)": (0.708, 0.776, 0.824),
         "Matrix Profile STUMPY": (0.512, 0.684, 0.744),
         "Convolutional AE": (0.352, 0.412, 0.448)}
# Ordered as a DECOMPOSITION LADDER: each rung adds exactly one thing to the one above, so
# adjacent differences are single-axis contrasts rather than bundles.
#
#   local                    nothing crosses the network                       (the arm to beat)
#   commoninit_cblocal       + shared encoder INIT                             (init alone)
#   cb_only                  + shared CODEBOOK, independent init               (dictionary alone)
#   commoninit_cbshared      + shared init AND shared codebook                 (= legacy `commoninit`)
#   cb_only_ema              codebook + server-side EMA over rounds
#   fedavg / fedprox         + the encoder WEIGHT-AVERAGED every round
#   fedavg_cblocal / fedprox_cblocal   encoder averaged, dictionary NOT shared (encoder alone)
#   fedproto                 encoder in FUNCTION space (no weight averaging)
#   centralized              pool everything                                   (the skyline)
#
# `cb_only` vs `commoninit_cbshared` is the sharpest read on the shared init: identical
# codebook regime, and the ONLY difference is that cb_only leaves fed_encoder="off" (each
# client draws its own encoder init) while commoninit broadcasts one.
ARM_ORDER = ["local",
             "federated_enc_commoninit_cblocal",
             "federated_cb_only",
             "federated_enc_commoninit_cbshared", "federated_enc_commoninit",
             "federated_cb_only_ema",
             "federated_enc_fedavg", "federated_enc_fedprox",
             "federated_enc_fedavg_cblocal", "federated_enc_fedprox_cblocal",
             "federated_enc_fedproto",
             "centralized"]
SHORT = {a: a.replace("federated_enc_", "").replace("federated_", "") for a in ARM_ORDER}


def canon_arm(arm_dir: str) -> str:
    """On-disk arm directory → canonical arm id.

    federated_eval appends the hyperparameters that define a run to the directory name
    (`federated_enc_fedprox_mu0.1`, `federated_enc_fedproto_lam0.1_count`) and a `_cb-local`
    marker when the codebook regime is non-default. Without folding those back, μ=0.1 and
    μ=1.0 are two unrelated rows and the ladder order collapses — so match the LONGEST
    canonical prefix and re-attach the codebook marker.
    """
    a = arm_dir
    cb_local = False
    if a.endswith("_cb-local"):
        a, cb_local = a[: -len("_cb-local")], True
    best = None
    for k in ARM_ORDER:
        if a == k or a.startswith(k + "_"):
            if best is None or len(k) > len(best):
                best = k
    a = best or a
    if cb_local and not a.endswith("_cblocal"):
        a += "_cblocal"
    return a


def load_reports(res_root: Path) -> tuple[dict, dict]:
    """(per_client, truncated) — per_client[cluster][arm][entity] = {metric: value}.

    Reads report.json (~40 metrics/client) rather than the summary jsons, which surface only
    6 and omit the top-k block entirely.
    """
    per = defaultdict(lambda: defaultdict(dict))
    pat = str(res_root / "ckpt" / "*" / "seed*" / "*" / "*" / "report.json")
    for f in glob.glob(pat):
        parts = f.split(os.sep)
        cluster, arm, entity = parts[-5], parts[-3], parts[-2]
        arm = canon_arm(arm)
        try:
            per[cluster][arm][entity] = json.load(open(f))
        except Exception as e:                       # a job killed mid-write
            print(f"[warn] unreadable {f}: {e}", file=sys.stderr)
    trunc = {}
    for f in glob.glob(str(res_root / "ckpt" / "**" / "fed_history.json"), recursive=True):
        try:
            h = json.load(open(f))
        except Exception:
            continue
        s1 = h.get("stage1", [])
        if s1 and s1[-1].get("truncated"):
            trunc[os.path.relpath(f, res_root)] = s1[-1].get("round")
    return per, trunc


def cluster_means(per: dict, metric: str) -> dict:
    """{cluster: {arm: macro-mean over its clients}} — the ONLY legitimate cross-cluster unit."""
    out = defaultdict(dict)
    for cl, arms in per.items():
        for arm, ents in arms.items():
            vals = [r.get(metric) for r in ents.values()]
            vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
            if vals:
                out[cl][arm] = float(np.mean(vals))
    return out


def paired(cm: dict, a: str, b: str) -> tuple[np.ndarray, np.ndarray, list]:
    """Clusters holding BOTH arms → (values_a, values_b, cluster_ids)."""
    cl = sorted(c for c in cm if a in cm[c] and b in cm[c])
    return (np.array([cm[c][a] for c in cl]), np.array([cm[c][b] for c in cl]), cl)


def wilcoxon_p(d: np.ndarray) -> float:
    nz = d[d != 0]
    if nz.size < 6:
        return float("nan")
    try:
        from scipy.stats import wilcoxon
        return float(wilcoxon(nz).pvalue)
    except Exception:
        return float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--res", type=Path, default=REPO / "artifacts" / "ucrsplit")
    ap.add_argument("--compare", type=Path, nargs=2, default=None,
                    help="two result roots → per-cluster A/B on the primary metric "
                         "(e.g. the W=128 run vs the W=2*period run)")
    ap.add_argument("--skyline-min", type=float, default=0.0,
                    help="drop a cluster when centralized's primary metric is <= this "
                         "(default 0.0 = drop only outright failures). ALWAYS reported.")
    ap.add_argument("--reference", type=str, default="local",
                    help="arm every other arm is paired against (default: the arm to beat)")
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    if args.compare:
        return do_compare(args)

    per, trunc = load_reports(args.res)
    if not per:
        raise SystemExit(f"no report.json under {args.res}/ckpt — wrong --res?")
    cm = cluster_means(per, PRIMARY)

    # ── exclusion, declared before results are read ──────────────────────────────────
    have_sky = {c for c in cm if "centralized" in cm[c]}
    dropped = sorted(c for c in have_sky if cm[c]["centralized"] <= args.skyline_min)
    kept = sorted(have_sky - set(dropped))
    no_sky = sorted(set(cm) - have_sky)

    print(f"=== ucr_split — {args.res.name} ===")
    print(f"clusters with any result: {len(cm)}   with a skyline: {len(have_sky)}   "
          f"kept: {len(kept)}")
    if dropped:
        print(f"DROPPED (centralized {PRIMARY} <= {args.skyline_min}): {len(dropped)} -> "
              f"{', '.join(dropped)}")
    if no_sky:
        print(f"no centralized yet (excluded from paired stats): {len(no_sky)}")
    if trunc:
        print(f"!! TRUNCATED, NOT CONVERGED: {len(trunc)} run(s) — NOT reportable (hard rule "
              f"2026-07-24): {list(trunc)[:4]}")

    present = {a for c in kept for a in cm[c]}
    extra = sorted(present - set(ARM_ORDER))
    if extra:
        print(f"[warn] arms outside the ladder (append them to ARM_ORDER): {extra}")
    # Keep the ladder order and SHOW the missing rungs: an absent row is information
    # ("cb_only was never run here"), and silently dropping it hides a hole in the design.
    arms = [a for a in ARM_ORDER if a in present] + extra
    missing = [a for a in ARM_ORDER if a not in present]

    # ── headline: top-k at cluster level ─────────────────────────────────────────────
    print(f"\n--- top-k (cluster-level macro-means, n = clusters holding the arm) ---")
    print(f"{'arm':<26}{'n':>4}{'top1':>8}{'top3':>8}{'top5':>8}")
    for a in arms:
        row = []
        for m in TOPK:
            c2 = cluster_means(per, m)
            v = [c2[c][a] for c in kept if a in c2.get(c, {})]
            row.append(np.mean(v) if v else float("nan"))
        n = sum(1 for c in kept if a in cm[c])
        print(f"{SHORT.get(a,a):<26}{n:>4}{row[0]:>8.3f}{row[1]:>8.3f}{row[2]:>8.3f}")
    for a in missing:
        print(f"{SHORT.get(a,a):<26}{0:>4}{'-':>8}{'-':>8}{'-':>8}   (non eseguito)")
    print(f"{'':-<54}")
    for name, (t1, t3, t5) in PAPER.items():
        print(f"{name:<26}{250:>4}{t1:>8.3f}{t3:>8.3f}{t5:>8.3f}")
    print("  (published = full archive, full train per series; NOT like-for-like — context only)")

    # ── paired vs the reference, at cluster level ────────────────────────────────────
    ref = args.reference
    print(f"\n--- paired vs `{ref}` on {PRIMARY} (unit = cluster) ---")
    print(f"{'arm':<26}{'n':>4}{'mean diff':>11}{'wins':>6}{'ties':>6}{'loss':>6}{'p':>9}")
    for a in arms:
        if a == ref:
            continue
        va, vb, cl = paired(cm, a, ref)
        if va.size == 0:
            continue
        d = va - vb
        print(f"{SHORT.get(a,a):<26}{d.size:>4}{d.mean():>+11.3f}"
              f"{int((d>0).sum()):>6}{int((d==0).sum()):>6}{int((d<0).sum()):>6}"
              f"{wilcoxon_p(d):>9.3f}")

    # ── VUS-PR: within-cluster only, plus the cap diagnostic ─────────────────────────
    cmv = cluster_means(per, "vus_pr")
    print(f"\n--- vus_pr paired vs `{ref}` (WITHIN-cluster diffs only; the cross-cluster "
          f"mean of vus_pr is meaningless — see the module docstring) ---")
    print(f"{'arm':<26}{'n':>4}{'mean diff':>11}{'p':>9}")
    for a in arms:
        if a == ref:
            continue
        va, vb, _ = paired(cmv, a, ref)
        if va.size == 0:
            continue
        d = va - vb
        print(f"{SHORT.get(a,a):<26}{d.size:>4}{d.mean():>+11.3f}{wilcoxon_p(d):>9.3f}")

    capped = []
    for c in kept:
        r = next(iter(per[c].get("centralized", {}).values()), None)
        if not r:
            continue
        P, N = r.get("n_pos_labels"), (r.get("n_pos_labels", 0) + r.get("n_neg_labels", 0))
        if P and N:
            cap = P * 250 / N
            if cap < 1.0:
                capped.append((c, cap, r.get("vus_pr"), r.get("auprc")))
    if capped:
        print(f"\n  VUS grid cap active on {len(capped)}/{len(kept)} clusters "
              f"(P*250/N < 1 ⇒ vus_pr cannot reach 1 regardless of the model):")
        for c, cap, v, ap in sorted(capped, key=lambda x: x[1])[:8]:
            print(f"    {c:<10} cap≈{cap:.3f}  vus_pr={v:.3f}  auprc={ap:.3f}")

    if args.csv:
        import csv
        allm = TOPK + ["vus_pr", "auroc", "auprc", "pate_f1", "affiliation_f1"]
        cms = {m: cluster_means(per, m) for m in allm}
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["cluster", "arm", "kept"] + allm)
            for c in sorted(cm):
                for a in sorted(cm[c]):
                    w.writerow([c, a, int(c in kept)] +
                               [cms[m].get(c, {}).get(a, "") for m in allm])
        print(f"\ncsv -> {args.csv}")
    return 0


def do_compare(args) -> int:
    """A/B two result roots on the primary metric, paired by cluster."""
    A, B = args.compare
    pa, _ = load_reports(A)
    pb, _ = load_reports(B)
    ca, cb = cluster_means(pa, PRIMARY), cluster_means(pb, PRIMARY)
    arms = sorted({a for c in ca for a in ca[c]} & {a for c in cb for a in cb[c]})
    print(f"=== A/B on {PRIMARY}: {A.name}  vs  {B.name} ===")
    print(f"{'arm':<26}{'n':>4}{A.name[:11]:>12}{B.name[:11]:>12}{'diff':>9}{'p':>9}")
    for a in arms:
        cl = sorted(c for c in ca if a in ca[c] and a in cb.get(c, {}))
        if not cl:
            continue
        va = np.array([ca[c][a] for c in cl]); vb = np.array([cb[c][a] for c in cl])
        d = vb - va
        print(f"{SHORT.get(a,a):<26}{len(cl):>4}{va.mean():>12.3f}{vb.mean():>12.3f}"
              f"{d.mean():>+9.3f}{wilcoxon_p(d):>9.3f}")
    print("  diff = B - A, paired by cluster. Positive ⇒ B is better.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
