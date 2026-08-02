#!/usr/bin/env python3
"""FLOOR vs deep — matched statistics on identical (cluster, entity) keys.

Exists because nothing else in the repo does this: `fed_aggregate.py` tests at
the SEED level (useless at n_seeds=1 against a deterministic baseline) and
`summarize_converged.py:64-65` only ever pairs the two literals
`local`/`centralized`. The rule this file enforces is the one §8.9 of
documentation/FLOOR_BASELINE.md now imposes: no number enters the write-up
without a script that regenerates it.

Input is `artifacts/floor/<ds>_all_models.csv` (produced by floor_table.py).
Pairing is on (cluster, entity); the honest unit is the entity on wsd_fed and
the CLUSTER on ucr_split — pass --unit cluster there and the per-cluster median
is taken first.

Reported per model:
  n, the arm's marginal median, the PAIRED median difference Δmed = median(a-b), the
  Hodges-Lehmann estimate, mean Δ, SD(Δ), SE, paired Wilcoxon p, a bootstrap CI of the
  paired median, and — when --tost-margin is given — a TOST verdict. Also m80, the margin
  that WOULD be needed for 80 % power at that arm's own paired n, because a margin wide
  enough to "prove equivalence" is manufactured, not found.

⚠ Δmed is median(a-b), NOT median(a)-median(b). Until 2026-07-30 this column printed the
latter — the difference of MARGINAL medians — beside a signed-rank p and a bootstrap CI that
both describe the former. Medians are not additive under pairing, so the two disagree: on the
impulse ablation the old column read +0.181 where the paired median is +0.058 (3.1x), and the
printed point estimate fell outside the printed CI on the same line. Every effect size in
FLOOR_BASELINE §0.6 came from the old column and must be re-derived.

    python scripts/floor_stats.py --dataset wsd_fed --metric vus_pr
    python scripts/floor_stats.py --dataset wsd_fed --metric vus_pr --tost-margin 0.15
"""
from __future__ import annotations

import argparse
import collections
import csv
import re
from pathlib import Path

# The pre-registered headline head. Everything is paired AGAINST this; if it is not on disk
# the run is not ready to be summarised, and picking something else is never acceptable.
CANONICAL_REF = "floor_ma_c_k10"

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parent.parent


def load(ds: str, metric: str, unit: str, keep_stale: bool, csv_dir: str):
    path = Path(csv_dir)
    if not path.is_absolute():
        path = REPO / path
    path = path / f"{ds}_all_models.csv"
    if not path.exists():
        raise SystemExit(f"missing {path} — run scripts/floor_table.py first "
                         f"(and point --csv-dir at its --out-dir if you moved it)")
    by = collections.defaultdict(dict)          # (tree, model) -> key -> value
    excluded: collections.Counter = collections.Counter()
    for r in csv.DictReader(path.open()):
        if not r.get(metric):
            continue
        # floor_table.TRUST emits "ok" / "archived" / "archived-stale" — never the bare string
        # "stale", which is the vocabulary of the pre-2026-07-29 CSVs. Testing for that one
        # literal made the filter UNSATISFIABLE when the trees were renamed: --keep-stale
        # became a no-op and every un-fingerprinted pre-cohort tree was pooled into the stats
        # by DEFAULT, which is the opposite of the intended default. Key on "not ok" instead,
        # so a tree added later is excluded until someone decides otherwise.
        if r.get("trust", "ok") != "ok" and not keep_stale:
            excluded[r["trust"]] += 1
            continue
        key = r["cluster"] if unit == "cluster" else (r["cluster"], r["entity"])
        # Pair on the MODEL, never on the arm. On a per-series-window build the same model
        # wears one arm name per distinct window (79 of them on ucr_split_w2p), so pairing on
        # the arm gives ~5-entity fragments and nothing to test. Older CSVs have no `model`
        # column: derive it the way floor_eval.model_tag does.
        model = r.get("model") or re.sub(r"__w\d+", "", r["arm"])
        by[(r["tree"], model)].setdefault(key, []).append(float(r[metric]))
    if excluded:
        print("[stats] EXCLUDED pre-cohort rows (no fingerprint was ever recorded for those "
              "trees, so they are not matchable — pass --keep-stale to include them): "
              + ", ".join(f"{k}={v}" for k, v in sorted(excluded.items())))
    return {k: {kk: float(np.median(v)) for kk, v in d.items()} for k, d in by.items()}


def hodges_lehmann(d: np.ndarray) -> float:
    """Median of the Walsh averages (d_i + d_j)/2, i <= j.

    This is the point estimate the WILCOXON SIGNED-RANK test is consistent with: the value the
    test would place at the centre. Reported next to the paired median because a signed-rank
    p-value with a plain median beside it invites the reader to assume they estimate the same
    thing."""
    n = len(d)
    if n > 2000:                                    # O(n^2) pairs; sample above that
        rng = np.random.default_rng(0)
        d = d[rng.choice(n, 2000, replace=False)]
    iu = np.triu_indices(len(d))
    return float(np.median((d[iu[0]] + d[iu[1]]) / 2.0))


def boot_ci(d: np.ndarray, n_boot: int = 10_000, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    meds = np.median(d[idx], axis=1)
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="wsd_fed")
    ap.add_argument("--metric", default="vus_pr")
    ap.add_argument("--floor-arm", default=None, help="default: the ma_c k=10 arm")
    ap.add_argument("--unit", choices=["entity", "cluster"], default=None)
    ap.add_argument("--tost-margin", type=float, default=None)
    ap.add_argument("--keep-stale", action="store_true", help="include the pre-purge tree")
    ap.add_argument("--csv-dir", default="artifacts/floor",
                    help="dir holding <ds>_all_models.csv (floor_table.py --out-dir)")
    ap.add_argument("--min-n", type=int, default=5,
                    help="arms with fewer paired keys than this are reported as SKIPPED, "
                         "never silently dropped")
    args = ap.parse_args()

    unit = args.unit or ("cluster" if args.dataset.startswith("ucr_split") else "entity")
    data = load(args.dataset, args.metric, unit, args.keep_stale, args.csv_dir)
    floor_keys = [k for k in data if k[0] == "floor"]
    if not floor_keys:
        raise SystemExit("no floor rows in the CSV")
    # The reference is the PRE-REGISTERED head or nothing. The old code fell back to
    # `floor_keys[0]` — the alphabetically first floor arm — whenever the exact string
    # `floor_ma_c_k10` was absent. On ucr_split_w2p that string IS absent, because every arm
    # there carries a per-series `__w<W>` suffix, so the reference silently became
    # `floor_ar_p32_lam0.0001__w408`: every delta and p-value on that dataset was computed
    # against the ridge-AR head instead of the moving average, announced only by one header
    # line nobody reads. Pairing on the model (see load()) removes the suffix, and an
    # arbitrary fallback is now fatal rather than silent.
    if args.floor_arm:
        ref = ("floor", args.floor_arm)
        if ref not in data:
            raise SystemExit(f"--floor-arm {args.floor_arm!r} not in CSV; available: "
                             f"{sorted(a for _t, a in floor_keys)}")
    else:
        cands = sorted(k for k in floor_keys if k[1] == CANONICAL_REF)
        if not cands:
            raise SystemExit(
                f"the pre-registered reference {CANONICAL_REF!r} is not in the CSV, and there "
                f"is no safe default: pairing against an arbitrary arm silently changes what "
                f"every delta MEANS.\n  floor models present: "
                f"{sorted(a for _t, a in floor_keys)}\n"
                f"  run the headline batch (run_floor.sh wsd / ucr), or name one explicitly "
                f"with --floor-arm.")
        ref = cands[0]
    base = data[ref]

    print(f"dataset={args.dataset}  metric={args.metric}  unit={unit}  "
          f"reference={ref[1]}  n_ref={len(base)}")
    print(f"reference median={np.median(list(base.values())):.4f} "
          f"mean={np.mean(list(base.values())):.4f}\n")
    hdr = (f'{"tree":14s} {"model":32s} {"n":>3s} {"median":>8s} {"Δmed":>8s} {"HL":>8s} '
           f'{"meanΔ":>8s} {"SD":>7s} {"SE":>7s} {"p":>9s} {"CI(Δmed)":>18s} '
           f'{"m80":>6s}  verdict')
    print(hdr); print("-" * len(hdr))

    skipped, m80s = [], []
    for key in sorted(data):
        if key[0] == "floor" and key[1] == ref[1]:
            continue
        other = data[key]
        common = sorted(set(base) & set(other), key=str)
        if len(common) < args.min_n:
            # Never a silent drop: an arm that vanishes from the table looks like an
            # arm that was never run, and the two have opposite meanings.
            skipped.append((key, len(common), len(other)))
            continue
        a = np.array([other[k] for k in common])
        b = np.array([base[k] for k in common])
        d = a - b
        p = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        lo, hi = boot_ci(d)
        sd, se = float(np.std(d, ddof=1)), float(np.std(d, ddof=1) / np.sqrt(len(d)))
        # The margin an equivalence claim would need, at THIS arm's PAIRED n — not at
        # n_ref. The old line used n = len(base), the reference arm's key count, which
        # is >= the paired n for every arm and therefore reports a margin SMALLER than
        # the one actually required: exactly the direction that makes a TOST look
        # adequately powered when it is not.
        m80 = 2.8 * sd / np.sqrt(len(d))              # ~80% power, two one-sided t
        m80s.append((key[1], len(d), m80))
        if p < 0.05:
            verdict = "beats floor" if np.median(d) > 0 else "BELOW floor"
        else:
            verdict = "not distinguishable"
        if args.tost_margin is not None:
            m = args.tost_margin
            t1 = stats.ttest_1samp(d, -m, alternative="greater").pvalue
            t2 = stats.ttest_1samp(d, m, alternative="less").pvalue
            tost = "EQUIV" if max(t1, t2) < 0.05 else "not equiv"
            if m < m80:
                tost += f"(UNDERPOWERED: margin {m:g} < m80 {m80:.3f})"
            verdict += f" | TOST({m:g}) {tost}"
        # Δmed is the PAIRED median, median(d) — NOT median(a) - median(b). Those are different
        # functionals (medians are not additive under pairing) and the old column printed the
        # marginal difference on the same line as a signed-rank p and a bootstrap CI of the
        # PAIRED median. On the impulse ablation that overstated the effect 3.1x (+0.181 vs
        # +0.058) and put the printed point estimate OUTSIDE the printed CI on its own line.
        # The marginal medians are still both visible: `median` here and `reference median`
        # in the header above.
        print(f"{key[0]:14s} {key[1]:32s} {len(d):3d} {np.median(a):8.4f} "
              f"{np.median(d):8.3f} {hodges_lehmann(d):8.3f} {d.mean():8.3f} "
              f"{sd:7.3f} {se:7.3f} "
              f"{p:9.5f} [{lo:+.3f},{hi:+.3f}] {m80:6.3f}  {verdict}")

    if skipped:
        print(f"\nSKIPPED (fewer than --min-n={args.min_n} keys paired with "
              f"{ref[1]} — NOT the same as 'not run'):")
        for key, n_common, n_own in skipped:
            print(f"  {key[0]:14s} {key[1]:32s} paired={n_common:3d} of its own {n_own}")

    if m80s:
        lo_m, hi_m = min(m[2] for m in m80s), max(m[2] for m in m80s)
        ns = sorted({m[1] for m in m80s})
        print(f"\nm80 = the TOST margin needed for ~80% power AT EACH ARM'S OWN PAIRED n "
              f"(n in {ns[0]}..{ns[-1]}): {lo_m:.3f}–{hi_m:.3f}.")
        print("A margin wider than an arm's own m80 FABRICATES the 'equivalent' verdict — "
              "declare the achieved power whenever equivalence is claimed "
              "(FLOOR_BASELINE.md §6).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
