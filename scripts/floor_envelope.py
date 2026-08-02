#!/usr/bin/env python3
"""FLOOR envelope — the UPPER bar of the calibration band, as a real arm.

Why this file exists. `documentation/FLOOR_BASELINE.md` §0.1a reports two bars, not
one, and insists the paper carry both:

    ma_c k=10 (the pre-registered single head)   median 0.5054 / mean 0.4421
    envelope = best of {ma_c, ma_causal, ar, pca}  median 0.5585 / mean 0.4993

The second bar is the load-bearing one for the verdicts: against `ma_c` three deep
arms clear the floor, against the envelope only two do, and `centralized` falls from
p=0.003 to p=0.058. But `floor_heads.DECISION_HEADS` was defined at line 602 and read
by NOTHING — the 0.4993/0.5585 pair was computed outside the repo and was therefore
unreproducible, in violation of §8.9 ("no number enters the write-up without a script
that regenerates it").

What the envelope IS, precisely. Per (cluster, entity) it picks the SINGLE head with
the best `--select-metric`, then reports that head's whole metric row. It is head
SELECTION, not per-metric maximisation: the latter would take `vus_pr` from one head
and `auprc` from another and report a detector nobody can build. This definition
corresponds to something a practitioner could actually deploy given an oracle, which
is what makes it a legitimate upper bound rather than a number.

It IS an oracle on the test set — selection uses the test metric, because §6 records
that no label-free criterion in this repo can order these heads (AR predicts one step
ahead, PCA reconstructs a window containing the point; their validation MSEs are not
on the same scale). So every row is stamped `_oracle: true` and the arm is named
`floor_env4`, and it must be captioned as an upper bound, never as the headline.

    python scripts/floor_envelope.py --dataset wsd_fed
    python scripts/floor_envelope.py --dataset wsd_fed --records-dir "artifacts/runs/*/floor"

Output: <out-dir>/records_<ds>.jsonl holding ONLY the synthetic arm, so the raw record
files stay raw. Feed both to the table:

    python scripts/floor_table.py --dataset wsd_fed \
        --records-dir "artifacts/floor,artifacts/floor/_derived"
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import floor_heads as FH                                          # noqa: E402
from floor_table import floor_record_paths, rel_repo              # noqa: E402

ARM = "floor_env4"


def preregistered_knobs(heads: list[str]) -> dict:
    """head -> its PRE-REGISTERED knob dict, via floor_eval's own helper so the two
    can never drift apart."""
    from floor_eval import head_knobs
    # FH.DEFAULTS has no `seed`; floor_eval.knobs_from_args adds it, and head_knobs reads
    # it for the `random` head. Supply it so a head set that includes `random` cannot
    # KeyError here instead of at the point where it matters.
    base = dict(FH.DEFAULTS, seed=0)
    return {h: head_knobs(h, base, "local", {}) for h in heads}


def is_candidate(r: dict, head: str, knobs: dict) -> bool:
    """Match a record on SEMANTICS, not on its arm-name string.

    Reconstructing the arm name would work on wsd_fed and break on ucr_split_w2p,
    where every arm carries a per-series `__w<W>` suffix — there is no single name to
    reconstruct. Matching on (head, mode, knobs, impulse, fit stride) is window-blind
    by construction, and the window is checked separately for consistency below."""
    if r.get("_head") != head:
        return False
    # zero-parameter heads are stamped `arm_invariant`, fitted ones `local`
    if r.get("_mode") not in ("local", "arm_invariant"):
        return False
    if not r.get("_impulse", False):
        return False
    if r.get("_fit_stride") not in (None, 1):
        return False
    return (r.get("_knobs") or {}) == knobs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="wsd_fed")
    ap.add_argument("--records-dir", default="artifacts/floor",
                    help="comma-separated dirs (globs allowed) holding records_<ds>.jsonl")
    ap.add_argument("--out-dir", default="artifacts/floor/_derived")
    ap.add_argument("--select-metric", default="vus_pr",
                    help="the metric the per-entity head selection maximises")
    ap.add_argument("--heads", default=",".join(FH.DECISION_HEADS),
                    help="the pre-registered decision-head set; changing it changes what "
                         "the published envelope MEANS")
    args = ap.parse_args()

    heads = [h.strip() for h in args.heads.split(",") if h.strip()]
    bad = [h for h in heads if h not in FH.HEAD_NAMES]
    if bad:
        raise SystemExit(f"unknown head(s) {bad}; available: {FH.HEAD_NAMES}")
    if sorted(heads) != sorted(FH.DECISION_HEADS):
        print(f"⚠ head set {heads} != pre-registered {FH.DECISION_HEADS}: this is a "
              f"DIFFERENT envelope and must not be reported as the published one.")

    want = preregistered_knobs(heads)
    print("[env] pre-registered knobs: "
          + ", ".join(f"{h}={want[h]}" for h in heads))

    # ── read every record, keep the local/arm-invariant decision-head rows ──────
    by_head: dict = collections.defaultdict(dict)     # head -> (cluster, entity) -> row
    clash: list = []
    paths = floor_record_paths(args.dataset, args.records_dir)
    if not paths:
        raise SystemExit(f"no records_{args.dataset}.jsonl under {args.records_dir!r}")
    arms_seen: set = set()
    for p in paths:
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            arms_seen.add(r["_arm"])
            for h in heads:
                if not is_candidate(r, h, want[h]):
                    continue
                key = (r["_cluster"], r["_entity"])
                prev = by_head[h].get(key)
                if prev is not None and prev["_arm"] != r["_arm"]:
                    clash.append((h, key, prev["_arm"], r["_arm"]))
                by_head[h][key] = r

    if clash:
        lines = "".join(f"  {h} {k}: {a} vs {b}\n" for h, k, a, b in clash[:8])
        raise SystemExit(
            f"two DIFFERENT arms match the same (head, entity) — ambiguous envelope:\n"
            f"{lines}"
            "Most likely two windows for the same series on disk. Pick one records dir, "
            "or drop the stale rows: silently taking the last would put two geometries "
            "in one bar (§8.10).")

    missing = [h for h in heads if not by_head[h]]
    if missing:
        raise SystemExit(
            "the envelope needs ALL of the pre-registered decision heads; no rows for:\n"
            + "".join(f"  {h:10s} knobs={want[h]}\n" for h in missing)
            + "A PARTIAL envelope is not a weaker envelope, it is a DIFFERENT and lower\n"
              "number that would silently understate the upper bar.\n"
              f"arms on disk: {sorted(arms_seen)}")

    # ── the entity set must be identical across heads, or the bar is not paired ──
    keysets = {h: set(by_head[h]) for h in heads}
    common = set.intersection(*keysets.values())
    union = set.union(*keysets.values())
    if common != union:
        short = {h: sorted(union - k)[:5] for h, k in keysets.items() if k != union}
        raise SystemExit(
            f"entity sets differ across heads: {len(common)} common of {len(union)}.\n"
            + "".join(f"  {h} is missing {v}{' ...' if len(v) == 5 else ''}\n"
                      for h, v in short.items())
            + "Taking a max over a ragged set would compare the envelope on easy entities\n"
              "against the single head on all of them. Finish the head batch first.")

    # ── geometry must match, or the heads are not on the same yardstick ─────────
    for field in ("_window", "_tolerance", "_eval_stride"):
        for key in common:
            distinct = {by_head[h][key].get(field) for h in heads}
            if len(distinct) > 1:
                raise SystemExit(f"heads disagree on {field} for {key}: "
                                 f"{sorted(map(str, distinct))} — different geometry, "
                                 f"not comparable (§8.10)")

    # ── select, per entity, the head with the best --select-metric ──────────────
    out_rows, wins, no_metric = [], collections.Counter(), []
    for key in sorted(common):
        cand = []
        for h in heads:
            v = by_head[h][key].get(args.select_metric)
            if isinstance(v, (int, float)) and np.isfinite(v):
                cand.append((float(v), h))
        if not cand:
            no_metric.append(key)
            continue
        _best_v, best_h = max(cand)
        wins[best_h] += 1
        src = by_head[best_h][key]
        row = dict(src)
        row.update({
            "_arm": ARM,
            # MUST be overwritten. `row = dict(src)` inherits the WINNING HEAD's `_model`, and
            # `_model` is the analysis identity everything downstream groups and pairs on
            # (floor_table.collect / floor_table.main / floor_stats.load). Leaving it meant the
            # envelope rows were silently absorbed into the winning heads as phantom extra
            # seeds — no floor_env4 row in the CSV, none in the stats table, and the
            # 0.4993/0.5585 pair still not regenerable: exactly the §8.9 violation this file
            # exists to fix. It failed SILENTLY: the JSONL was written and nothing errored.
            "_model": ARM,
            "_head": "envelope",
            "_mode": "oracle_max",
            "_knobs": {"heads": heads, "select_metric": args.select_metric},
            "_oracle": True,
            "_envelope_winner": best_h,
            "_envelope_source_arm": src["_arm"],
            "_note": ("UPPER BOUND, NOT THE HEADLINE: the head is selected per entity on "
                      f"test-set {args.select_metric}, because no label-free criterion in "
                      "this repo can order these heads (FLOOR_BASELINE.md §6). Report next "
                      "to the pre-registered single head, never instead of it."),
        })
        out_rows.append(row)

    if no_metric:
        print(f"⚠ {len(no_metric)} entities had no finite {args.select_metric} on any "
              f"head and are ABSENT from the envelope: {no_metric[:5]}")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"records_{args.dataset}.jsonl"
    kept = [r for r in (json.loads(l) for l in out.read_text().splitlines() if l.strip())
            if r.get("_arm") != ARM] if out.exists() else []
    out.write_text("".join(json.dumps(r) + "\n" for r in kept + out_rows))

    v = np.array([r[args.select_metric] for r in out_rows])
    single = {h: np.array([by_head[h][k][args.select_metric] for k in sorted(common)
                          if isinstance(by_head[h][k].get(args.select_metric),
                                        (int, float))]) for h in heads}
    print(f"\n{ARM}  n={len(v)}  {args.select_metric}: "
          f"median={np.median(v):.4f} mean={np.mean(v):.4f}")
    print(f"{'single heads':14s} " + "  ".join(f"{h}={np.median(s):.4f}"
                                              for h, s in single.items() if len(s)))
    print("wins per head (no head dominates entity by entity — that is why the envelope "
          "is above every single head): "
          + ", ".join(f"{h}={wins[h]}" for h in heads))
    print(f"-> {rel_repo(out)}")
    print("\nFeed BOTH dirs to the table:\n"
          f"  python scripts/floor_table.py --dataset {args.dataset} "
          f'--records-dir "{args.records_dir},{args.out_dir}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
