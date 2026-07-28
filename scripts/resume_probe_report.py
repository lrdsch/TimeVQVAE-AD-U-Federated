#!/usr/bin/env python3
"""Read the convergence probe: did the arm converge, and did the METRIC move?

    /home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10 scripts/resume_probe_report.py

For each probed cluster it prints two things, in this order, because they answer different
questions and only the second one decides anything:

  1. THE CURVE — the tail of the cohort validation loss, whether the run stopped on patience
     (`CONVERGED`) or ran out of budget, and how much the loss still moved per round at the
     end. This is what "did it converge?" means, shown rather than asserted.

  2. THE METRIC — per-entity VUS-PR before (the truncated 30-round run) versus after
     (converged), paired per entity. This is the only thing a result table reports, and the
     two need not agree: the anomaly score contains a reconstruction term
     (`s_local = (x - recon)^2`, pipeline/detect.py), so a stage 1 that reconstructs anomalies
     more faithfully is a WORSE detector. A falling val loss is therefore not evidence that
     detection improved, and can be evidence against it.

Reads only what the runs wrote — no re-training, no GPU.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PROBE = REPO / "artifacts" / "resume_probe"
CKPT_ROOT = REPO / "artifacts" / "fed_eval"

# THE "BEFORE" MUST BE THE RUN THAT WROTE THE CHECKPOINTS, and that is not the obvious one.
# `artifacts/fed_eval/<ds>/<cl>/seed0/federated_cb_only` is the DEFAULT scratch path, so it is
# whichever `federated_cb_only` job ran LAST — and that was the cb_only baseline leg of
# run_cbonly_ema_sweep.sh (2026-07-22 18:23), not run_cbonly_converged.sh (03:18) whose logs
# live in logs/cbonly_converged/. The two runs end at materially different points (wsd c0:
# val 0.0562 vs 0.1036), so pairing the continuation against the wrong one would manufacture
# a ~50% "improvement" that is really just two different runs. Verified by the boundary being
# continuous against this source (0.0456 -> 0.0471 train, 0.0562 -> 0.0550 val) and not
# against the other. `_provenance` re-checks the pairing every time this report runs.
BEFORE = {}
for _c in ("c0", "c1", "c2", "c3"):
    BEFORE[("wsd_fed", _c)] = REPO / "artifacts" / "cbema_sweep" / f"wsd_fed_{_c}__federated_cb_only.json"
for _m in ("M1_rotary", "M2_valve", "M3_pump", "M4_cardiac", "M5_bearing", "M6_drive"):
    BEFORE[("toy_fed_uni", _m)] = (REPO / "artifacts" / "cbema_sweep"
                                   / f"toy_fed_uni_{_m}__federated_cb_only.json")


def _provenance(ds: str, cl: str, before_json: Path) -> str:
    """Is the BEFORE json plausibly from the run that wrote the resumed checkpoints?

    Compares mtimes: the json is written at the very end of a run, the checkpoints a few
    minutes earlier in the same run. A gap of hours means they are different runs and the
    before/after pairing is invalid — the single most damaging error this report could make,
    so it is checked rather than assumed."""
    ck = sorted((CKPT_ROOT / ds / cl / "seed0" / "federated_cb_only").glob("*/stage1.ckpt"))
    if not ck or not before_json.exists():
        return "  [provenance: UNVERIFIED — missing checkpoints or before-json]"
    gap_h = abs(ck[0].stat().st_mtime - before_json.stat().st_mtime) / 3600
    tag = "OK" if gap_h < 2 else "*** MISMATCH ***"
    return (f"  [provenance {tag}: ckpt and before-json written {gap_h:.1f} h apart"
            + ("" if gap_h < 2 else " — these are DIFFERENT runs; the comparison below is invalid]")
            + ("]" if gap_h < 2 else ""))


def _per_entity(path: Path, metric: str = "vus_pr") -> dict:
    """{entity: metric} from a federated_eval --out-json, seed 0 only."""
    try:
        recs = json.loads(path.read_text(encoding="utf-8")).get("records", [])
    except Exception:
        return {}
    return {r["_entity"]: r[metric] for r in recs
            if r.get("_seed") in (0, None) and isinstance(r.get(metric), (int, float))}


def _history(out_dir: Path) -> list:
    """The per-round stage-1 history of the continued run."""
    for p in sorted(out_dir.rglob("fed_history.json")):
        try:
            h = json.loads(p.read_text(encoding="utf-8")).get("stage1") or []
        except Exception:
            continue
        if h:
            return h
    return []


def main() -> int:
    if not PROBE.exists():
        print(f"no probe output at {PROBE} — run scripts/resume_probe.sh first")
        return 1

    jsons = sorted(PROBE.glob("*__cbonly_converged.json"))
    if not jsons:
        print(f"no finished probe jobs in {PROBE} (expected *__cbonly_converged.json)")
        return 1

    deltas_all = []
    for j in jsons:
        stem = j.name.replace("__cbonly_converged.json", "")
        ds = "wsd_fed" if stem.startswith("wsd_fed") else "toy_fed_uni"
        cl = stem[len(ds) + 1:]
        print("=" * 78)
        print(f"{ds} / {cl}")

        # ── 1. the curve ────────────────────────────────────────────────────
        hist = _history(PROBE / f"{ds}_{cl}")
        if hist:
            vals = [(h.get("round"), h.get("val_loss")) for h in hist
                    if isinstance(h.get("val_loss"), (int, float))]
            stopped = any(h.get("converged_stop") for h in hist)
            print(f"  continued rounds {hist[0].get('round')}..{hist[-1].get('round')}   "
                  f"{'STOPPED ON PATIENCE (converged)' if stopped else 'ran out of budget — NOT converged'}")
            if len(vals) >= 2:
                print("   round     val        Δ vs previous")
                for (r0, v0), (r1, v1) in list(zip(vals, vals[1:]))[-6:]:
                    print(f"   {r1:<8} {v1:<10.5f} {(v1 - v0) / v0 * 100:+.2f}%")
                first, last = vals[0][1], vals[-1][1]
                print(f"  val over the continuation: {first:.5f} -> {last:.5f} "
                      f"({(last - first) / first * 100:+.1f}%)")
        else:
            print("  [no fed_history.json found — cannot show the curve]")

        # ── 2. the metric ───────────────────────────────────────────────────
        before_json = BEFORE.get((ds, cl), Path("/nonexistent"))
        print(_provenance(ds, cl, before_json))
        after = _per_entity(j)
        before = _per_entity(before_json)
        shared = sorted(set(before) & set(after))
        if not shared:
            print(f"  [no paired per-entity VUS-PR — expected {before_json}]")
            continue
        d = [after[e] - before[e] for e in shared]
        deltas_all += d
        print(f"  VUS-PR, paired over {len(shared)} entities:")
        print(f"    truncated (30 rounds) : {statistics.mean(before[e] for e in shared):.4f}")
        print(f"    converged             : {statistics.mean(after[e] for e in shared):.4f}")
        print(f"    mean Δ                : {statistics.mean(d):+.4f}"
              + (f"   (sd {statistics.stdev(d):.4f})" if len(d) > 1 else ""))
        print(f"    entities improved     : {sum(x > 0 for x in d)}/{len(d)}")
        worst = min(zip(d, shared)); best = max(zip(d, shared))
        print(f"    worst {worst[1]} {worst[0]:+.4f}    best {best[1]} {best[0]:+.4f}")

    if deltas_all:
        print("=" * 78)
        m = statistics.mean(deltas_all)
        print(f"POOLED over {len(deltas_all)} entities: mean Δ VUS-PR = {m:+.4f}"
              + (f", sd {statistics.stdev(deltas_all):.4f}" if len(deltas_all) > 1 else ""))
        print(f"  improved: {sum(x > 0 for x in deltas_all)}/{len(deltas_all)}")
        # The honest MDE for this benchmark is 0.11-0.15 VUS-PR at n=31 (the ledger's 0.087 is
        # refuted); a per-entity pooled delta far below that is a null however clean the curve.
        verdict = ("BELOW the detection floor — training to convergence does not change what "
                   "gets reported" if abs(m) < 0.05 else
                   "LARGE ENOUGH TO MATTER — the truncation was costing (or buying) real "
                   "detection; every arm in a table must get the same budget")
        print(f"  verdict: {verdict}")
        print("  NOTE: single seed, paired per entity. Treat as a go/no-go on the budget "
              "question, not as a publishable effect size.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
