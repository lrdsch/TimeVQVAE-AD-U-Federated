#!/usr/bin/env python3
"""FLOOR — the calibration baseline runner.

Scores every client of a federated dataset with a closed-form / zero-parameter
head and pushes the result through the EXACT `detect` path the neural arms use,
so the numbers are comparable to `artifacts/converged_all` and
`artifacts/converge60` by construction.

The deciding head is `ma_c` (centred moving-average residual, k=10) — the
calibration baseline pre-registered at federated_method.tex:200,259,281.

    s_t = ( x_t - MA_k(x)_t )**2         MA_k = detect._moving_average_paper

Zero fitted parameters, therefore ARM-INVARIANT: local == centralized == every
federated mode, bit-identically. Rows are stamped `_mode="arm_invariant"` and no
federation question can be asked of them.

Canonical accumulation geometry (documentation/FLOOR_BASELINE.md §3):
  * windows enumerated by data.SlidingWindowDataset(records, W, eval_stride)
  * the WHOLE window contributes (no `skip` offset) so `coverage` stays uniform
    under rolling_aggregation="sum", which never divides by coverage
  * _finalize_entities(train) runs BEFORE _fit_threshold_paper (the quantile
    fallback reads overall_scores)

Outputs (append-only, resumable):
    artifacts/floor/records_<dataset>.jsonl
    artifacts/floor/<dataset>__<arm>.json      {meta, summary, records}

Examples:
    python scripts/floor_eval.py --selftest
    python scripts/floor_eval.py --dataset wsd_fed --jobs 3
    python scripts/floor_eval.py --dataset wsd_fed --heads ma_c,diff1,random --k 10
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pipeline"))

METRIC_KEYS = ["vus_pr", "auprc", "auroc", "pate_f1", "affiliation_f1", "f1"]
DEFAULT_K = 10                      # pre-registered: 10 min at dt_sec=60


# ─────────────────────────────────────────────────────────────────────────────
# Heads.  Each is score-only: no fit, no state, no data from other clients.
# ─────────────────────────────────────────────────────────────────────────────
def head_ma_c(x: np.ndarray, k: int, D) -> np.ndarray:
    """Centred moving-average residual — THE calibration baseline."""
    return (x - D._moving_average_paper(x, k)) ** 2


def head_diff1(x: np.ndarray, k: int, D) -> np.ndarray:
    """First difference squared. NB: identical up to a positive scale to ma_c
    at k=3, so it yields the same metrics (all metrics are scale-invariant)."""
    s = np.empty_like(x)
    s[1:] = (x[1:] - x[:-1]) ** 2
    s[0] = s[1]
    return s


def head_random(x: np.ndarray, k: int, D) -> np.ndarray:
    """Bottom anchor. Seeded on length so a rerun is bit-identical."""
    return np.random.default_rng(abs(hash(("floor_random", len(x)))) % (2**32)).random(len(x))


HEADS = {"ma_c": head_ma_c, "diff1": head_diff1, "random": head_random}
ZERO_PARAM = set(HEADS)             # every head here fits nothing


def build_cfg(dataset: str, impulse: bool):
    from config import Config, apply_dataset_overrides, apply_env_overrides
    cfg = Config()
    cfg.dataset.name = dataset
    with contextlib.redirect_stdout(io.StringIO()):
        apply_dataset_overrides(cfg)      # wsd_fed -> metrics_tolerance 14
        apply_env_overrides(cfg)
    cfg.evaluation.save_plots = False
    cfg.evaluation.save_scores = False
    cfg.scoring.use_impulse_term = bool(impulse)
    return cfg


def arm_tag(head: str, k: int, impulse: bool) -> str:
    base = f"floor_{head}" + (f"_k{k}" if head == "ma_c" else "")
    return base + ("" if impulse else "_noimpulse")


# ─────────────────────────────────────────────────────────────────────────────
def score_entity(job: dict) -> list[dict]:
    """One client, every requested head. Returns flat record rows."""
    import data as D_data
    import detect as D

    cfg = build_cfg(job["dataset"], job["impulse"])
    W = cfg.dataset.window_length
    stride = D._resolve_eval_stride(cfg)

    c = copy.deepcopy(cfg)
    c.dataset.entity_id = job["entity"]
    with contextlib.redirect_stdout(io.StringIO()):
        tr, va, te = D_data.load_scaled_records(c)

    rows = []
    for head in job["heads"]:
        t0 = time.time()
        fn = HEADS[head]
        packs, nwin = {}, 0
        for tag, recs in (("train", tr), ("test", te)):
            ents = D._init_entity(recs)
            series = [np.asarray(r.X[:, 0], dtype=np.float64) for r in recs]
            scored = [fn(x, job["k"], D) for x in series]
            for wi in D_data.SlidingWindowDataset(recs, W, stride).indices:
                e = ents[wi.record_index]
                # WHOLE window contributes -> coverage stays uniform
                e["channel_sum"][wi.start:wi.stop, 0] += scored[wi.record_index][wi.start:wi.stop]
                e["coverage"][wi.start:wi.stop] += 1.0
                if tag == "test":
                    nwin += 1
            for e in ents:
                e["channel_scores"] = D._assemble_rolling(
                    e["channel_sum"], e["coverage"], cfg.scoring.rolling_aggregation)
            packs[tag] = ents

        # train FIRST: the quantile fallback reads overall_scores
        D._finalize_entities(packs["train"], cfg)
        scores, labels = D._finalize_entities(packs["test"], cfg)
        thr = D._fit_threshold_paper(packs["train"], cfg)
        preds = (scores > thr).astype(np.int64)
        rep = D._detection_metrics(labels, preds, scores, cfg)

        row = {
            "_arm": arm_tag(head, job["k"], job["impulse"]),
            "_head": head,
            "_mode": "arm_invariant" if head in ZERO_PARAM else "local",
            "_knobs": {"k": job["k"]} if head == "ma_c" else {},
            "_cluster": job["cluster"],
            "_entity": job["entity"],
            "_seed": 0,
            "_impulse": bool(job["impulse"]),
            "_threshold_rule": "quantile_fallback",
            "_threshold": float(thr),
            "_unit": job["unit"],
            "_n_windows": int(nwin),
            "_fit_stride": None,          # nothing is fitted
            "_eval_stride": int(stride),
            "_window": int(W),
            "_tolerance": int(cfg.evaluation.paper_metrics_tolerance),
            "_secs": round(time.time() - t0, 2),
        }
        # non-finite metrics are OMITTED, never NaN (federated_eval.py:1547-1551)
        for k_ in METRIC_KEYS:
            v = rep.get(k_)
            if isinstance(v, (int, float)) and np.isfinite(v):
                row[k_] = float(v)
        rows.append(row)
        print(f"    {row['_arm']:22s} {job['entity']:10s} "
              f"vus_pr={row.get('vus_pr', float('nan')):.4f} ({row['_secs']:.1f}s)", flush=True)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
def selftest() -> int:
    """Asserted invariants (documentation/FLOOR_BASELINE.md §7 step 1)."""
    import detect as D
    ok = True

    # 1. ma_c at k=3 IS the first difference, scaled by 1/4 -- detect's
    #    edge-clipped kernel averages 2 points at window=3.
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    d = np.abs(head_ma_c(x, 3, D)[1:] - 0.25 * head_diff1(x, 3, D)[1:]).max()
    print(f"  [1] ma_c(k=3) == 0.25*diff1     max|delta| = {d:.2e}  "
          f"{'PASS' if d < 1e-12 else 'FAIL'}")
    ok &= d < 1e-12

    # 2. the head is a pure function of its own series (no hidden state)
    a, b = head_ma_c(x, 10, D), head_ma_c(x.copy(), 10, D)
    same = np.array_equal(a, b)
    print(f"  [2] head is deterministic       {'PASS' if same else 'FAIL'}")
    ok &= same

    # 3. the kernel really averages k points at even k
    j, k = 250, 10
    manual = x[j - k // 2: j + k // 2].mean()
    d3 = abs(D._moving_average_paper(x, k)[j] - manual)
    print(f"  [3] MA_10 averages 10 points    max|delta| = {d3:.2e}  "
          f"{'PASS' if d3 < 1e-12 else 'FAIL'}")
    ok &= d3 < 1e-12

    print(f"\n  selftest: {'ALL PASS' if ok else 'FAILURES'}")
    return 0 if ok else 1


# ─────────────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="wsd_fed")
    ap.add_argument("--clusters", default="all", help="'all' or comma list, e.g. c0,c2")
    ap.add_argument("--heads", default="ma_c", help=f"comma list of {sorted(HEADS)}")
    ap.add_argument("--k", type=int, default=DEFAULT_K,
                    help=f"ma_c window; PRE-REGISTERED at {DEFAULT_K}, do not tune")
    ap.add_argument("--impulse", choices=["on", "off"], default="on")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--out-dir", default="artifacts/floor")
    ap.add_argument("--force", action="store_true", help="rescore rows already on disk")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    heads = [h.strip() for h in args.heads.split(",") if h.strip()]
    bad = [h for h in heads if h not in HEADS]
    if bad:
        raise SystemExit(f"unknown head(s) {bad}; available: {sorted(HEADS)}")
    impulse = args.impulse == "on"

    # Be a good neighbour: the 16-core CPU is the wall for the neural sweeps.
    live = subprocess.run(["pgrep", "-f", "federated_eval.py"],
                          capture_output=True, text=True).stdout.strip()
    jobs = args.jobs
    if live and jobs > 4:
        jobs = 4
        print(f"[floor] federated_eval.py is running -> capping --jobs at {jobs}")
    os.nice(19)

    from federated import resolve_clients
    cfg = build_cfg(args.dataset, impulse)
    meta_path = REPO / "data" / "raw" / args.dataset / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    clusters = (sorted(meta.get("clusters", {})) if args.clusters == "all"
                else [c.strip() for c in args.clusters.split(",")])
    # ucr_split: the honest unit is the cluster, not the 5 shards inside it
    unit = "cluster" if args.dataset.startswith("ucr_split") else "entity"

    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rec_path = out_dir / f"records_{args.dataset}.jsonl"
    done = set()
    if rec_path.exists() and not args.force:
        for line in rec_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["_arm"], r["_entity"]))

    jobs_list = []
    for cl in clusters:
        for e in resolve_clients(cfg, None, cl):
            todo = [h for h in heads if (arm_tag(h, args.k, impulse), e) not in done]
            if todo:
                jobs_list.append({"dataset": args.dataset, "cluster": cl, "entity": e,
                                  "heads": todo, "k": args.k, "impulse": impulse,
                                  "unit": unit})

    print(f"[floor] {args.dataset}: {len(clusters)} clusters, {len(jobs_list)} clients to score, "
          f"heads={heads}, k={args.k}, impulse={args.impulse}, jobs={jobs}, nice 19")
    print(f"[floor] W={cfg.dataset.window_length} eval_stride={_stride(cfg)} "
          f"tolerance={cfg.evaluation.paper_metrics_tolerance} q={cfg.threshold.q}")
    if not jobs_list:
        print("[floor] nothing to do (use --force to rescore)")
    t0 = time.time()
    rows: list[dict] = []
    if jobs_list:
        with Pool(max(1, jobs)) as p:
            for sub in p.imap_unordered(score_entity, jobs_list):
                rows.extend(sub)
                with rec_path.open("a") as fh:      # append as we go: crash-safe
                    for r in sub:
                        fh.write(json.dumps(r) + "\n")
    print(f"[floor] scored {len(rows)} rows in {time.time() - t0:.0f}s -> {rec_path}")

    # ── summary JSON per arm, over EVERY row on disk ────────────────────────
    allrows = [json.loads(l) for l in rec_path.read_text().splitlines() if l.strip()]
    for arm in sorted({r["_arm"] for r in allrows}):
        sub = [r for r in allrows if r["_arm"] == arm]
        summary = {}
        for k_ in METRIC_KEYS:
            v = [r[k_] for r in sub if k_ in r]
            if v:
                summary[k_] = {"mean": float(np.mean(v)), "median": float(np.median(v)),
                               "std": float(np.std(v)), "worst": float(np.min(v)),
                               "n": len(v)}
        payload = {
            "meta": {"dataset": args.dataset, "arm": arm, "unit": unit,
                     "window": cfg.dataset.window_length,
                     "eval_stride": _stride(cfg),
                     "tolerance": cfg.evaluation.paper_metrics_tolerance,
                     "threshold_q": cfg.threshold.q,
                     "threshold_rule": "quantile_fallback",
                     "impulse": impulse, "n_clients": len(sub),
                     "fitted_parameters": 0,
                     "note": ("zero fitted parameters -> arm-invariant: local == centralized "
                              "== every federated mode, bit-identically. No federation "
                              "question can be asked of this head.")},
            "summary": summary,
            "records": sub,
        }
        p_json = out_dir / f"{args.dataset}__{arm}.json"
        p_json.write_text(json.dumps(payload, indent=2))
        s = summary.get("vus_pr", {})
        if s:
            print(f"  {arm:22s} n={s['n']:4d}  vus_pr mean={s['mean']:.4f} "
                  f"median={s['median']:.4f}  -> {p_json.relative_to(REPO)}")
    return 0


def _stride(cfg) -> int:
    import detect as D
    return D._resolve_eval_stride(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
