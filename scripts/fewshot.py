"""Few-shot SOURCE SELECTION — can a handful of labeled anomalies pick a neighbor
model that beats local, and how much of the oracle ceiling does it recover?

Protocol (per target client j, deployment-realistic temporal split):
  * reveal the first k anomaly EVENTS of j's test series (+ everything up to the
    end of the k-th event) as a labeled CALIBRATION region; the rest is EVAL.
  * for each candidate source model i, score j's test; measure detection quality
    on CALIBRATION; select the best source i*.
  * report i*'s quality on the held-out EVAL region, against:
       local_j (j's own model on EVAL) — the incumbent,
       oracle  (best source on EVAL, chosen with EVAL labels) — upper bound,
       mean-foreign (average source on EVAL) — blind transfer.

Two phases:
  --score : GPU pass, recompute test_scores for every (source,target) and cache
            to artifacts/fed_eval/cross/scorevecs_<ds>_s<seed>.npz.
  (default): CPU pass, load the cache and run the few-shot selection for k in K.

    CUDA_VISIBLE_DEVICES=1 python scripts/fewshot.py --dataset wsd_fed --seed 0 --score
    python scripts/fewshot.py --dataset wsd_fed --seed 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

from utils import resolve_path
import detect as D

CROSS = resolve_path("artifacts/fed_eval/cross")


def cache_path(ds, seed):
    return CROSS / f"scorevecs_{ds}_s{seed}.npz"


# ---------------------------------------------------------------- GPU: score all
def score_all(ds, seed):
    import torch
    from cross_eval import build_cfg, scan_arm, score_target, _example_inputs
    from stage1 import load_stage1
    from stage2 import load_stage2
    cfg = build_cfg(ds, seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reg = scan_arm(ds, "local", seed)
    ents = sorted(reg)
    print(f"[fewshot] scoring {len(ents)}x{len(ents)} cells for {ds} seed{seed}", flush=True)
    store = {}
    for si, src in enumerate(ents):
        info = reg[src]
        ex = _example_inputs(cfg, src, device)
        s1 = load_stage1(info["s1"], cfg, ex, device=device)
        s2 = load_stage2(info["s2"], cfg, stage1_ckpt=info["s1"],
                         stage1_example_inputs=ex, device=device)
        for tgt in ents:
            ts, tl, _ = score_target(s1, s2, cfg, tgt, adaptive=False)
            store[f"s|{src}|{tgt}"] = ts.astype(np.float32)
            if si == 0:
                store[f"y|{tgt}"] = tl.astype(np.int8)
                store[f"cl|{tgt}"] = np.array([reg[tgt]["cluster"]])
        del s1, s2
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[fewshot] source {si+1}/{len(ents)} ({src}) done", flush=True)
    CROSS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path(ds, seed), **store)
    print(f"[fewshot] cached -> {cache_path(ds, seed)}", flush=True)


# ---------------------------------------------------------------- CPU: few-shot
def segments(y):
    """[(start,end_exclusive)] anomaly runs."""
    y = np.asarray(y).astype(bool)
    out, i, n = [], 0, len(y)
    while i < n:
        if y[i]:
            j = i
            while j < n and y[j]:
                j += 1
            out.append((i, j)); i = j
        else:
            i += 1
    return out


def vus_pr(labels, scores, tol):
    if labels.sum() == 0 or labels.sum() == len(labels):
        return np.nan
    try:
        return float(D._vus_metrics(labels.astype(np.int64), scores.astype(np.float64), tol)["vus_pr"])
    except Exception:
        from sklearn.metrics import average_precision_score
        return float(average_precision_score(labels, scores))


def run_select(ds, seed, tol, ks):
    z = np.load(cache_path(ds, seed), allow_pickle=True)
    ents = sorted({k.split("|")[2] for k in z.files if k.startswith("s|")})
    labels = {e: z[f"y|{e}"].astype(np.int64) for e in ents}
    cluster = {e: str(z[f"cl|{e}"][0]) for e in ents}
    S = {(a, b): z[f"s|{a}|{b}"] for a in ents for b in ents}  # (src,tgt)->scores

    print(f"\n{'='*72}\n  FEW-SHOT SOURCE SELECTION — {ds} seed{seed} · metric=vus_pr(tol={tol})")
    print(f"  reveal first-k anomaly events of the target, pick best source on that,")
    print(f"  report on the held-out rest. n = targets with enough events.\n{'='*72}")
    print(f"\n{'k (labeled events)':>18s} | {'few-shot':>9s} {'local':>9s} {'oracle':>9s} {'mean-fgn':>9s} | "
          f"{'Δ few-local':>12s} {'win%':>5s} {'recovered':>9s}  n")
    for k in ks:
        rows = []
        for j in ents:
            y = labels[j]; ev = segments(y)
            if len(ev) < k + 1:            # need k for calib + ≥1 for eval
                continue
            cal_end = ev[k - 1][1]
            ycal, yev = y[:cal_end], y[cal_end:]
            if ycal.sum() == 0 or yev.sum() == 0:
                continue
            # select on calibration
            cal_scores = {i: S[(i, j)][:cal_end] for i in ents}
            sel = max(ents, key=lambda i: (vus_pr(ycal, cal_scores[i], tol) if not np.isnan(
                vus_pr(ycal, cal_scores[i], tol)) else -1))
            # report on eval
            ev_v = {i: vus_pr(yev, S[(i, j)][cal_end:], tol) for i in ents}
            fewshot = ev_v[sel]
            local = ev_v[j]
            oracle = np.nanmax([ev_v[i] for i in ents if i != j])
            meanf = np.nanmean([ev_v[i] for i in ents if i != j])
            if np.isnan(fewshot) or np.isnan(local):
                continue
            rows.append((fewshot, local, oracle, meanf))
        if not rows:
            print(f"{k:>18d} |  (no targets with enough events)")
            continue
        a = np.array(rows)
        fs, lo, orc, mf = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        d = fs - lo
        gap = orc - lo
        recovered = np.nanmean(d[gap > 1e-6]) / np.nanmean(gap[gap > 1e-6]) if (gap > 1e-6).any() else np.nan
        print(f"{k:>18d} | {fs.mean():>9.3f} {lo.mean():>9.3f} {orc.mean():>9.3f} {mf.mean():>9.3f} | "
              f"{d.mean():>+12.3f} {(d>0).mean()*100:>4.0f}% {recovered*100:>8.0f}%  {len(rows)}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--score", action="store_true", help="GPU phase: recompute+cache score vectors")
    ap.add_argument("--tol", type=int, default=None,
                    help="VUS-PR buffer; default = cfg.evaluation.paper_metrics_tolerance for "
                         "--dataset (wsd_fed 14, toy_fed_uni 128). NOT comparable across datasets.")
    ap.add_argument("--ks", type=str, default="1,2,3,5")
    args = ap.parse_args()
    if args.score:
        score_all(args.dataset, args.seed)
        return
    ks = [int(x) for x in args.ks.split(",")]
    tol = args.tol
    if tol is None:
        # The VUS/PATE buffer is a per-dataset property; hardcoding 14 (wsd_fed's value) made the
        # number silently non-comparable to every report.json on non-wsd datasets.
        from config import Config, apply_dataset_overrides
        _c = Config(); _c.dataset.name = args.dataset
        apply_dataset_overrides(_c)
        tol = int(_c.evaluation.paper_metrics_tolerance)
    run_select(args.dataset, args.seed, tol, ks)


if __name__ == "__main__":
    main()
