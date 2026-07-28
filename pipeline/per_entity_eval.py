"""
=============================================================================
  per_entity_eval.py — PER-ENTITY metric breakdown from the detect cache.
=============================================================================
detect.py reports POOLED metrics (all entities concatenated, one threshold).
For multi-entity datasets (e.g. smd:all = 28 machines, smap:all = 54 channels)
the per-entity protocol — evaluate each entity on its own score scale, then
average — is the standard one and the headline numbers for the paper. The
pooling artefact otherwise depresses every metric (a single threshold across
heterogeneous entities).

This stage REPLAYS the cached RAW per-channel scores (no GPU, no forward pass):
re-applies the pipeline finalize (impulse + channel aggregation) exactly like
detect, then computes, PER ENTITY:
  * threshold-free (no-PA) : AUROC, AUPRC, best-F1, VUS-ROC, VUS-PR, PATE-F1
  * point-adjusted (ref.)  : PA-AUROC, PA-AUPRC, PA-best-F1
  * operating point        : precision/recall/F1/FPR/event-F1/affiliation-F1/
                             detection-delay, at a PER-ENTITY train-quantile
                             threshold (q = cfg.threshold.q)
  * threshold strategies   : anomaly_ratio-F1, POT-F1
  * channel attribution    : channel-localization (P@1/P@3/oracle-F1, IPS) and
                             per-channel detection (macro/weighted/joint),
                             only when the dataset surfaces y_channel.
Aggregates (mean/median/std/min/max over entities) + a per-entity table are
written under `<reports>/<run>/<norm_agg>/per_entity/`. A pooled sanity-check
(AUROC/AUPRC) is printed; it must match detect's report.json.

Single entry point: `python pipeline/per_entity_eval.py` (reads the active cfg /
DATASET_* env exactly like detect.py). Requires detect.py to have run first
(reuses its score cache); if the cache is absent it no-ops gracefully.
"""
from __future__ import annotations

# repo root on sys.path so this pipeline/ script can import the shared
# libs (config / utils / detect / metrics_core) that live at the project root.
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent.parent))

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score, f1_score, precision_score, recall_score, roc_auc_score,
)

from config import load_config
from utils import best_checkpoint, resolve_path, run_dir_for
from detect import (
    _detect_cache_path, _load_score_cache, _finalize_entities, _fit_thresholds_per_channel,
)
from metrics_core import (
    POT_LEVEL_BY_DATASET, threshold_free_metrics, adjust_scores, best_threshold_f1,
    event_metrics, affiliation_metrics, detection_delay_metrics, _pot_threshold,
    channel_localization_at_k, ips, per_channel_metrics, aggregate_macro_weighted,
    joint_micro_metrics,
)

# Scalar metrics aggregated across entities (order = column order in the CSV).
TIMESTAMP_KEYS = [
    "n", "n_pos", "base_rate",
    "auroc", "auprc", "best_f1", "vus_roc", "vus_pr", "pate_f1",
    "pa_auroc", "pa_auprc", "pa_best_f1",
    "precision", "recall", "f1", "fpr", "event_f1", "affiliation_f1", "detection_delay",
    "anomaly_ratio_f1", "pot_f1",
]
# Channel attribution — emitted only when the dataset is multivariate AND surfaces
# y_channel. At C=1 these are constants (chan_p_at_1 = chan_oracle_f1 = ips = 1.0
# for any model) or duplicates of the timestamp metrics, so the columns are dropped
# from the table entirely rather than filled with a misleading 1.000.
CHANNEL_KEYS = [
    "chan_p_at_1", "chan_p_at_3", "chan_oracle_f1", "ips",
    "pc_macro_f1", "pc_weighted_f1", "pc_macro_auroc", "pc_weighted_auprc", "pc_joint_f1",
]
SCALAR_KEYS = TIMESTAMP_KEYS + CHANNEL_KEYS


def _safe(d: dict, k: str) -> float:
    v = d.get(k, float("nan"))
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _has_channel_attribution(yc: np.ndarray | None) -> bool:
    """Channel attribution is only meaningful with >= 2 channels (see metrics_core)."""
    return yc is not None and np.asarray(yc).ndim == 2 and np.asarray(yc).shape[1] >= 2


def _entity_row(eid: str, y: np.ndarray, s: np.ndarray, train_s: np.ndarray,
                yc: np.ndarray | None, train_entity: dict[str, Any],
                q: float, buffer: int, pot_level: float) -> dict[str, Any]:
    r: dict[str, Any] = {k: float("nan") for k in SCALAR_KEYS}
    r["entity"] = eid
    r["n"] = int(y.size)
    r["n_pos"] = int(y.sum())
    r["base_rate"] = float(y.mean()) if y.size else float("nan")

    has_both = 0 < int(y.sum()) < int(y.size)
    if has_both:
        tf = threshold_free_metrics(y, s, point_adjust=False, buffer=buffer)
        for k_out, k_in in [("auroc", "auroc"), ("auprc", "auprc"), ("best_f1", "best_f1"),
                            ("vus_roc", "vus_roc"), ("vus_pr", "vus_pr"), ("pate_f1", "pate_f1")]:
            r[k_out] = _safe(tf, k_in)
        sa = adjust_scores(s, y)
        r["pa_auroc"] = float(roc_auc_score(y, sa))
        r["pa_auprc"] = float(average_precision_score(y, sa))
        _, r["pa_best_f1"] = best_threshold_f1(y, sa)

        # operating point — PER-ENTITY train-quantile threshold
        thr = float(np.quantile(train_s, q))
        pred = (s > thr).astype(np.int64)
        tn = int(((y == 0) & (pred == 0)).sum()); fp = int(((y == 0) & (pred == 1)).sum())
        r["precision"] = precision_score(y, pred, zero_division=0)
        r["recall"] = recall_score(y, pred, zero_division=0)
        r["f1"] = f1_score(y, pred, zero_division=0)
        r["fpr"] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        r["event_f1"] = _safe(event_metrics(y, pred), "event_f1")
        r["affiliation_f1"] = _safe(affiliation_metrics(y, pred), "affiliation_f1")
        r["detection_delay"] = _safe(detection_delay_metrics(y, pred), "detection_delay_mean")

        # anomaly_ratio (predicted positive fraction == base rate)
        thr_ar = float(np.quantile(s, 1.0 - r["base_rate"]))
        r["anomaly_ratio_f1"] = f1_score(y, (s > thr_ar).astype(np.int64), zero_division=0)
        # POT / SPOT (label-free)
        pthr = _pot_threshold(train_s, s, q=1e-3, level=pot_level)
        r["pot_f1"] = (f1_score(y, (s > pthr).astype(np.int64), zero_division=0)
                       if pthr is not None else float("nan"))

    # channel attribution — only when per-channel GT exists AND C >= 2
    if _has_channel_attribution(yc):
        cs = np.asarray(train_entity["__test_channel_scores__"], np.float64)
        loc = channel_localization_at_k(cs, yc, ks=(1, 3))
        r["chan_p_at_1"] = _safe(loc, "channel_precision_at_1")
        r["chan_p_at_3"] = _safe(loc, "channel_precision_at_3")
        r["chan_oracle_f1"] = _safe(loc, "channel_precision_at_oracle")
        r["ips"] = _safe(ips(cs, yc), "ips")
        thr_ch = _fit_thresholds_per_channel([train_entity], q)
        preds_ch = (cs > thr_ch[None, :]).astype(np.int64)
        agg = aggregate_macro_weighted(per_channel_metrics(yc, cs, preds_ch, thr_ch))
        jm = joint_micro_metrics(yc, cs, preds_ch)
        r["pc_macro_f1"] = _safe(agg, "macro_f1")
        r["pc_weighted_f1"] = _safe(agg, "weighted_f1")
        r["pc_macro_auroc"] = _safe(agg, "macro_auroc")
        r["pc_weighted_auprc"] = _safe(agg, "weighted_auprc")
        r["pc_joint_f1"] = _safe(jm, "joint_f1")
    return r


def _aggregate(rows: list[dict[str, Any]], keys: list[str] = SCALAR_KEYS) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for k in keys:
        vals = np.array([r[k] for r in rows if isinstance(r.get(k), (int, float)) and r[k] == r[k]], dtype=float)
        if vals.size == 0:
            continue
        out[k] = {"mean": float(vals.mean()), "median": float(np.median(vals)),
                  "std": float(vals.std()), "min": float(vals.min()),
                  "max": float(vals.max()), "n": int(vals.size)}
    return out


def per_entity_eval(cfg=None) -> dict[str, Any] | None:
    cfg = cfg or load_config()
    q = float(cfg.threshold.q)
    buffer = int(cfg.evaluation.paper_metrics_tolerance)
    pot_level = float(POT_LEVEL_BY_DATASET.get(cfg.dataset.name.lower(), 0.02))

    # ── Locate the detect score cache (same recipe as detect.py) ──────────
    try:
        stage1_ckpt = best_checkpoint(cfg, "stage1")
        try:
            stage2_ckpt = best_checkpoint(cfg, "stage2")
        except Exception:
            stage2_ckpt = None
        cache_path = _detect_cache_path(stage1_ckpt, stage2_ckpt)
    except Exception as exc:
        print(f"[per_entity_eval] could not resolve cache path: {exc}")
        return None
    if not cache_path.exists():
        print(f"[per_entity_eval] no detect cache at {cache_path} — run detect.py first; skipping.")
        return None

    print(f"[per_entity_eval] cache: {cache_path}")
    train_e, test_e = _load_score_cache(cache_path)
    _finalize_entities(train_e, cfg)
    _finalize_entities(test_e, cfg)
    tr = {e["entity_id"]: e for e in train_e}

    rows: list[dict[str, Any]] = []
    any_channel_attribution = False
    for e in test_e:
        eid = e["entity_id"]
        y = np.asarray(e["labels"], np.int64)
        s = np.asarray(e["overall_scores"], np.float64)
        te = tr.get(eid)
        if te is None:
            print(f"[per_entity_eval] {eid}: no matching train entity, skipping.")
            continue
        train_s = np.asarray(te["overall_scores"], np.float64)
        yc = np.asarray(e["y_channel"]).astype(np.int64) if e.get("y_channel") is not None else None
        if _has_channel_attribution(yc):
            any_channel_attribution = True
            te["__test_channel_scores__"] = np.asarray(e["channel_scores"], np.float64)
        r = _entity_row(eid, y, s, train_s, yc, te, q, buffer, pot_level)
        rows.append(r)
        print(f"  {eid}: AUROC={r['auroc']:.3f} bestF1={r['best_f1']:.3f} VUS-ROC={r['vus_roc']:.3f} "
              f"PATE-F1={r['pate_f1']:.3f} f1@q={r['f1']:.3f}", flush=True)

    if not rows:
        print("[per_entity_eval] no entities evaluated; skipping.")
        return None

    # Univariate (or no y_channel): drop the channel-attribution columns instead of
    # writing a column of 1.000 / NaN that reads as a result.
    cols = TIMESTAMP_KEYS + (CHANNEL_KEYS if any_channel_attribution else [])
    if not any_channel_attribution:
        for r in rows:
            for k in CHANNEL_KEYS:
                r.pop(k, None)
        print("[per_entity_eval] channel attribution not scored "
              "(univariate or no per-channel ground truth).")

    summary = _aggregate(rows, cols)
    # pooled sanity-check (must match detect's report.json AUROC/AUPRC)
    ys = np.concatenate([np.asarray(e["labels"], np.int64) for e in test_e])
    ss = np.concatenate([np.asarray(e["overall_scores"], np.float64) for e in test_e])
    pooled = {"auroc": float(roc_auc_score(ys, ss)) if 0 < ys.sum() < ys.size else float("nan"),
              "auprc": float(average_precision_score(ys, ss)) if 0 < ys.sum() < ys.size else float("nan")}

    report = {
        "dataset_name": cfg.dataset.name, "entity_id": cfg.dataset.entity_id,
        "seed": cfg.seed, "n_entities": len(rows),
        "threshold_q": q, "buffer": buffer, "pot_level": pot_level,
        "normalization": cfg.scoring.normalization, "aggregation": cfg.scoring.aggregation,
        "operating_point": "per-entity train-quantile @ threshold_q",
        "channel_attribution_scored": bool(any_channel_attribution),
        "summary": summary, "pooled_sanity": pooled, "per_entity": rows,
    }

    # ── Output dir (same layout as detect.py) ─────────────────────────────
    base = resolve_path(cfg.paths.reports) / run_dir_for(cfg, "stage1").relative_to(
        resolve_path(cfg.paths.runs) / "stage1")
    out_dir = base / f"{cfg.scoring.normalization}_{cfg.scoring.aggregation}" / "per_entity"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    with (out_dir / "per_entity_table.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["entity"] + cols)
        for r in rows:
            w.writerow([r["entity"]] + [r.get(k, "") for k in cols])

    def _ms(k):
        return summary.get(k, {}).get("mean", float("nan")), summary.get(k, {}).get("median", float("nan"))
    print(f"\n[per_entity_eval] {len(rows)} entities — PER-ENTITY mean / median:")
    for k in ["auroc", "auprc", "best_f1", "vus_roc", "vus_pr", "pate_f1",
              "precision", "f1", "affiliation_f1", "chan_p_at_1", "pc_macro_f1"]:
        if k in summary:
            m, md = _ms(k); print(f"    {k:>16}: mean={m:.4f}  median={md:.4f}")
    print(f"[per_entity_eval] POOLED sanity: AUROC={pooled['auroc']:.4f} AUPRC={pooled['auprc']:.4f} "
          f"(must match detect report.json)")
    print(f"[per_entity_eval] report: {(out_dir / 'report.json').resolve()}")
    return report


def main() -> None:
    per_entity_eval()


if __name__ == "__main__":
    main()
