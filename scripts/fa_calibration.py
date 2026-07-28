"""
E4 — FEDERATED THRESHOLD / SCORE CALIBRATION (pure Federated Analytics).

The deployed anomaly detector is the `federated_cb_only` arm: one DEEP MaskGIT
prior per client, all on the SHARED (suff-stat merged) codebook. Each client's
test window is scored under its OWN prior — no weight fusion, no mixture. That is
exactly the cb_only detector.

The VUS-PR / AUROC / AUPRC axis is threshold-FREE, so federating the *decision
threshold* cannot move it. What a federated threshold CAN move is the operational,
threshold-DEPENDENT trio (F1 / event-F1 / precision / recall / PATE-binary): the
cold-start / cross-silo comparability story. Instead of every client fitting its
OWN paper threshold on its OWN train split (which makes thresholds incomparable
across silos and fragile when a client has few/odd train windows), all clients in
a cluster agree on ONE threshold = a quantile of the POOLED per-timestep train-
score distribution. Pooling per-timestep train scores is additive-in-spirit FA:
no labels, no gradients, just each client's train-score histogram shared once.

Variants (same heavy scoring pass, only the threshold differs):
  * local_threshold          — control == cb_only: each entity keeps its OWN paper
                               threshold (detect._fit_threshold_paper). Reproduces
                               the converged cb_only report.
  * federated_quantile_0.99  — threshold = quantile_0.99(POOLED cluster train scores)
  * federated_quantile_0.995 — threshold = quantile_0.995(POOLED cluster train scores)

Everything downstream (rolling assembly, impulse, channel aggregation, VUS/PATE)
is the EXACT detect.py machinery, so numbers are directly comparable to the
converged local / cb_only / centralized reports. Threshold-free metrics
(auroc/auprc/vus_pr/pate_f1) are identical across variants by construction; the
deltas live in f1 / event_f1 / precision / recall / pate (binary).

Usage:
  CUDA_VISIBLE_DEVICES=1 python scripts/fa_calibration.py \
      --dataset wsd_fed --clusters all --seeds 0,1,2 --quantiles 0.99,0.995
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pipeline"))
sys.path.insert(0, str(REPO / "scripts"))

# Reuse the tested mixture_eval infra (sets up REPO/pipeline on sys.path on import).
from mixture_eval import _build_cfg, MixtureStage2, CONVERGED  # noqa: E402
from federated import resolve_clients  # noqa: E402
from data import make_dataloaders  # noqa: E402
from stage2 import load_stage2  # noqa: E402
import detect as D  # noqa: E402

OUT_ROOT = REPO / "artifacts" / "fed_eval" / "fa_calibration"


def _load_cb_only_pairs(dataset, cluster, seed, entities, cfg, device):
    """Load, PER ENTITY, its OWN cb_only (stage1, prior). Unlike mixture_eval's
    `_load_pool` — which keeps a SINGLE shared stage1 because the mixture needs
    every expert on one token grid — the deployed cb_only detector scores each
    client with its OWN encoder (only the CODEBOOK is suff-stat merged; encoder /
    decoder are trained locally). To reproduce cb_only's report (the calibration
    control) exactly, each entity MUST be tokenised by its own stage1. Returns
    ({entity: (stage1, prior)}, have) or (None, None) if the run is missing."""
    cb = CONVERGED / dataset / cluster / f"seed{seed}" / "federated_cb_only"
    if not cb.exists():
        return None, None
    s1_ckpts = {e: cb / e / "stage1.ckpt" for e in entities}
    s2_ckpts = {e: cb / e / "stage2.ckpt" for e in entities}
    have = [e for e in entities if s1_ckpts[e].exists() and s2_ckpts[e].exists()]
    if len(have) < 2:
        return None, None
    # Example (1, C, W) for lazy-layer materialisation. Window shape is identical
    # across clients (univariate, fixed W), so have[0]'s example works for all.
    c0 = copy.deepcopy(cfg); c0.dataset.entity_id = have[0]
    example = next(iter(make_dataloaders(c0, stage="eval").train_loader))["inputs"][:1].cpu()
    pairs = {}
    for e in have:
        ce = copy.deepcopy(cfg); ce.dataset.entity_id = e
        s2 = load_stage2(s2_ckpts[e], ce, stage1_ckpt=s1_ckpts[e],
                         stage1_example_inputs=example, device=device)
        s2.stage1.eval(); s2.prior.eval()
        pairs[e] = (s2.stage1, s2.prior)
    return pairs, have

# Metrics that DEPEND on the threshold (change with the calibration statistic).
# Everything else in the report (auroc/auprc/vus_pr/pate_f1/best_f1) is threshold-free.
THRESH_DEP_KEYS = (
    "precision", "recall", "f1", "fpr", "pate",
    "event_precision", "event_recall", "event_f1",
    "detection_delay_mean",
)


def _score_one(stage1, prior, cfg, entity):
    """Heavy GPU pass for ONE entity, scored under its OWN cb_only (stage1, prior)
    pair — exactly the deployed cb_only detector. Returns the per-timestep TRAIN
    scores (the pooled-threshold input), the per-timestep TEST scores + labels,
    the entity's own paper threshold (the local control), and the per-entity cfg."""
    c = copy.deepcopy(cfg)
    c.dataset.entity_id = entity
    c.scoring.weight_s_local = 0.0                 # pure-prior deployed score
    c.scoring.weight_s_prior = 1.0
    c.evaluation.save_plots = False
    c.evaluation.save_scores = False
    # Single-prior scorer: MixtureStage2 with one prior + mean_nll collapses to
    # that prior's per-token NLL (stack of 1 -> mean over the singleton axis).
    scorer = MixtureStage2(stage1, [prior], "mean_nll")
    data = make_dataloaders(c, stage="eval")
    train_entities = D._compute_entities_raw(
        stage1, scorer, c, data.train_records, record_per_rate=True)
    test_entities = D._compute_entities_raw(
        stage1, scorer, c, data.test_records)
    train_scores, _ = D._finalize_entities(train_entities, c)      # (T_train,) per-timestep
    test_scores, test_labels = D._finalize_entities(test_entities, c)
    local_thr = float(D._fit_threshold_paper(train_entities, c))
    return train_scores, test_scores, test_labels, local_thr, c


def _metrics_at(test_labels, test_scores, threshold, cfg, entity, extra=None):
    """detect's metric suite at a given threshold. Same keys detect writes."""
    preds = (test_scores > float(threshold)).astype(np.int64)
    rep = {"dataset_name": cfg.dataset.name, "entity_id": entity, "threshold": float(threshold)}
    if extra:
        rep.update(extra)
    rep.update(D._detection_metrics(test_labels, preds, test_scores, cfg))
    rep.update(D._event_metrics(test_labels, preds))
    return rep


def _rec(arm, cluster, seed, entity, rep):
    r = {"_arm": arm, "_cluster": cluster, "_seed": seed, "_entity": entity}
    for k in ("vus_pr", "auprc", "pate_f1", "auroc", "f1", "event_f1",
              "precision", "recall", "pate", "threshold"):
        v = rep.get(k)
        if isinstance(v, (int, float)) and np.isfinite(v):
            r[k] = float(v)
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="wsd_fed")
    ap.add_argument("--clusters", default="all",
                    help="comma list or 'all' (discovered from converged tree)")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--quantiles", default="0.99,0.995",
                    help="pooled-train-score quantiles for the federated threshold")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = _build_cfg(args.dataset, args.batch)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    quantiles = [float(q) for q in args.quantiles.split(",") if q.strip()]

    if args.clusters == "all":
        clusters = sorted(p.name for p in (CONVERGED / args.dataset).iterdir() if p.is_dir())
    else:
        clusters = [c.strip() for c in args.clusters.split(",") if c.strip()]

    def _qtag(q):  # 0.99 -> "099", 0.995 -> "0995"
        return str(q).replace(".", "").rstrip() or "q"

    variant_names = ["local_threshold"] + [f"federated_quantile_{_qtag(q)}" for q in quantiles]
    print(f"[fa_calibration] dataset={args.dataset} clusters={clusters} seeds={seeds} "
          f"quantiles={quantiles} variants={variant_names} device={device}")

    all_records = []
    for cluster in clusters:
        entities = resolve_clients(cfg, None, cluster)
        for seed in seeds:
            pairs, have = _load_cb_only_pairs(args.dataset, cluster, seed, entities, cfg, device)
            if pairs is None:
                print(f"[fa_calibration] SKIP {cluster} seed{seed}: cb_only run missing/incomplete")
                continue
            print(f"[fa_calibration] {cluster} seed{seed}: {len(have)} (stage1,prior) pairs over {have}")

            # ── Heavy pass: score every client ONCE with its OWN pair, cache. ──
            per_entity = {}          # entity -> (train_scores, test_scores, test_labels, local_thr, c)
            for e in have:
                s1_e, prior_e = pairs[e]
                per_entity[e] = _score_one(s1_e, prior_e, cfg, e)

            # ── Federated threshold = quantile of the POOLED cluster train scores. ──
            pooled_train = np.concatenate([per_entity[e][0] for e in have])
            fed_thr = {q: float(np.quantile(pooled_train, q)) for q in quantiles}
            print(f"[fa_calibration]   pooled train scores: n={pooled_train.size} "
                  f"fed_thr={{ " + ", ".join(f'{q}:{fed_thr[q]:.4g}' for q in quantiles) + " }}")

            # Per variant: local control keeps each entity's own paper threshold;
            # federated variants apply the pooled-quantile threshold to everyone.
            for vi, variant in enumerate(variant_names):
                arm = f"fa_calibration_{variant}"
                out_dir = OUT_ROOT / args.dataset / cluster / f"seed{seed}" / variant
                for e in have:
                    _tr, test_scores, test_labels, local_thr, c = per_entity[e]
                    if variant == "local_threshold":
                        thr = local_thr
                        extra = {"threshold_kind": "local_paper"}
                    else:
                        q = quantiles[vi - 1]
                        thr = fed_thr[q]
                        extra = {"threshold_kind": f"federated_quantile_{q}",
                                 "quantile": q, "pooled_train_n": int(pooled_train.size),
                                 "local_paper_threshold": local_thr}
                    rep = _metrics_at(test_labels, test_scores, thr, c, e, extra=extra)
                    d = out_dir / e
                    d.mkdir(parents=True, exist_ok=True)
                    (d / "report.json").write_text(json.dumps(rep, indent=2))
                    all_records.append(_rec(arm, cluster, seed, e, rep))
                    print(f"  [{variant} {cluster} s{seed}] {e}: "
                          f"thr={thr:.4g} "
                          f"vus_pr={rep.get('vus_pr', float('nan')):.3f} "
                          f"auprc={rep.get('auprc', float('nan')):.3f} "
                          f"f1={rep.get('f1', float('nan')):.3f} "
                          f"event_f1={rep.get('event_f1', float('nan')):.3f} "
                          f"pate_f1={rep.get('pate_f1', float('nan')):.3f}")

            del pairs, per_entity
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rec_path = OUT_ROOT / f"records_{args.dataset}.jsonl"
    with rec_path.open("w") as fh:
        for r in all_records:
            fh.write(json.dumps(r) + "\n")
    print(f"[fa_calibration] wrote {len(all_records)} records -> {rec_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
