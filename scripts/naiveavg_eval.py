"""
P0-B — NaiveAvg: post-hoc FedAvg of the converged cb_only priors, native readout.

The matched-fusion matrix compares four ways to combine the SAME converged
per-client `federated_cb_only` priors (identical (cluster,seed,entity) keys):
    NaiveAvg   — average the prior WEIGHTS, score the single averaged prior   (this file)
    AlignedAvg — permutation-aligned weight average                           (needs the aligner)
    Ensemble   — mixture (soft-min of per-token NLL)                          (mixture_eval.py)
    FedDF      — distill the ensemble into one student                        (feddf_distill.py)

NaiveAvg is the missing cell: unlike `federated_shared` (ITERATIVE FedAvg that
re-syncs every round, so the bodies never diverge), this averages priors that
converged INDEPENDENTLY from their own init — the regime where mode-barrier /
parameter-misalignment can actually bite. If NaiveAvg << Ensemble, weight-space
averaging uniquely destroys signal the function-space keeps → build the aligner.
If NaiveAvg ~ Ensemble, the weight space is not the culprit → cause is (b).

Averaging is n_k-weighted (mirrors pipeline/federated.py:_fedavg_shared). Scored
through the EXACT detect.py machinery via mixture_eval._score_entity, so numbers
sit on the same axis as local / cb_only / centralized / mixture / feddf.

Usage:
  CUDA_VISIBLE_DEVICES=1 python scripts/naiveavg_eval.py \
      --dataset wsd_fed --clusters all --seeds 0,1,2,3
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

from data import make_dataloaders  # noqa: E402
from federated import resolve_clients  # noqa: E402
from mixture_eval import (  # noqa: E402
    CONVERGED, MixtureStage2, _build_cfg, _load_pool, _score_entity,
)

OUT_ROOT = REPO / "artifacts" / "fed_eval" / "naiveavg"


@torch.no_grad()
def _client_nwindows(cfg, entities, device) -> list[int]:
    """n_k = #train windows per client (the _fedavg_shared aggregation weight)."""
    ns = []
    for e in entities:
        ce = copy.deepcopy(cfg); ce.dataset.entity_id = e
        loader = make_dataloaders(ce, stage="stage2").train_loader
        n = sum(int(b["inputs"].shape[0]) for b in loader)
        ns.append(max(1, n))
    return ns


@torch.no_grad()
def _fedavg_priors(priors, weights):
    """n_k-weighted average of every floating-point parameter/buffer. All priors
    share architecture (same cfg) → identical state_dict keys. Non-float entries
    (none expected) are copied from priors[0]."""
    w = torch.tensor(weights, dtype=torch.float64)
    w = (w / w.sum()).tolist()
    avg = copy.deepcopy(priors[0])
    ref = avg.state_dict()
    sds = [p.state_dict() for p in priors]
    new = {}
    for k in ref:
        if ref[k].is_floating_point():
            acc = None
            for wi, sd in zip(w, sds):
                term = sd[k].float() * wi
                acc = term if acc is None else acc + term
            new[k] = acc.to(ref[k].dtype)
        else:
            new[k] = ref[k]
    avg.load_state_dict(new)
    avg.eval()
    return avg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="wsd_fed")
    ap.add_argument("--clusters", default="all")
    ap.add_argument("--seeds", default="0,1,2,3")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = _build_cfg(args.dataset, args.batch)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if args.clusters == "all":
        clusters = sorted(p.name for p in (CONVERGED / args.dataset).iterdir() if p.is_dir())
    else:
        clusters = [c.strip() for c in args.clusters.split(",") if c.strip()]

    print(f"[naiveavg] dataset={args.dataset} clusters={clusters} seeds={seeds} device={device}")
    all_records = []
    for cluster in clusters:
        entities = resolve_clients(cfg, None, cluster)
        for seed in seeds:
            shared_stage1, pool = _load_pool(args.dataset, cluster, seed, entities, cfg, device)
            if shared_stage1 is None:
                print(f"[naiveavg] SKIP {cluster} seed{seed}: cb_only run missing/incomplete")
                continue
            priors, have = pool
            nk = _client_nwindows(cfg, have, device)
            avg_prior = _fedavg_priors(priors, nk)
            wrap = MixtureStage2(shared_stage1, [avg_prior], "mean_nll")  # 1 prior ⇒ its own score
            print(f"[naiveavg] {cluster} seed{seed}: avg over {len(priors)} priors (n_k={nk})")
            out_dir = OUT_ROOT / args.dataset / cluster / f"seed{seed}"
            for e in have:
                rep = _score_entity(shared_stage1, wrap, cfg, e)
                d = out_dir / e; d.mkdir(parents=True, exist_ok=True)
                (d / "report.json").write_text(json.dumps(rep, indent=2))
                rec = {"_arm": "naiveavg", "_cluster": cluster, "_seed": seed, "_entity": e,
                       **{k: float(rep[k]) for k in ("auroc", "auprc", "vus_pr", "pate_f1")
                          if isinstance(rep.get(k), (int, float)) and np.isfinite(rep.get(k))}}
                all_records.append(rec)
                print(f"  [naiveavg {cluster} s{seed}] {e}: "
                      f"vus_pr={rep.get('vus_pr', float('nan')):.3f} "
                      f"auprc={rep.get('auprc', float('nan')):.3f} "
                      f"pate_f1={rep.get('pate_f1', float('nan')):.3f}")
            del shared_stage1, priors, avg_prior, wrap
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rec_path = OUT_ROOT / f"records_{args.dataset}.jsonl"
    with rec_path.open("w") as fh:
        for r in all_records:
            fh.write(json.dumps(r) + "\n")
    print(f"[naiveavg] wrote {len(all_records)} records -> {rec_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
