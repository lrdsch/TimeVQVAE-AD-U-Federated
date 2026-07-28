"""
E6 — ZERO-SHOT ONBOARDING of a NEW client (leave-one-out interoperability).

For every (cluster, seed) we HOLD OUT one client at a time. The held-out client
gets NO local prior training. We score its test data zero-shot using ONLY the
federated statistics of the OTHER clients, on top of the SHARED cb_only tokenizer
(all clients speak the same K-codeword language). Two scorers:

  * onboard_count  : a POOLED per-position count prior (E1-style unigram over the
                     C*F*W token grid, Laplace-smoothed) built from the OTHER
                     clients' train tokens only. This object is EXACTLY mergeable
                     (additive counts) — the true "federated statistic".
  * onboard_mixture: the uniform MixtureStage2 over the OTHER clients' DEEP cb_only
                     priors (soft-min in distribution space), held-out excluded.

Everything downstream (rolling assembly, paper threshold, VUS-PR / AUPRC / PATE)
is the EXACT detect.py machinery via mixture_eval._score_entity, so the held-out
client's zero-shot number sits on the same axis as its OWN converged `local`
report — the value it WOULD have gotten had it trained. The gap is the price (or
free lunch) of onboarding with no local training: the interoperability value of
the shared vocabulary.

Usage (smoke):
  CUDA_VISIBLE_DEVICES=1 python scripts/fa_onboard.py \
      --dataset wsd_fed --clusters c3 --seeds 0

Usage (full):
  CUDA_VISIBLE_DEVICES=1 python scripts/fa_onboard.py \
      --dataset wsd_fed --clusters all --seeds 0,1,2
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pipeline"))

from mixture_eval import (  # noqa: E402
    _build_cfg, _load_pool, _score_entity, MixtureStage2, _Out, CONVERGED,
)
from federated import resolve_clients  # noqa: E402
from stage2 import _flatten_token_indices  # noqa: E402
from data import make_dataloaders  # noqa: E402

OUT_ROOT = REPO / "artifacts" / "fed_eval" / "fa_onboard"


class _EvalStub(nn.Module):
    """Minimal eval-mode module used only to satisfy detect's
    `assert not stage2.prior.training` for the count scorer, which never touches
    a deep prior."""


@torch.no_grad()
def _pooled_position_logp(shared_stage1, cfg, entities, device, K, alpha=1.0):
    """Pooled per-position unigram over the C*F*W token grid, built from the
    train windows of `entities` (the OTHER clients). Laplace(alpha)-smoothed.
    Returns (logp (P, K), P). Additive across clients ⇒ EXACTLY mergeable."""
    counts = None            # (P, K) float64 on device
    P = None
    for e in entities:
        ce = copy.deepcopy(cfg)
        ce.dataset.entity_id = e
        loader = make_dataloaders(ce, stage="stage2").train_loader
        for batch in loader:
            x = batch["inputs"].to(device, non_blocking=True)
            _, idx, _ = shared_stage1.encode_tokens(x)
            tok = _flatten_token_indices(idx).long()          # (B, P)
            if counts is None:
                P = tok.shape[1]
                counts = torch.full((P, K), float(alpha), device=device, dtype=torch.float64)
            # per-position bincount accumulation
            flat = (torch.arange(P, device=device)[None, :] * K + tok).reshape(-1)
            counts.view(-1).scatter_add_(
                0, flat, torch.ones_like(flat, dtype=torch.float64)
            )
    if counts is None:
        raise RuntimeError(f"no train windows for onboarding donors {entities}")
    logp = (counts / counts.sum(dim=1, keepdim=True)).log()   # (P, K)
    return logp, P


class CountPrior3D:
    """Quacks like a Stage2System for detect: shared stage1 tokenizer + a pooled
    per-position count prior. token_scores[b,c,f,w] = -log p(token | position),
    reshaped to the (B, C, F, W) grid detect expects. The count model has a single
    'rate', so per_rate stacks one copy (sum over rates == per_rate=False)."""

    def __init__(self, shared_stage1, logp, C, F, W, stub):
        self.stage1 = shared_stage1     # detect asserts .stage1.training is False
        self.prior = stub               # detect asserts .prior.training is False
        self.logp = logp                # (P, K)
        self.C, self.F, self.W = C, F, W
        self._pos = torch.arange(logp.shape[0], device=logp.device)[None, :]

    @torch.no_grad()
    def score_batch(self, batch: dict, per_rate: bool = False) -> _Out:
        _, indices, _ = self.stage1.encode_tokens(batch["inputs"])
        tokens = _flatten_token_indices(indices).long()       # (B, P)
        B, P = tokens.shape
        lp = self.logp[self._pos, tokens]                     # (B, P) advanced index
        grid = (-lp).reshape(B, self.C, self.F, self.W).float()
        if per_rate:
            return _Out(grid.unsqueeze(0))                    # (1, B, C, F, W)
        return _Out(grid)                                     # (B, C, F, W)


@torch.no_grad()
def _grid_dims(shared_stage1, cfg, entity, device):
    """One tokenizer forward to discover (C, F, W) of the latent grid."""
    ce = copy.deepcopy(cfg)
    ce.dataset.entity_id = entity
    x = next(iter(make_dataloaders(ce, stage="stage2").train_loader))["inputs"][:1].to(device)
    _, idx, latent_spatial = shared_stage1.encode_tokens(x)
    C = int(idx.shape[1])
    F_ = int(latent_spatial[0])
    W = int(latent_spatial[1])
    return C, F_, W


def _local_vus(dataset, cluster, seed, entity):
    """The held-out client's OWN converged local VUS-PR (the value it would get if
    it had trained). None if the report is missing."""
    rp = CONVERGED / dataset / cluster / f"seed{seed}" / "local" / entity / "report.json"
    if not rp.exists():
        return None
    try:
        return float(json.load(open(rp)).get("vus_pr"))
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="wsd_fed")
    ap.add_argument("--clusters", default="all",
                    help="comma list or 'all' (discovered from converged tree)")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--variants", default="onboard_count,onboard_mixture")
    ap.add_argument("--alpha", type=float, default=1.0, help="Laplace smoothing for onboard_count")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit-holdout", type=int, default=0,
                    help="if >0, only hold out the first N clients (smoke only; 0 = all, full LOO)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = _build_cfg(args.dataset, args.batch)
    K = cfg.quantizer.codebook_size
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    if args.clusters == "all":
        clusters = sorted(p.name for p in (CONVERGED / args.dataset).iterdir() if p.is_dir())
    else:
        clusters = [c.strip() for c in args.clusters.split(",") if c.strip()]

    print(f"[onboard] dataset={args.dataset} clusters={clusters} seeds={seeds} "
          f"variants={variants} alpha={args.alpha} K={K} device={device}")

    all_records = []
    for cluster in clusters:
        entities = resolve_clients(cfg, None, cluster)
        for seed in seeds:
            shared_stage1, pool = _load_pool(args.dataset, cluster, seed, entities, cfg, device)
            if shared_stage1 is None:
                print(f"[onboard] SKIP {cluster} seed{seed}: cb_only run missing/incomplete")
                continue
            priors, have = pool
            print(f"[onboard] {cluster} seed{seed}: {len(have)} clients {have}")
            if len(have) < 2:
                print(f"[onboard] SKIP {cluster} seed{seed}: need >=2 clients for LOO")
                continue

            C, F_, W = _grid_dims(shared_stage1, cfg, have[0], device)
            stub = _EvalStub().to(device).eval()

            # Leave-one-out: hold out each client, onboard it from the others.
            holdouts = have if args.limit_holdout <= 0 else have[: args.limit_holdout]
            for hi, held in enumerate(holdouts):
                others = [e for e in have if e != held]
                other_priors = [priors[j] for j in range(len(have)) if have[j] != held]
                loc = _local_vus(args.dataset, cluster, seed, held)

                for variant in variants:
                    if variant == "onboard_count":
                        logp, _P = _pooled_position_logp(
                            shared_stage1, cfg, others, device, K, alpha=args.alpha
                        )
                        scorer = CountPrior3D(shared_stage1, logp, C, F_, W, stub)
                    elif variant == "onboard_mixture":
                        scorer = MixtureStage2(shared_stage1, other_priors, "mixture")
                    else:
                        raise ValueError(f"unknown variant {variant!r}")

                    rep = _score_entity(shared_stage1, scorer, cfg, held)
                    rep["onboard_held_out"] = held
                    rep["onboard_donors"] = others
                    rep["local_vus_pr"] = loc

                    out_dir = OUT_ROOT / args.dataset / cluster / f"seed{seed}" / variant / held
                    out_dir.mkdir(parents=True, exist_ok=True)
                    (out_dir / "report.json").write_text(json.dumps(rep, indent=2))

                    rec = {"_arm": f"fa_onboard_{variant}", "_cluster": cluster,
                           "_seed": seed, "_entity": held,
                           "local_vus_pr": loc,
                           **{k: float(rep[k]) for k in ("auroc", "auprc", "vus_pr", "pate_f1")
                              if isinstance(rep.get(k), (int, float)) and np.isfinite(rep.get(k))}}
                    all_records.append(rec)

                    lv = f"{loc:.3f}" if isinstance(loc, float) else "n/a"
                    print(f"  [{variant} {cluster} s{seed}] {held}: "
                          f"vus_pr={rep.get('vus_pr', float('nan')):.3f} "
                          f"auprc={rep.get('auprc', float('nan')):.3f} "
                          f"pate_f1={rep.get('pate_f1', float('nan')):.3f} "
                          f"(own_local_vus_pr={lv})")

            del shared_stage1, priors
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rec_path = OUT_ROOT / f"records_{args.dataset}.jsonl"
    with rec_path.open("w") as fh:
        for r in all_records:
            fh.write(json.dumps(r) + "\n")
    print(f"[onboard] wrote {len(all_records)} records -> {rec_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
