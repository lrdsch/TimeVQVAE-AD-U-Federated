"""
#1 — Mixture-of-priors at inference (distribution-space collaboration).

We already trained, in the `federated_cb_only` arm, one deep MaskGIT prior per
client on a SHARED (suff-stat merged) codebook — so all clients speak the same
token language. This script does NOT average their weights. Instead, at
inference, every test window is scored under ALL K priors and the per-token
scores are combined in distribution space:

  * "mixture"  : soft-min  -logsumexp_j(-nll_j) + log K   (OR: normal if SOME expert says normal)
  * "best"     : hard-min  min_j nll_j                    (oracle router upper bound)
  * "mean_nll" : mean_j nll_j                             (AND: normal only if all agree)

Everything downstream (rolling assembly, paper threshold, VUS-PR / AUPRC / PATE)
is the EXACT detect.py machinery — so these numbers are on the real axis and
directly comparable to the converged `local` / `cb_only` / `centralized` reports.

Decision this experiment gates:
  * mixture ≳ centralized  → the prize is regime coverage → build #3 (FedDF) to
    compress the mixture into one deployable model.
  * mixture ≈ cb_only      → prediction-space fusion has no juice → the prize is
    representational → put budget on #2 (small-τ FedSGD).

Usage:
  CUDA_VISIBLE_DEVICES=1 python scripts/mixture_eval.py \
      --dataset toy_fed_uni --clusters all --seeds 0,1,2,3 \
      --combines mixture,best,mean_nll
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pipeline"))

from config import Config, apply_dataset_overrides, apply_env_overrides  # noqa: E402
from data import make_dataloaders  # noqa: E402
from stage1 import load_stage1  # noqa: E402
from stage2 import load_stage2, _flatten_token_indices  # noqa: E402
import detect as D  # noqa: E402
from federated import resolve_clients  # noqa: E402

# Checkpoint root for the whole FA family (all 14 scripts/fa_*.py import this constant).
#
# The historical default below WAS DELETED in the 2026-07-23 purge, so it now resolves to a
# missing directory and _load_pool raises. That is DELIBERATE, not an oversight: the FA suite
# assumes a tokenizer shared across clients, but `federated_cb_only` federates only the
# codebook while _load_pool tokenizes every client with have[0]'s encoder (measured
# cross-client token agreement 0.0000). Re-pointing without fixing that would spend 14
# experiments x 3 seeds on an ill-posed comparison.
#
# Override deliberately and per-invocation — an env var so all 14 scripts inherit it without
# 14 separate flags:
#     TVQ_CONVERGED_ROOT=artifacts/converge60/ckpt python scripts/fa_calibration.py ...
# (converge60/ckpt has the identical <ds>/<cluster>/seed<N>/<arm>/<entity>/ leaf shape.)
CONVERGED = Path(os.environ.get(
    "TVQ_CONVERGED_ROOT", str(REPO / "artifacts" / "fed_eval" / "converged")))
OUT_ROOT = REPO / "artifacts" / "fed_eval" / "mixture"


class _Out:
    __slots__ = ("token_scores",)

    def __init__(self, token_scores):
        self.token_scores = token_scores


class MixtureStage2:
    """Quacks like a Stage2System for detect._compute_entities_raw, but scores a
    window under a POOL of priors and combines their per-token NLL in
    distribution space. Tokenises ONCE with the shared stage1 so every expert
    sees the identical token grid."""

    def __init__(self, shared_stage1, priors, combine: str, logn=None, logfreqs=None):
        self.stage1 = shared_stage1          # detect asserts .stage1.training is False
        self.priors = priors                 # list[MaskGITPrior3DPos], all eval()
        self.prior = priors[0]               # detect asserts .prior.training is False
        self.combine = combine
        # Density-weighting inputs (needed only for nk / target_gate / ctx_gate):
        self.logn = logn                     # (Kc,) log train-window count per client
        self.logfreqs = logfreqs             # list[Kc] of (K,) log token-occupancy per client

    @torch.no_grad()
    def score_batch(self, batch: dict, per_rate: bool = False) -> _Out:
        _, indices, _ = self.stage1.encode_tokens(batch["inputs"])
        tokens = _flatten_token_indices(indices).long()   # (B, L)
        per_prior = []
        for p in self.priors:
            s = p.score_tokens_per_rate(tokens) if per_rate else p.score_tokens(tokens)
            per_prior.append(s)
        stack = torch.stack(per_prior, dim=0)            # (Kc, [n_τ,] B, C, F, W) — per-token NLL
        Kc, nd = stack.shape[0], stack.ndim
        batch_axis = 2 if per_rate else 1                 # where B lives in `stack`

        # Simple (unweighted) combiners.
        if self.combine == "best":
            return _Out(stack.amin(dim=0))               # oracle-router upper bound
        if self.combine == "mean_nll":
            return _Out(stack.mean(dim=0))               # AND (all experts agree)

        # Weighted mixture:  nll_mix = -logsumexp_k(lw - nll_k) + logsumexp_k(lw),
        # where lw is the (normalised implicitly) per-client log-weight. lw shapes:
        #   mixture      : 0                         (uniform 1/Kc)
        #   nk           : log n_k                   (data-volume weighting)
        #   ctx_gate     : log n_k + log q_k(window) (density-weighted by the CONTEXT marginal)
        #   target_gate  : log n_k - nll_k           (BROKEN foil: gates on target likelihood)
        def _perclient(vec):
            return vec.reshape([Kc] + [1] * (nd - 1))

        if self.combine == "mixture":
            lw = _perclient(torch.zeros(Kc, device=stack.device))
        elif self.combine == "nk":
            lw = _perclient(self.logn)
        elif self.combine == "ctx_gate":
            logq = torch.stack([lf[tokens].sum(dim=1) for lf in self.logfreqs], dim=0)  # (Kc, B)
            raw = self.logn[:, None] + logq                                            # (Kc, B)
            sh = [Kc] + [1] * (nd - 1); sh[batch_axis] = raw.shape[1]
            lw = raw.reshape(sh)
        elif self.combine == "target_gate":
            lw = _perclient(self.logn) - stack           # full-shape (gates per token)
        else:
            raise ValueError(f"unknown combine {self.combine!r}")

        lwf = lw.expand_as(stack)
        combined = -torch.logsumexp(lwf - stack, dim=0) + torch.logsumexp(lwf, dim=0)
        return _Out(combined)


def _build_cfg(dataset: str, batch: int) -> Config:
    cfg = Config()
    cfg.dataset.name = dataset
    cfg.dataset.batch_size_stage1 = batch
    cfg.dataset.batch_size_stage2 = batch
    apply_dataset_overrides(cfg)
    apply_env_overrides(cfg)
    return cfg


def _score_entity(stage1, mixture, cfg: Config, entity: str) -> dict:
    """detect()'s cold-path core for ONE entity with a custom stage2. Returns the
    report dict with the same metric keys detect writes (auroc/auprc/vus_pr/pate_f1/…)."""
    c = copy.deepcopy(cfg)
    c.dataset.entity_id = entity
    c.scoring.weight_s_local = 0.0                 # pure-prior deployed score
    c.scoring.weight_s_prior = 1.0
    c.evaluation.save_plots = False
    c.evaluation.save_scores = False
    data = make_dataloaders(c, stage="eval")
    train_entities = D._compute_entities_raw(stage1, mixture, c, data.train_records, record_per_rate=True)
    test_entities = D._compute_entities_raw(stage1, mixture, c, data.test_records)
    _train_scores, _ = D._finalize_entities(train_entities, c)
    test_scores, test_labels = D._finalize_entities(test_entities, c)
    threshold = D._fit_threshold_paper(train_entities, c)
    preds = (test_scores > threshold).astype(np.int64)
    rep = {"dataset_name": c.dataset.name, "entity_id": entity, "threshold": float(threshold)}
    rep.update(D._detection_metrics(test_labels, preds, test_scores, c))
    rep.update(D._event_metrics(test_labels, preds))
    return rep


def _load_pool(dataset: str, cluster: str, seed: int, entities: list[str], cfg: Config, device):
    """Load the shared stage1 + every client's cb_only prior for (cluster, seed).
    Returns (shared_stage1, [priors]) or (None, None) if the run is missing."""
    # Fail LOUDLY when the checkpoint root itself is gone. Returning (None, None) here made
    # every caller print SKIP and exit 0 with a 0-byte records file — indistinguishable
    # downstream from "not yet run" (cross_reduce.py:38-40 reports empty and absent the same
    # way). The root HAS been gone since the 2026-07-23 purge deleted artifacts/fed_eval/
    # converged/, so every fa_*.py run since then has been a silent no-op.
    if not CONVERGED.is_dir():
        raise SystemExit(
            f"[mixture_eval] checkpoint root does not exist: {CONVERGED}\n"
            f"  It was deleted in the 2026-07-23 purge. Live checkpoints now live under\n"
            f"  artifacts/converge60/ckpt/<ds>/<cluster>/seed<N>/<arm>/ (same leaf shape).\n"
            f"  Re-point deliberately via TVQ_CONVERGED_ROOT=..., NOT by editing the constant:\n"
            f"  the FA suite is SHELVED because federated_cb_only shares only the codebook\n"
            f"  while _load_pool (below) tokenizes every client with have[0]'s encoder\n"
            f"  (cross-client token agreement measured at 0.0000). Re-pointing without\n"
            f"  first fixing that would burn 14 experiments on an ill-posed comparison."
        )
    cb = CONVERGED / dataset / cluster / f"seed{seed}" / "federated_cb_only"
    if not cb.exists():
        return None, None
    s1_ckpts = {e: cb / e / "stage1.ckpt" for e in entities}
    s2_ckpts = {e: cb / e / "stage2.ckpt" for e in entities}
    have = [e for e in entities if s1_ckpts[e].exists() and s2_ckpts[e].exists()]
    if len(have) < 2:
        return None, None

    # Example (1, C, W) for materialize, from the first client's eval loader.
    c0 = copy.deepcopy(cfg); c0.dataset.entity_id = have[0]
    example = next(iter(make_dataloaders(c0, stage="eval").train_loader))["inputs"][:1].cpu()

    shared_stage1 = load_stage1(s1_ckpts[have[0]], cfg, example, device=device)
    shared_stage1.eval()
    priors = []
    for e in have:
        ce = copy.deepcopy(cfg); ce.dataset.entity_id = e
        s2 = load_stage2(s2_ckpts[e], ce, stage1_ckpt=s1_ckpts[e],
                         stage1_example_inputs=example, device=device)
        s2.prior.eval()
        priors.append(s2.prior)
    return shared_stage1, (priors, have)


@torch.no_grad()
def _client_token_stats(shared_stage1, cfg: Config, entities: list[str], device, K: int):
    """Per client: log train-window count (log n_k) and Laplace-smoothed log token
    occupancy (log freq_k over the K codewords). The anomaly-independent context
    marginal used by ctx_gate — computed from each client's OWN train windows,
    tokenized with the shared codebook. No gradients, no labels."""
    logn, logfreqs = [], []
    for e in entities:
        ce = copy.deepcopy(cfg); ce.dataset.entity_id = e
        loader = make_dataloaders(ce, stage="stage2").train_loader
        counts = torch.zeros(K, device=device); nwin = 0
        for batch in loader:
            x = batch["inputs"].to(device, non_blocking=True)
            _, idx, _ = shared_stage1.encode_tokens(x)
            t = _flatten_token_indices(idx).long().reshape(-1)
            counts += torch.bincount(t, minlength=K).float()
            nwin += x.shape[0]
        freq = (counts + 1.0) / (counts.sum() + K)            # Laplace
        logfreqs.append(freq.log())
        logn.append(float(max(1, nwin)))
    return torch.tensor(logn, device=device).log(), logfreqs   # (Kc,), list[(K,)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="toy_fed_uni")
    ap.add_argument("--clusters", default="all", help="comma list or 'all' (discovered from converged tree)")
    ap.add_argument("--seeds", default="0,1,2,3")
    ap.add_argument("--combines", default="mixture,best,mean_nll")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = _build_cfg(args.dataset, args.batch)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    combines = [c.strip() for c in args.combines.split(",") if c.strip()]

    if args.clusters == "all":
        ds_root = CONVERGED / args.dataset
        if not ds_root.is_dir():                      # .iterdir() on a missing dir throws deep
            raise SystemExit(f"[mixture_eval] no such dataset root: {ds_root}")
        clusters = sorted(p.name for p in ds_root.iterdir() if p.is_dir())
    else:
        clusters = [c.strip() for c in args.clusters.split(",") if c.strip()]

    print(f"[mixture] dataset={args.dataset} clusters={clusters} seeds={seeds} combines={combines} device={device}")
    all_records = []
    for cluster in clusters:
        entities = resolve_clients(cfg, None, cluster)
        for seed in seeds:
            shared_stage1, pool = _load_pool(args.dataset, cluster, seed, entities, cfg, device)
            if shared_stage1 is None:
                print(f"[mixture] SKIP {cluster} seed{seed}: cb_only run missing/incomplete")
                continue
            priors, have = pool
            print(f"[mixture] {cluster} seed{seed}: {len(priors)} priors over {have}")
            logn = logfreqs = None
            if any(c in ("nk", "target_gate", "ctx_gate") for c in combines):
                K = cfg.quantizer.codebook_size
                logn, logfreqs = _client_token_stats(shared_stage1, cfg, have, device, K)
            for combine in combines:
                mixture = MixtureStage2(shared_stage1, priors, combine, logn=logn, logfreqs=logfreqs)
                out_dir = OUT_ROOT / args.dataset / cluster / f"seed{seed}" / combine
                for e in have:
                    rep = _score_entity(shared_stage1, mixture, cfg, e)
                    d = out_dir / e; d.mkdir(parents=True, exist_ok=True)
                    (d / "report.json").write_text(json.dumps(rep, indent=2))
                    rec = {"_arm": f"mixture_{combine}", "_cluster": cluster, "_seed": seed, "_entity": e,
                           **{k: float(rep[k]) for k in ("auroc", "auprc", "vus_pr", "pate_f1")
                              if isinstance(rep.get(k), (int, float)) and np.isfinite(rep.get(k))}}
                    all_records.append(rec)
                    print(f"  [{combine} {cluster} s{seed}] {e}: "
                          f"vus_pr={rep.get('vus_pr', float('nan')):.3f} "
                          f"auprc={rep.get('auprc', float('nan')):.3f} "
                          f"pate_f1={rep.get('pate_f1', float('nan')):.3f}")
            del shared_stage1, priors, mixture
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Refuse to truncate the ledger to nothing. The file is opened in "w", so a run that
    # scored zero entities used to OVERWRITE a good records file with 0 bytes — and a
    # 0-byte ledger is indistinguishable from "never run" for every downstream reducer.
    if not all_records:
        raise SystemExit(
            "[mixture_eval] produced 0 records — refusing to write an empty ledger.\n"
            "  Every (cluster, seed) was skipped: the per-arm checkpoints are missing.\n"
            "  Fix the inputs; do not let this overwrite an existing records file."
        )

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rec_path = OUT_ROOT / f"records_{args.dataset}.jsonl"
    with rec_path.open("w") as fh:
        for r in all_records:
            fh.write(json.dumps(r) + "\n")
    print(f"[mixture] wrote {len(all_records)} records -> {rec_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
