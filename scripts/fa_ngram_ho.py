"""
E3 — HIGHER-ORDER FEDERATED COUNT PRIOR (federated analytics, exactly-mergeable).

Question: how far can an *exactly-mergeable* count prior over the shared token
grid climb toward the deep federated / centralized MaskGIT prior, once we go past
the unigram (E2) to time-adjacent higher-order n-grams?

We do NOT federate any weights. We federate additive counts, which merge EXACTLY
(sum of per-client count tensors == counts on the pooled data). Concretely, over
the shared K=64 codebook and the (C, F, W) token grid produced by the shared
tokenizer (the `federated_cb_only` shared stage1), we build, WITHIN each frequency
row f and treating the W time-columns as a stationary Markov chain:

  * unigram  N1[c,f,k]            — row-marginal token occupancy      (control)
  * bigram   N2[c,f,i,j]          — time-adjacent transition i -> j    (t -> t+1)
  * trigram  N3[c,f,i,j,l]        — i -> j -> l                        (t-1,t -> t+1)

All three are ADDITIVE across clients → the pooled (federated) prior is exact.
Smoothing (additive alpha) + hierarchical linear-interpolation backoff
(trigram -> bigram -> unigram) is applied ONCE to the MERGED counts — the correct
way that preserves exact mergeability. We also MEASURE where naive federation
would break it: KL between (a) averaging per-client LOCALLY-smoothed conditionals
[prob-space FedAvg of a smoothed prior] and (b) the exact pooled-smoothed
conditional — nonzero, because Laplace smoothing is not additive.

The per-token score is the n-gram NLL assigned to each grid position given its
time-predecessor(s):
  variant "unigram"        : nll[.,c,f,w] = -log p1(tok_w)
  variant "bigram"         : w=0 -> unigram ; w>=1 -> -log p2i(tok_w | tok_{w-1})
  variant "bigram_trigram" : w=0 -> unigram ; w=1 -> bigram ; w>=2 -> -log p3i(tok_w | tok_{w-2}, tok_{w-1})

token_scores has shape (B,C,F,W) [per_rate=False] or (1,B,C,F,W) [per_rate=True]
(a count prior has no MaskGIT masking-rate axis → a single "rate", n_τ=1). Every-
thing downstream — rolling assembly, paper per-τ per-(C,F) threshold, VUS-PR /
AUPRC / PATE — is the EXACT detect.py machinery, reusing mixture_eval._score_entity,
so the numbers are on the same axis as the converged local / cb_only / centralized.

Usage (smoke):
  CUDA_VISIBLE_DEVICES=1 python scripts/fa_ngram_ho.py \
      --dataset wsd_fed --clusters c3 --seeds 0 --variants unigram,bigram,bigram_trigram
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

# Tested infra (read scripts/mixture_eval.py for the contracts).
from mixture_eval import _build_cfg, _load_pool, _score_entity, _Out, CONVERGED  # noqa: E402
from federated import resolve_clients  # noqa: E402
from data import make_dataloaders  # noqa: E402
from stage2 import _flatten_token_indices  # noqa: E402

OUT_ROOT = REPO / "artifacts" / "fed_eval" / "fa_ngram_ho"


# ─── Scorer: quacks like a Stage2System, scores with the pooled n-gram counts ──

class NgramStage2:
    """Stage2System look-alike. Tokenises with the SHARED stage1, then assigns
    every grid token the -log of its smoothed conditional given its time-
    predecessor(s) within its frequency row. Holds precomputed log-prob tables on
    the pooled (federated) counts."""

    def __init__(self, shared_stage1, dims, logp1, logp2i, logp3i, variant):
        self.stage1 = shared_stage1                 # detect asserts .stage1.training is False
        # detect asserts .prior.training is False and only calls .score_batch — a
        # frozen dummy in eval mode satisfies the assert without a real prior.
        self.prior = torch.nn.Identity().eval()
        self.C, self.F, self.W, self.K = dims
        self.logp1 = logp1                          # (C,F,K)
        self.logp2i = logp2i                        # (C,F,K,K)  interpolated bigram log-cond, [c,f,i,j]=log p(j|i)
        self.logp3i = logp3i                        # (C,F,K,K,K) interpolated trigram, [c,f,i,j,l]=log p(l|i,j)
        self.variant = variant

    @torch.no_grad()
    def score_batch(self, batch: dict, per_rate: bool = False) -> _Out:
        _, indices, _ = self.stage1.encode_tokens(batch["inputs"])
        B = indices.shape[0]
        C, F, W, K = self.C, self.F, self.W, self.K
        grid = indices.long().reshape(B, C, F, W)              # (B,C,F,W), W = time axis

        # Unigram baseline for every position (also used at w=0 for higher orders).
        lp1 = self.logp1[None].expand(B, C, F, K)              # (B,C,F,K)
        nll = -torch.gather(lp1, 3, grid)                      # (B,C,F,W)

        if self.variant in ("bigram", "bigram_trigram"):
            prev = grid[..., :-1]; cur = grid[..., 1:]         # (B,C,F,W-1)
            lin2 = prev * K + cur
            lp2 = self.logp2i.reshape(C, F, K * K)[None].expand(B, C, F, K * K)
            nll[..., 1:] = -torch.gather(lp2, 3, lin2)         # w>=1 -> bigram

        if self.variant == "bigram_trigram":
            p2 = grid[..., :-2]; p1 = grid[..., 1:-1]; cur = grid[..., 2:]   # (B,C,F,W-2)
            lin3 = (p2 * K + p1) * K + cur
            lp3 = self.logp3i.reshape(C, F, K ** 3)[None].expand(B, C, F, K ** 3)
            nll[..., 2:] = -torch.gather(lp3, 3, lin3)         # w>=2 -> trigram

        return _Out(nll.unsqueeze(0) if per_rate else nll)


# ─── Pooled count construction (the exactly-mergeable object) ──────────────────

@torch.no_grad()
def _build_pooled_counts(shared_stage1, cfg, have, device, need_trigram):
    """Tokenise every client's train windows with the shared stage1 and build
    per-client + pooled additive count tensors. Returns dims (C,F,W) plus per-
    client N1/N2 (for the mergeability-KL) and pooled N1/N2/(N3)."""
    K = cfg.quantizer.codebook_size
    dims = None
    pc_N1, pc_N2 = [], []            # per-client (for the KL diagnostic)
    pooled_N1 = pooled_N2 = pooled_N3 = None

    for e in have:
        ce = copy.deepcopy(cfg); ce.dataset.entity_id = e
        loader = make_dataloaders(ce, stage="stage2").train_loader
        N1 = N2 = None
        for batch in loader:
            x = batch["inputs"].to(device, non_blocking=True)
            _, idx, latent_spatial = shared_stage1.encode_tokens(x)
            if dims is None:
                C = int(idx.shape[1]); F = int(latent_spatial[0]); W = int(latent_spatial[1])
                dims = (C, F, W)
            C, F, W = dims
            grid = idx.long().reshape(idx.shape[0], C, F, W)         # (B,C,F,W)
            if N1 is None:
                N1 = torch.zeros(C, F, K, dtype=torch.float64, device=device)
                N2 = torch.zeros(C, F, K, K, dtype=torch.float64, device=device)
            for c in range(C):
                for f in range(F):
                    row = grid[:, c, f, :]                            # (B,W)
                    N1[c, f] += torch.bincount(row.reshape(-1), minlength=K).double()
                    if W >= 2:
                        lin = (row[:, :-1] * K + row[:, 1:]).reshape(-1)
                        N2[c, f] += torch.bincount(lin, minlength=K * K).double().reshape(K, K)
                    if need_trigram and W >= 3:
                        p2 = row[:, :-2]; p1 = row[:, 1:-1]; cur = row[:, 2:]
                        lin3 = ((p2 * K + p1) * K + cur).reshape(-1)
                        bc = torch.bincount(lin3, minlength=K ** 3).double().reshape(K, K, K)
                        if pooled_N3 is None:
                            pooled_N3 = torch.zeros(C, F, K, K, K, dtype=torch.float64, device=device)
                        pooled_N3[c, f] += bc
        if N1 is None:                                              # empty client — skip
            continue
        pc_N1.append(N1); pc_N2.append(N2)
        pooled_N1 = N1.clone() if pooled_N1 is None else pooled_N1 + N1
        pooled_N2 = N2.clone() if pooled_N2 is None else pooled_N2 + N2

    return dims, pc_N1, pc_N2, pooled_N1, pooled_N2, pooled_N3


def _build_tables(pooled_N1, pooled_N2, pooled_N3, alpha, lam2, lam3, K, need_trigram):
    """Additive smoothing + hierarchical linear-interpolation backoff on the
    MERGED counts (the exact-mergeable, deploy-once prior). Returns log-prob
    tables logp1 (C,F,K), logp2i (C,F,K,K), logp3i (C,F,K,K,K) or None."""
    # Unigram
    p1 = (pooled_N1 + alpha) / (pooled_N1.sum(-1, keepdim=True) + alpha * K)     # (C,F,K)
    logp1 = p1.clamp_min(1e-30).log().float()

    # Bigram conditional p2(j|i), interpolated with the unigram marginal.
    ctx2 = pooled_N2.sum(-1, keepdim=True)                                       # (C,F,K,1)
    p2 = (pooled_N2 + alpha) / (ctx2 + alpha * K)                                # (C,F,K,K) [c,f,i,j]
    p2i = lam2 * p2 + (1.0 - lam2) * p1[:, :, None, :]                           # back off over predecessor i
    logp2i = p2i.clamp_min(1e-30).log().float()

    logp3i = None
    if need_trigram and pooled_N3 is not None:
        ctx3 = pooled_N3.sum(-1, keepdim=True)                                   # (C,F,K,K,1)
        p3 = (pooled_N3 + alpha) / (ctx3 + alpha * K)                            # (C,F,K,K,K) [c,f,i,j,l]
        # backoff to interpolated bigram p2i(l|j), broadcast over the older predecessor i
        p3i = lam3 * p3 + (1.0 - lam3) * p2i[:, :, None, :, :]                   # p2i indexed [c,f,j,l]
        logp3i = p3i.clamp_min(1e-30).log().float()

    return logp1, logp2i, logp3i


def _mergeability_kl(pc_N2, pooled_N2, alpha, K):
    """Where additive smoothing breaks exact mergeability. Compares:
      * p_pooled(j|i)  : smooth ONCE on merged raw counts (== centralized refit, exact)
      * p_fedavg(j|i)  : average of per-client LOCALLY-smoothed conditionals (naive prob-space FL)
    Returns (mean context-weighted KL(p_fedavg || p_pooled) in nats,
             max abs diff between summing per-client raw counts and pooled_N2 == 0 by construction)."""
    pooled_ctx = pooled_N2.sum(-1, keepdim=True)                                 # (C,F,K,1)
    p_pooled = (pooled_N2 + alpha) / (pooled_ctx + alpha * K)                    # (C,F,K,K)
    p_avg = torch.zeros_like(p_pooled)
    for N2k in pc_N2:
        p_avg += (N2k + alpha) / (N2k.sum(-1, keepdim=True) + alpha * K)
    p_avg /= max(1, len(pc_N2))
    kl = (p_avg * (p_avg.clamp_min(1e-30).log() - p_pooled.clamp_min(1e-30).log())).sum(-1)  # (C,F,K)
    w = pooled_ctx.squeeze(-1)                                                   # weight by context occurrence
    mean_kl = float((kl * w).sum() / w.sum().clamp_min(1.0))
    merged = None
    for N2k in pc_N2:
        merged = N2k.clone() if merged is None else merged + N2k
    max_diff = float((merged - pooled_N2).abs().max()) if merged is not None else 0.0
    return mean_kl, max_diff


# ─── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="wsd_fed")
    ap.add_argument("--clusters", default="all",
                    help="comma list or 'all' (discovered from the converged tree)")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--variants", default="unigram,bigram,bigram_trigram",
                    help="subset of {unigram,bigram,bigram_trigram}")
    ap.add_argument("--alpha", type=float, default=0.1, help="additive (Laplace) smoothing")
    ap.add_argument("--lam2", type=float, default=0.9, help="bigram interpolation weight (rest -> unigram)")
    ap.add_argument("--lam3", type=float, default=0.9, help="trigram interpolation weight (rest -> bigram)")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = _build_cfg(args.dataset, args.batch)
    K = cfg.quantizer.codebook_size
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variants:
        if v not in ("unigram", "bigram", "bigram_trigram"):
            raise ValueError(f"unknown variant {v!r}")
    need_trigram = any(v == "bigram_trigram" for v in variants)

    if args.clusters == "all":
        clusters = sorted(p.name for p in (CONVERGED / args.dataset).iterdir() if p.is_dir())
    else:
        clusters = [c.strip() for c in args.clusters.split(",") if c.strip()]

    print(f"[ngram_ho] dataset={args.dataset} clusters={clusters} seeds={seeds} "
          f"variants={variants} alpha={args.alpha} lam2={args.lam2} lam3={args.lam3} device={device}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rec_path = OUT_ROOT / f"records_{args.dataset}.jsonl"
    rec_path.write_text("")            # fresh start for this dataset; records appended below (crash-resilient)
    n_records = 0

    for cluster in clusters:
        entities = resolve_clients(cfg, None, cluster)
        for seed in seeds:
            shared_stage1, pool = _load_pool(args.dataset, cluster, seed, entities, cfg, device)
            if shared_stage1 is None:
                print(f"[ngram_ho] SKIP {cluster} seed{seed}: cb_only run missing/incomplete")
                continue
            priors, have = pool
            del priors                                   # count prior does not use the deep priors
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[ngram_ho] {cluster} seed{seed}: shared tokenizer over {have}")

            dims, pc_N1, pc_N2, pooled_N1, pooled_N2, pooled_N3 = _build_pooled_counts(
                shared_stage1, cfg, have, device, need_trigram)
            C, F, W = dims
            print(f"[ngram_ho] {cluster} seed{seed}: grid C={C} F={F} W={W} K={K} "
                  f"| pooled train tokens/row≈{int(pooled_N1.sum() / (C * F)):,}")

            # Mergeability diagnostic (bigram level).
            mean_kl, max_diff = _mergeability_kl(pc_N2, pooled_N2, args.alpha, K)
            print(f"[ngram_ho] {cluster} seed{seed}: mergeability — raw-count merge exact "
                  f"(max|Δcount|={max_diff:.1f}); local-smooth-then-avg breaks it: "
                  f"mean KL(fedavg||pooled)={mean_kl:.4e} nats")
            seed_dir = OUT_ROOT / args.dataset / cluster / f"seed{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            (seed_dir / "kl_mergeability.json").write_text(json.dumps(
                {"mean_kl_fedavg_vs_pooled_nats": mean_kl,
                 "max_abs_count_diff_rawmerge_vs_pooled": max_diff,
                 "alpha": args.alpha, "n_clients": len(pc_N2)}, indent=2))

            logp1, logp2i, logp3i = _build_tables(
                pooled_N1, pooled_N2, pooled_N3, args.alpha, args.lam2, args.lam3, K, need_trigram)

            dims4 = (C, F, W, K)
            for variant in variants:
                scorer = NgramStage2(shared_stage1, dims4, logp1, logp2i, logp3i, variant)
                out_dir = seed_dir / variant
                for e in have:
                    rep = _score_entity(shared_stage1, scorer, cfg, e)
                    d = out_dir / e; d.mkdir(parents=True, exist_ok=True)
                    (d / "report.json").write_text(json.dumps(rep, indent=2))
                    rec = {"_arm": f"fa_ngram_ho_{variant}", "_cluster": cluster, "_seed": seed, "_entity": e}
                    for k in ("vus_pr", "auprc", "pate_f1", "auroc"):
                        v = rep.get(k)
                        if isinstance(v, (int, float)) and np.isfinite(v):
                            rec[k] = float(v)
                    with rec_path.open("a") as fh:
                        fh.write(json.dumps(rec) + "\n")
                    n_records += 1
                    print(f"  [{variant} {cluster} s{seed}] {e}: "
                          f"vus_pr={rep.get('vus_pr', float('nan')):.3f} "
                          f"auprc={rep.get('auprc', float('nan')):.3f} "
                          f"pate_f1={rep.get('pate_f1', float('nan')):.3f}")

            del shared_stage1, pooled_N1, pooled_N2, pooled_N3, logp1, logp2i, logp3i, pc_N1, pc_N2
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"[ngram_ho] wrote {n_records} records -> {rec_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
