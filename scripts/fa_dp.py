# ═══════════ RETRACTED / FROZEN — DO NOT RUN — the scripts/fa_*.py suite, 2026-07-27 ═══════════
# CAUSE      mixture_eval._load_pool loads ONE stage-1 (client have[0]'s) and tokenizes EVERY
#            client with it, while `federated_cb_only` federates only the CODEBOOK: encoders stay
#            local and diverge (cross-client token agreement measured 0.0000). Each client's prior
#            is scored on symbols it never saw. Deliberately NOT fixed here — repairing the
#            contamination is the owner's research decision, not a cleanup.
# RESULTS    NONE, ever: zero fa_* rows anywhere under artifacts/ (including the read-only
#            history artifacts/_archive_20260729/) and zero logs under logs/. No number this
#            file could print has ever been measured, so there is nothing here to cite.
# RETRACTED  The claim carried by 13 of the 14 fa_* docstrings — that these numbers sit on "the
#            same axis as the converged local / cb_only / centralized reports" — is FALSE
#            (documentation/RESEARCH_LEDGER.md, Group 4): a deployed cb_only client tokenizes
#            with its OWN encoder, so this layer measures an upper bound no deployment can
#            reach. Marked [RETRACTED] inline below wherever it occurs.
# REOPENING  needs an arm whose encoders are bit-identical across clients (`federated_enc_fedavg`);
#            see documentation/LAUNCH_RUNBOOK.md §5.2b. Entry points are guarded: this suite's
#            launcher scripts/launch_all_fa.sh refuses with exit 2 unless FA_I_KNOW_ITS_SHELVED=1.
# ═══════════════════════════════════════════════════════════════════════════════════════════════
#
# ── E4/DP-SPECIFIC RETRACTION (2026-07-15 audit; documentation/RESEARCH_LEDGER.md, header) ──
# THIS FILE DOES NOT PROVIDE DIFFERENTIAL PRIVACY. `sigma` below is NOT an ε and no (ε, δ) can
# be derived from it:
#   * there is no per-client clipping, so the L2 sensitivity of the count aggregate is UNBOUNDED;
#   * the noise scale is calibrated to N_train / K, i.e. it is a function of the PRIVATE data;
#   * the codebook statistic this rides on is released EVERY round (37-93 of them), so its budget
#     composes exactly like DP-SGD — the "ε paid once" advantage exists only for a genuinely
#     one-shot release such as the token histogram.
# The honest reading of anything this script could produce is "robustness to additive noise",
# NOT a privacy guarantee, and the "DP-graceful vs DP-SGD" selling point is WITHDRAWN. The code
# is kept as-is on purpose; only the claim is retracted.
"""
E8 — DIFFERENTIAL-PRIVACY TRADEOFF for the pooled per-position count prior (FA).
[RETRACTED — not DP: no clipping, unbounded sensitivity, data-dependent noise scale, per-round
 composition. See the E4/DP-SPECIFIC RETRACTION banner above. Read "sigma" as noise, not ε.]

Federated Analytics angle: the pooled per-position unigram over the K-codeword
grid (E1) is an EXACTLY-mergeable aggregate — an additive per-position histogram
`counts[P, K]` over ALL clients' train windows, tokenised on the SHARED (cb_only)
codebook. Every client contributes its own additive counts; the server sums
them. Because it is a plain additive statistic we can release it under the
Gaussian mechanism: add calibrated Gaussian noise ~ N(0, (sigma * scale_p)^2)
to each count cell, clip to >= 0, and renormalise per position back to a valid
categorical. `sigma` is RELATIVE to the per-position count scale (mean count per
codeword bin at that position = N_train / K). sigma=0 is the clean E1 aggregate.

The noised count prior is then scored with detect.py's REAL machinery (rolling
assembly, paper per-tau threshold, VUS / PATE) via mixture_eval._score_entity,
so the VUS-PR / AUPRC / PATE numbers land on the SAME axis as the converged
local / cb_only / centralized reports.
  ^^^ [RETRACTED 2026-07-27 — FALSE. See the banner at the top of this file: cb_only shares only
      the codebook, so the common tokenizer this sentence assumes does not exist.]

Optionally backs off to each client's own deep-local cb_only prior with weight
`--lam`:  combined per-token NLL = (1-lam) * count_nll + lam * deep_local_nll.
With lam=0 (default) the noised count prior is a STANDALONE scorer.

EXPECTED: graceful degradation of VUS-PR as sigma grows — a robust additive
aggregate tolerates DP noise, a selling point vs FL where per-round DP-SGD noise
on gradients compounds across communication rounds.
  ^^^ [RETRACTED 2026-07-15 — the "selling point vs DP-SGD" is withdrawn: the codebook statistic
      is re-released every round, so this composes like DP-SGD too. Noise robustness ≠ privacy.]

Variants = the sigma levels. Per (cluster, seed, sigma[, lam]) we write
    artifacts/fed_eval/fa_dp/<ds>/<cluster>/seed<n>/<variant>/<entity>/report.json
and append to artifacts/fed_eval/fa_dp/records_<ds>.jsonl.

Usage (smoke):
  CUDA_VISIBLE_DEVICES=1 python scripts/fa_dp.py \
      --dataset wsd_fed --clusters c3 --seeds 0 --sigmas 0,1
Usage (full):
  CUDA_VISIBLE_DEVICES=1 python scripts/fa_dp.py \
      --dataset wsd_fed --clusters all --seeds 0,1,2 --sigmas 0,0.5,1,2,4
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

from data import make_dataloaders  # noqa: E402
from stage2 import _flatten_token_indices  # noqa: E402
from federated import resolve_clients  # noqa: E402

# Reuse the tested FA infra from mixture_eval.
from mixture_eval import (  # noqa: E402
    _build_cfg, _load_pool, _score_entity, _Out, CONVERGED,
)

OUT_ROOT = REPO / "artifacts" / "fed_eval" / "fa_dp"


# ─── Pooled per-position count aggregate (the exactly-mergeable FA statistic) ──

@torch.no_grad()
def _build_pooled_counts(shared_stage1, cfg, entities, device, K):
    """Sum every client's additive per-position token histogram, tokenised with
    the SHARED codebook. Returns (counts (P,K) float64, C, F, W, n_train_windows).

    counts[p, k] = # of pooled train windows whose token at flat position p is
    codeword k. sum_k counts[p, k] = n_train (one token per position per window)."""
    counts = None
    C = F_ = W_ = None
    n_train = 0
    for e in entities:
        ce = copy.deepcopy(cfg)
        ce.dataset.entity_id = e
        loader = make_dataloaders(ce, stage="stage2").train_loader
        for batch in loader:
            x = batch["inputs"].to(device, non_blocking=True)
            _, idx, latent = shared_stage1.encode_tokens(x)      # idx (B, C, F*W)
            if counts is None:
                C = int(idx.shape[1]); F_ = int(latent[0]); W_ = int(latent[1])
                P = C * F_ * W_
                counts = torch.zeros(P, K, device=device, dtype=torch.float64)
            tokens = _flatten_token_indices(idx).long()          # (B, P)
            B, P = tokens.shape
            pos_idx = torch.arange(P, device=device).unsqueeze(0).expand(B, -1).reshape(-1)
            tok_idx = tokens.reshape(-1)
            counts.index_put_(
                (pos_idx, tok_idx),
                torch.ones(B * P, device=device, dtype=torch.float64),
                accumulate=True,
            )
            n_train += B
    return counts, C, F_, W_, n_train


def _noised_logp(counts, sigma, generator, laplace=1.0, eps=1e-9):
    """Gaussian-mechanism release of the per-position count aggregate.

    scale_p  = mean count per codeword bin at position p (= N_train/K, the
               per-position count scale). noise ~ N(0, (sigma*scale_p)^2) per cell.
    Clip to >= 0, add Laplace smoothing, renormalise per position -> valid
    categorical. Returns log-probabilities (P, K). sigma=0 reproduces the clean
    E1 Laplace-smoothed pooled prior EXACTLY."""
    P, K = counts.shape
    scale = counts.mean(dim=1, keepdim=True)                     # (P,1) per-position count scale
    if sigma > 0:
        noise = torch.randn(P, K, generator=generator, device=counts.device,
                            dtype=counts.dtype) * (sigma * scale)
        noised = counts + noise
    else:
        noised = counts
    noised = noised.clamp_min(0.0)                               # keep non-negative
    prob = (noised + laplace) / (noised.sum(dim=1, keepdim=True) + K * laplace)
    return prob.clamp_min(eps).log()                             # (P, K)


# ─── Count-prior scorer (quacks like a Stage2System for detect.py) ───────────

class CountPriorScorer:
    """Scores each window under a (noised) pooled per-position count prior and
    returns per-token NLL of shape (B,C,F,W) [per_rate=False] or (1,B,C,F,W)
    [per_rate=True]. With lam>0, backs off to the entity's own deep-local prior:
    combined = (1-lam)*count_nll + lam*deep_nll (per-rate consistent)."""

    def __init__(self, shared_stage1, logp, C, F_, W_, deep_prior, lam=0.0):
        self.stage1 = shared_stage1          # detect asserts .stage1.training is False
        self.prior = deep_prior              # detect asserts .prior.training is False
        self.logp = logp                     # (P, K) on device
        self.C, self.F, self.W = C, F_, W_
        self.lam = float(lam)
        self.deep = deep_prior if lam > 0 else None

    @torch.no_grad()
    def score_batch(self, batch: dict, per_rate: bool = False) -> _Out:
        _, indices, _ = self.stage1.encode_tokens(batch["inputs"])
        tokens = _flatten_token_indices(indices).long()          # (B, P)
        B, P = tokens.shape
        pos = torch.arange(P, device=tokens.device).unsqueeze(0).expand(B, -1)
        nll_count = -self.logp[pos, tokens]                      # (B, P)
        nll_count = nll_count.reshape(B, self.C, self.F, self.W)  # (B,C,F,W)

        if self.deep is None:
            return _Out(nll_count.unsqueeze(0) if per_rate else nll_count)

        # Backoff to the deep-local prior, keeping per_rate=False == sum_tau(per_rate=True).
        if per_rate:
            deep_pr = self.deep.score_tokens_per_rate(tokens)    # (n_tau,B,C,F,W)
            n_tau = deep_pr.shape[0]
            combined = self.lam * deep_pr + (1.0 - self.lam) * nll_count.unsqueeze(0) / n_tau
            return _Out(combined)
        deep_sum = self.deep.score_tokens(tokens)                # (B,C,F,W)
        combined = self.lam * deep_sum + (1.0 - self.lam) * nll_count
        return _Out(combined)


def _variant_name(sigma: float, lam: float) -> str:
    s = f"sigma{sigma:g}".replace(".", "p")
    if lam > 0:
        s += f"_lam{lam:g}".replace(".", "p")
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="wsd_fed")
    ap.add_argument("--clusters", default="all",
                    help="comma list or 'all' (discovered from converged tree)")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--sigmas", default="0,0.5,1,2,4",
                    help="Gaussian-mechanism noise levels relative to per-position count scale")
    ap.add_argument("--lam", type=float, default=0.0,
                    help="backoff weight to the deep-local cb_only prior (0 = standalone count prior)")
    ap.add_argument("--laplace", type=float, default=1.0, help="Laplace smoothing added post-noise")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = _build_cfg(args.dataset, args.batch)
    K = cfg.quantizer.codebook_size
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    sigmas = [float(s) for s in args.sigmas.split(",") if s.strip()]

    if args.clusters == "all":
        clusters = sorted(p.name for p in (CONVERGED / args.dataset).iterdir() if p.is_dir())
    else:
        clusters = [c.strip() for c in args.clusters.split(",") if c.strip()]

    print(f"[fa_dp] dataset={args.dataset} clusters={clusters} seeds={seeds} "
          f"sigmas={sigmas} lam={args.lam} K={K} device={device}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rec_path = OUT_ROOT / f"records_{args.dataset}.jsonl"
    # Refuse to truncate the ledger to nothing (same guard as mixture_eval:295-303). This script
    # STREAMS records (one flushed write per row, crash-resilient), so it cannot decide at the
    # END whether to open the file: instead the truncating open("w") is DEFERRED to the first
    # record. A run that scores zero entities therefore never touches rec_path and raises below,
    # so a good records file is never overwritten with 0 bytes — and a 0-byte ledger is
    # indistinguishable from "never run" for every downstream reducer.
    rec_fh = None

    n_written = 0
    for cluster in clusters:
        entities = resolve_clients(cfg, None, cluster)
        for seed in seeds:
            shared_stage1, pool = _load_pool(args.dataset, cluster, seed, entities, cfg, device)
            if shared_stage1 is None:
                print(f"[fa_dp] SKIP {cluster} seed{seed}: cb_only run missing/incomplete")
                continue
            priors, have = pool
            print(f"[fa_dp] {cluster} seed{seed}: {len(priors)} priors over {have}")

            # Build the pooled additive count aggregate ONCE for this (cluster, seed).
            counts, C, F_, W_, n_train = _build_pooled_counts(shared_stage1, cfg, have, device, K)
            P = C * F_ * W_
            print(f"[fa_dp] {cluster} seed{seed}: pooled counts P={P} (C={C},F={F_},W={W_}) "
                  f"K={K} n_train_windows={n_train} count_scale~{n_train / K:.1f}")

            for sigma in sigmas:
                variant = _variant_name(sigma, args.lam)
                # Seeded noise realisation, shared across the entities of this (cluster,seed,sigma).
                gen = torch.Generator(device=device)
                gen.manual_seed(seed * 100003 + int(round(sigma * 1000)))
                logp = _noised_logp(counts, sigma, gen, laplace=args.laplace)

                out_dir = OUT_ROOT / args.dataset / cluster / f"seed{seed}" / variant
                for e in have:
                    deep_prior = priors[have.index(e)]
                    scorer = CountPriorScorer(shared_stage1, logp, C, F_, W_,
                                              deep_prior=deep_prior, lam=args.lam)
                    rep = _score_entity(shared_stage1, scorer, cfg, e)
                    d = out_dir / e
                    d.mkdir(parents=True, exist_ok=True)
                    (d / "report.json").write_text(json.dumps(rep, indent=2))
                    rec = {"_arm": f"fa_dp_{variant}", "_cluster": cluster, "_seed": seed,
                           "_entity": e,
                           **{k: float(rep[k]) for k in ("vus_pr", "auprc", "pate_f1", "auroc")
                              if isinstance(rep.get(k), (int, float)) and np.isfinite(rep.get(k))}}
                    if rec_fh is None:                     # first record: NOW truncate + open
                        rec_fh = rec_path.open("w")
                    rec_fh.write(json.dumps(rec) + "\n")
                    rec_fh.flush()
                    n_written += 1
                    print(f"  [{variant} {cluster} s{seed}] {e}: "
                          f"vus_pr={rep.get('vus_pr', float('nan')):.3f} "
                          f"auprc={rep.get('auprc', float('nan')):.3f} "
                          f"pate_f1={rep.get('pate_f1', float('nan')):.3f}")

            del shared_stage1, priors, counts
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if rec_fh is None:
        raise SystemExit(
            "[fa_dp] produced 0 records — refusing to write an empty ledger.\n"
            "  Every (cluster, seed) was skipped: the per-arm checkpoints are missing.\n"
            "  Fix the inputs; do not let this overwrite an existing records file."
        )
    rec_fh.close()
    print(f"[fa_dp] wrote {n_written} records -> {rec_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
