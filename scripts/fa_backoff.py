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
"""
E1 — COUNT-BACKOFF to the deep-local prior (Federated Analytics).

We already trained, in the `federated_cb_only` arm, one deep MaskGIT prior per
client on a SHARED (suff-stat merged) codebook. That deep-local prior overfits
its own client and STARVES its rare-token tails (cb_only ≪ local on VUS-PR). The
fix explored here does NOT touch weights: it interpolates, in PROBABILITY space
per token, the client's OWN deep prior with a POOLED per-position unigram count
model — an exactly-mergeable Federated-Analytics object (Laplace-smoothed token
occupancy per grid position, summed over all clients' train windows tokenized
with the shared codebook).

  nll_bo = -log( λ·exp(-nll_deep) + (1-λ)·exp(-nll_count) )

where
  * nll_deep  = THIS entity's own cb_only prior score_tokens output (B,C,F,W)
                — τ-summed (per_rate=False) or per-τ (per_rate=True).
  * nll_count = pooled per-position count NLL of the observed token, broadcast
                to (B,C,F,W) (and over the τ axis when per_rate=True).

Deploy is PER-CLIENT: entity e uses its OWN deep prior + the shared count model.
λ is swept in {0.1,0.3,0.5,0.7} as separate variants. λ→1 recovers cb_only;
λ→0 recovers the pooled count prior. The bet: a middle λ regularizes the starved
tails and BEATS local without dirtying weights.

Everything downstream (rolling assembly, paper threshold, VUS-PR/AUPRC/PATE) is
the EXACT detect.py machinery via mixture_eval._score_entity — so the numbers are
directly comparable to the converged local / cb_only / centralized reports.
  ^^^ [RETRACTED 2026-07-27 — FALSE. See the banner at the top of this file: cb_only shares only
      the codebook, so the common tokenizer this sentence assumes does not exist.]

Usage (smoke):
  CUDA_VISIBLE_DEVICES=1 python scripts/fa_backoff.py \
      --dataset wsd_fed --clusters c3 --seeds 0 --lams 0.5
Usage (full):
  CUDA_VISIBLE_DEVICES=1 python scripts/fa_backoff.py \
      --dataset wsd_fed --clusters all --seeds 0,1,2
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pipeline"))
sys.path.insert(0, str(REPO / "scripts"))

from config import Config  # noqa: E402
from data import make_dataloaders  # noqa: E402
from stage2 import _flatten_token_indices  # noqa: E402
from federated import resolve_clients  # noqa: E402

# Reuse the tested infra from mixture_eval verbatim.
from mixture_eval import _build_cfg, _load_pool, _score_entity, _Out, CONVERGED  # noqa: E402

OUT_ROOT = REPO / "artifacts" / "fed_eval" / "fa_backoff"


class BackoffStage2:
    """Quacks like a Stage2System for detect._compute_entities_raw, but scores a
    window under a SINGLE deep prior (this entity's own cb_only prior) backed off
    to a POOLED per-position unigram count model, combined per-token in
    probability space:

        nll_bo = -log( λ·exp(-nll_deep) + (1-λ)·exp(-nll_count) )

    Tokenises ONCE with the shared stage1 so the deep prior and the count model
    see the identical token grid. `logp` is (N, K) log token-probability per
    flattened grid position; the count NLL of the observed token aligns to the
    deep prior's (B, C, F, W) grid by the same reshape the prior uses internally
    (base = tokens.reshape(B, C, F, W))."""

    def __init__(self, shared_stage1, prior, logp: torch.Tensor, lam: float, C: int, F: int, W: int):
        self.stage1 = shared_stage1          # detect asserts .stage1.training is False
        self.prior = prior                   # this entity's OWN cb_only deep prior, eval()
        self.logp = logp                     # (N, K) pooled per-position log p(token|pos)
        self.lam = float(lam)
        self._la = math.log(lam)
        self._lb = math.log(1.0 - lam)
        self._C, self._F, self._W = int(C), int(F), int(W)
        self._N = self._C * self._F * self._W
        # Degeneracy telemetry: fraction of scored tokens where the DEEP term wins the
        # mixture. If this sits at ~0 the λ sweep is not a sweep — every variant is the
        # pure count prior. Read it after scoring and record it.
        self.n_tok = 0
        self.n_deep_wins = 0

    @torch.no_grad()
    def score_batch(self, batch: dict, per_rate: bool = False) -> _Out:
        _, indices, _ = self.stage1.encode_tokens(batch["inputs"])
        tokens = _flatten_token_indices(indices).long()          # (B, N)
        B, N = tokens.shape
        assert N == self._N, f"token count {N} != C*F*W = {self._N}"
        dev = tokens.device

        # Pooled per-position count NLL of the observed token, aligned to (B,C,F,W).
        pos = torch.arange(N, device=dev).unsqueeze(0).expand(B, -1)   # (B, N)
        nll_count = -self.logp[pos, tokens]                            # (B, N)
        nll_count = nll_count.reshape(B, self._C, self._F, self._W)    # (B, C, F, W)

        # This entity's own deep prior NLL — ALWAYS per-τ, then mixed per-τ.
        #
        # τ-SCALE BUG (fixed 2026-07-27). The old code took `score_tokens(tokens)` on the
        # per_rate=False path. That is the SUM over the n_τ=3 score_window_size_rates
        # (model/prior.py score_tokens; config.py:202), so a τ-summed deep term was mixed
        # against a SINGLE-τ nll_count — a 3x inflation of the deep NLL that drives the
        # logaddexp onto the count model at every λ, collapsing the four λ variants into
        # one (the pure count prior). It also broke the τ identity:
        #     Σ_τ mix(per_rate=True)  ≠  mix(per_rate=False)
        # because mixing-then-summing is not summing-then-mixing. detect._fit_threshold_paper
        # fits the paper threshold on the τ-sum of the per-rate stack while _score_entity
        # scores the test set through the per_rate=False path, so the threshold was fitted
        # on a different formula than the score — which is why every threshold-dependent
        # metric this script wrote (precision/recall/f1/event_*/pate/delay) came out
        # identically zero while vus_pr still looked plausible and the script exited 0.
        #
        # Mixing per-τ and summing afterwards restores the identity by construction. This
        # mirrors fa_coldstart.py:105-117; note the λ convention differs — here λ weights
        # the DEEP term, there it weights the COUNT term.
        nll_deep = self.prior.score_tokens_per_rate(tokens)             # (n_τ, B, C, F, W)
        nll_count_b = nll_count.unsqueeze(0).expand_as(nll_deep)        # broadcast across τ

        # Interpolate in probability space, per token, numerically stable via logaddexp.
        term_a = self._la - nll_deep
        term_b = self._lb - nll_count_b
        mix = -torch.logaddexp(term_a, term_b)                          # (n_τ, B, C, F, W)

        self.n_tok += mix.numel()
        self.n_deep_wins += int((term_a > term_b).sum())

        return _Out(mix if per_rate else mix.sum(dim=0))


@torch.no_grad()
def _pooled_position_logp(shared_stage1, cfg: Config, entities: list[str], device, K: int):
    """POOLED per-position unigram count model over ALL clients' train windows,
    tokenized with the shared stage1. Laplace-smoothed log p(token | position) on
    the flattened C*F*W grid. Additive across clients = exactly mergeable (a pure
    Federated-Analytics object). Returns (logp (N, K), N)."""
    counts = None
    N = None
    for e in entities:
        ce = copy.deepcopy(cfg); ce.dataset.entity_id = e
        loader = make_dataloaders(ce, stage="stage2").train_loader
        for batch in loader:
            x = batch["inputs"].to(device, non_blocking=True)
            _, idx, _ = shared_stage1.encode_tokens(x)
            t = _flatten_token_indices(idx).long()                 # (B, N)
            B, n = t.shape
            if counts is None:
                N = n
                counts = torch.zeros(N * K, device=device)
            elif n != N:
                raise ValueError(f"inconsistent token count: {n} != {N}")
            pos = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)  # (B, N)
            flat = (pos * K + t).reshape(-1)                        # position-major bucket
            counts += torch.bincount(flat, minlength=N * K).float()
    if counts is None:
        raise ValueError("no train windows found to build the pooled count model")
    counts = counts.reshape(N, K) + 1.0                            # +1 Laplace
    logp = (counts / counts.sum(dim=1, keepdim=True)).log()        # (N, K)
    return logp, N


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="wsd_fed")
    ap.add_argument("--clusters", default="all",
                    help="comma list or 'all' (discovered from converged tree)")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--lams", default="0.1,0.3,0.5,0.7",
                    help="backoff mixing weights on the deep prior (variants)")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = _build_cfg(args.dataset, args.batch)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    lams = [float(x) for x in args.lams.split(",") if x.strip()]
    for lam in lams:
        if not (0.0 < lam < 1.0):
            raise SystemExit(f"lam must be in (0,1); got {lam}")
    K = cfg.quantizer.codebook_size

    if args.clusters == "all":
        clusters = sorted(p.name for p in (CONVERGED / args.dataset).iterdir()
                          if p.is_dir())
    else:
        clusters = [c.strip() for c in args.clusters.split(",") if c.strip()]

    print(f"[fa_backoff] dataset={args.dataset} clusters={clusters} seeds={seeds} "
          f"lams={lams} K={K} device={device}")

    all_records = []
    for cluster in clusters:
        entities = resolve_clients(cfg, None, cluster)
        for seed in seeds:
            shared_stage1, pool = _load_pool(args.dataset, cluster, seed, entities, cfg, device)
            if shared_stage1 is None:
                print(f"[fa_backoff] SKIP {cluster} seed{seed}: cb_only run missing/incomplete")
                continue
            priors, have = pool
            prior_by_entity = dict(zip(have, priors))
            # C, F, W from the (materialized) deep prior — the grid the count model aligns to.
            p0 = priors[0]
            C, F_, W = int(p0.latent_channels), int(p0.latent_freq), int(p0.latent_time)
            print(f"[fa_backoff] {cluster} seed{seed}: {len(priors)} priors over {have} "
                  f"(C={C} F={F_} W={W})")

            # POOLED per-position count model (built ONCE, shared across λ + entities).
            logp, N = _pooled_position_logp(shared_stage1, cfg, have, device, K)
            assert N == C * F_ * W, f"count N={N} != C*F*W={C * F_ * W}"

            for lam in lams:
                variant = f"lam{lam:g}"
                out_dir = OUT_ROOT / args.dataset / cluster / f"seed{seed}" / variant
                for e in have:
                    scorer = BackoffStage2(shared_stage1, prior_by_entity[e], logp, lam, C, F_, W)
                    rep = _score_entity(shared_stage1, scorer, cfg, e)
                    d = out_dir / e; d.mkdir(parents=True, exist_ok=True)
                    (d / "report.json").write_text(json.dumps(rep, indent=2))
                    rec = {"_arm": f"fa_backoff_{variant}", "_cluster": cluster,
                           "_seed": seed, "_entity": e,
                           **{k: float(rep[k]) for k in ("auroc", "auprc", "vus_pr", "pate_f1")
                              if isinstance(rep.get(k), (int, float)) and np.isfinite(rep.get(k))}}
                    all_records.append(rec)
                    print(f"  [{variant} {cluster} s{seed}] {e}: "
                          f"vus_pr={rep.get('vus_pr', float('nan')):.3f} "
                          f"auprc={rep.get('auprc', float('nan')):.3f} "
                          f"pate_f1={rep.get('pate_f1', float('nan')):.3f}")

            del shared_stage1, priors, logp
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Refuse to truncate the ledger to nothing (same guard as mixture_eval:295-303). The file
    # is opened in "w", so a run that scored zero entities would OVERWRITE a good records file
    # with 0 bytes — and a 0-byte ledger is indistinguishable from "never run" for every
    # downstream reducer.
    if not all_records:
        raise SystemExit(
            "[fa_backoff] produced 0 records — refusing to write an empty ledger.\n"
            "  Every (cluster, seed) was skipped: the per-arm checkpoints are missing.\n"
            "  Fix the inputs; do not let this overwrite an existing records file."
        )

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rec_path = OUT_ROOT / f"records_{args.dataset}.jsonl"
    with rec_path.open("w") as fh:
        for r in all_records:
            fh.write(json.dumps(r) + "\n")
    print(f"[fa_backoff] wrote {len(all_records)} records -> {rec_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
