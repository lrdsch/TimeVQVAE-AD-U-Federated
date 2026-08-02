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
E13 — FEDERATED WHITENING of the VQ latent (novel; covariance is a sufficient
statistic).

The shared codebook (merged by suff-stat in the `federated_cb_only` arm) lives in
the RAW encoder-latent space and assigns tokens by plain Euclidean nearest
neighbour (`torch.cdist(z, codebook).argmin`). If the latent coordinates are
correlated / anisotropic, Euclidean assignment over-weights the high-variance
directions and wastes codewords — a worse "token language" for the downstream
count prior.

FEDERATED WHITENING fixes the *metric*, not the weights, with a purely additive
(exactly-mergeable) Federated-Analytics object: each client ships the second-order
suff-stats of its PRE-quantizer latent  z ∈ R^d  (count n_k, sum s_k, outer-sum
S_k). The server pools them,

    N = Σ n_k ,  μ = Σ s_k / N ,  Σ = (Σ S_k)/N − μμᵀ ,

eigendecomposes Σ = U Λ Uᵀ and builds the ZCA whitening transform
W = U Λ^(−α/2) Uᵀ (α = whitening strength). Re-quantizing each latent by
Euclidean NN in the WHITENED space is exactly Mahalanobis-NN in the raw space —
every latent dimension is given equal weight. Optionally the codebook centroids
are RE-FIT in the whitened space by a federated Lloyd M-step (`--refit-iters`),
each client reporting per-centroid (sum, count) against the current centroids —
the same suff-stat aggregation as the `cb_only` codebook merge.

The re-quantized token stream is scored by a POOLED per-position count prior
(Laplace + global-marginal backoff), then run through detect.py's REAL machinery
(rolling assembly, paper per-τ threshold, VUS-PR / AUPRC / PATE) via
mixture_eval._score_entity — so the numbers sit on the same axis as the converged
local / cb_only / centralized arms.
  ^^^ [RETRACTED 2026-07-27 — FALSE. See the banner at the top of this file: cb_only shares only
      the codebook, so the common tokenizer this sentence assumes does not exist.]

Variants:
  * no_whiten        — Euclidean NN on the raw codebook (control; reproduces the
                       cb_only tokenisation, count-prior scored).
  * federated_whiten — Mahalanobis NN via the pooled whitening transform (+refit).
We also report tok-agree (fraction of train tokens unchanged by whitening) — the
codebook-coherence change — and the VUS-PR delta whiten − control.

Usage (smoke):
  CUDA_VISIBLE_DEVICES=1 python scripts/fa_pca.py \
      --dataset wsd_fed --clusters c3 --seeds 0 --refit-iters 1
Full (bounded, exploratory):
  CUDA_VISIBLE_DEVICES=1 python scripts/fa_pca.py \
      --dataset wsd_fed --clusters c0,c3 --seeds 0,1 --refit-iters 2
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

from config import Config  # noqa: E402
from data import make_dataloaders  # noqa: E402
from stage2 import _flatten_token_indices  # noqa: E402
from federated import resolve_clients  # noqa: E402

# Reuse the TESTED mixture_eval infra (cfg build, cb_only pool loader, real-axis
# entity scorer, the _Out score container, the converged-tree root).
from mixture_eval import _build_cfg, _load_pool, _score_entity, _Out, CONVERGED  # noqa: E402

OUT_ROOT = REPO / "artifacts" / "fed_eval" / "fa_pca"


# ─── Pre-quantizer latent extraction (reproduces encode_tokens' Euclidean NN) ─

@torch.no_grad()
def _encode_latents(stage1, x: torch.Tensor):
    """Encoder-latent (pre-quantizer) vectors for a batch, folded to the SAME
    (channel, spatial) token order encode_tokens/_flatten_token_indices use:
    position p = c·(F·W) + f·W + w. Returns (z (B, N, d) float32, C, F, W)."""
    tf = stage1.transform(x)
    q = stage1.quantizer
    if hasattr(q, "set_groups") and getattr(q, "groups", None) is None:
        q.set_groups(tf.spec.original_channels)
    latent = stage1.encoder(tf)                    # (B, C*d, F, W)
    B, Cd, F_, W = latent.shape
    C = int(q.groups)
    d = Cd // C
    z = latent.reshape(B, C, d, F_, W).permute(0, 1, 3, 4, 2).reshape(B, C * F_ * W, d)
    return z.float(), C, F_, W


@torch.no_grad()
def _assign(z: torch.Tensor, mu: torch.Tensor, Wt: torch.Tensor, cent: torch.Tensor) -> torch.Tensor:
    """Nearest-centroid token ids in the (optionally whitened) space.
    z (B,N,d) → whitened zt=(z-μ)Wᵀ; centroids `cent` (K,d) already live in that
    space. Euclidean NN there == Mahalanobis NN in the raw space. (B, N)."""
    zt = (z - mu) @ Wt.T                            # (B, N, d)
    dist = torch.cdist(zt, cent.unsqueeze(0))       # (B, N, K)  broadcasts B vs 1
    return dist.argmin(dim=-1)


# ─── Per-position count prior (quacks like a MaskGIT prior for detect.py) ─────

class _CountPrior:
    """Per-position count prior. Exposes `.training = False` and the two scorers
    detect asks for, returning per-token NLL (B,C,F,W) / (1,B,C,F,W) [a count
    prior has no masking-rate axis → n_τ = 1]."""

    def __init__(self, logp: torch.Tensor, C: int, F_: int, W: int):
        self.logp = logp                 # (P, K) log p(codeword | position)
        self.C, self.F, self.W = int(C), int(F_), int(W)
        self.training = False

    @torch.no_grad()
    def score_tokens_per_rate(self, tokens: torch.Tensor) -> torch.Tensor:
        B, P = tokens.shape
        pos = torch.arange(P, device=tokens.device)
        nll = -self.logp[pos[None, :], tokens]       # (B, P) advanced-index
        return nll.reshape(1, B, self.C, self.F, self.W)

    @torch.no_grad()
    def score_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.score_tokens_per_rate(tokens).sum(dim=0)


class WhitenCountScorer:
    """Quacks like a Stage2System for detect._compute_entities_raw. Extracts the
    pre-quantizer latent with the SHARED stage1, RE-QUANTIZES it with a custom
    (whitened or raw) tokenizer, then scores each token under the pooled count
    prior. detect never re-tokenizes — this object owns the token stream."""

    def __init__(self, shared_stage1, prior: _CountPrior,
                 mu: torch.Tensor, Wt: torch.Tensor, cent: torch.Tensor):
        self.stage1 = shared_stage1      # detect asserts .stage1.training is False
        self.prior = prior               # detect asserts .prior.training is False
        self.mu, self.Wt, self.cent = mu, Wt, cent

    @torch.no_grad()
    def score_batch(self, batch: dict, per_rate: bool = False) -> _Out:
        z, _, _, _ = _encode_latents(self.stage1, batch["inputs"])
        tokens = _assign(z, self.mu, self.Wt, self.cent).long()   # (B, N)
        if per_rate:
            return _Out(self.prior.score_tokens_per_rate(tokens))
        return _Out(self.prior.score_tokens(tokens))


# ─── Federated covariance suff-stat → whitening transform ────────────────────

@torch.no_grad()
def _federated_covariance(stage1, cfg: Config, entities: list[str], device, d: int):
    """Pool per-client second-order latent suff-stats (additive → exactly
    mergeable). Returns (mu (d,), Cov (d,d), N) in float64 on `device`."""
    n = torch.zeros((), dtype=torch.float64, device=device)
    s = torch.zeros(d, dtype=torch.float64, device=device)
    S = torch.zeros(d, d, dtype=torch.float64, device=device)
    for e in entities:
        ce = copy.deepcopy(cfg); ce.dataset.entity_id = e
        loader = make_dataloaders(ce, stage="stage2").train_loader
        for batch in loader:
            x = batch["inputs"].to(device, non_blocking=True)
            z, _, _, _ = _encode_latents(stage1, x)               # (B, N, d)
            zf = z.reshape(-1, d).double()                        # (B*N, d)
            n += zf.shape[0]
            s += zf.sum(dim=0)
            S += zf.T @ zf
    if float(n) < 1.0:
        raise ValueError("no train latents found to build the federated covariance")
    mu = s / n
    Cov = S / n - torch.outer(mu, mu)
    Cov = 0.5 * (Cov + Cov.T)                                     # symmetrize
    return mu, Cov, float(n)


def _whitening_transform(Cov: torch.Tensor, alpha: float, ridge: float):
    """ZCA whitening W = U Λ^(−α/2) Uᵀ with a relative ridge on Σ. α=0 → identity,
    α=1 → full whitening. Returns (Wt (d,d) float64, condition_number)."""
    d = Cov.shape[0]
    tr = torch.diagonal(Cov).sum() / d
    Cov_r = Cov + ridge * tr * torch.eye(d, dtype=Cov.dtype, device=Cov.device)
    evals, evecs = torch.linalg.eigh(Cov_r)                      # ascending, symmetric
    evals = evals.clamp_min(1e-12)
    cond = float((evals[-1] / evals[0]).item())
    inv = evals.pow(-0.5 * float(alpha))
    Wt = (evecs * inv.unsqueeze(0)) @ evecs.T                    # U diag Uᵀ (symmetric ZCA)
    return Wt, cond


# ─── Federated Lloyd refit of the codebook in the whitened space ─────────────

@torch.no_grad()
def _federated_refit(stage1, cfg, entities, device, d, K, mu, Wt, cent, iters):
    """`iters` federated k-means M-steps in the whitened space. Each client
    reports per-centroid (sum, count) against the current centroids; the server
    sets e_j = Σsum_j / Σcount_j (empty centroids kept). Returns updated `cent`."""
    for _ in range(max(0, int(iters))):
        sumK = torch.zeros(K, d, dtype=torch.float64, device=device)
        cntK = torch.zeros(K, dtype=torch.float64, device=device)
        for e in entities:
            ce = copy.deepcopy(cfg); ce.dataset.entity_id = e
            loader = make_dataloaders(ce, stage="stage2").train_loader
            for batch in loader:
                x = batch["inputs"].to(device, non_blocking=True)
                z, _, _, _ = _encode_latents(stage1, x)
                zt = ((z - mu) @ Wt.T).reshape(-1, d)            # (M, d) whitened
                dist = torch.cdist(zt.unsqueeze(0), cent.unsqueeze(0))[0]  # (M, K)
                idx = dist.argmin(dim=-1)                         # (M,)
                oh = torch.zeros(zt.shape[0], K, device=device, dtype=torch.float64)
                oh[torch.arange(zt.shape[0], device=device), idx] = 1.0
                sumK += oh.T @ zt.double()
                cntK += oh.sum(dim=0)
        nonempty = cntK > 0
        new = cent.clone()
        new[nonempty] = (sumK[nonempty] / cntK[nonempty].unsqueeze(1)).to(cent.dtype)
        cent = new
    return cent


# ─── Pooled per-position count prior + tok-agree, ONE pass over train data ───

@torch.no_grad()
def _build_priors_and_agree(stage1, cfg, entities, device, K, lam,
                            mu, Wt_e, cent_e, mu_w, Wt_w, cent_w):
    """Single pass: tokenise each train window with BOTH tokenizers (eucl/control
    + whitened), accumulate a per-position codeword count table for each, and the
    token-agreement fraction. Returns (logp_eucl, logp_white, N, (C,F,W), agree)."""
    counts_e = counts_w = None
    N = None
    C = F_ = W = None
    n_agree = 0
    n_tok = 0
    for e in entities:
        ce = copy.deepcopy(cfg); ce.dataset.entity_id = e
        loader = make_dataloaders(ce, stage="stage2").train_loader
        for batch in loader:
            x = batch["inputs"].to(device, non_blocking=True)
            z, c, f, w = _encode_latents(stage1, x)              # (B, N, d)
            te = _assign(z, mu, Wt_e, cent_e).long()             # (B, N)
            tw = _assign(z, mu_w, Wt_w, cent_w).long()           # (B, N)
            B, n = te.shape
            if counts_e is None:
                N, C, F_, W = n, c, f, w
                counts_e = torch.zeros(N * K, device=device)
                counts_w = torch.zeros(N * K, device=device)
            elif n != N:
                raise ValueError(f"inconsistent token count {n} != {N}")
            pos = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)  # (B, N)
            counts_e += torch.bincount((pos * K + te).reshape(-1), minlength=N * K).float()
            counts_w += torch.bincount((pos * K + tw).reshape(-1), minlength=N * K).float()
            n_agree += int((te == tw).sum().item())
            n_tok += B * n
    if counts_e is None:
        raise ValueError("no train windows found to build the count prior")
    logp_e = _counts_to_logp(counts_e.reshape(N, K), lam, K)
    logp_w = _counts_to_logp(counts_w.reshape(N, K), lam, K)
    agree = (n_agree / n_tok) if n_tok else float("nan")
    return logp_e, logp_w, N, (C, F_, W), agree


def _counts_to_logp(counts: torch.Tensor, lam: float, K: int) -> torch.Tensor:
    """counts (P,K) → log p(codeword|position) with Laplace + global-marginal
    backoff at weight `lam` (a uniform floor keeps every NLL finite)."""
    eps = 1e-12
    pos = (counts + 1.0)
    pos = pos / pos.sum(dim=1, keepdim=True)                     # (P, K) Laplace
    g = counts.sum(dim=0)
    g = g / g.sum().clamp_min(eps)
    g = 0.999 * g + 0.001 * (1.0 / K)
    g = g / g.sum()
    pback = (1.0 - lam) * pos + lam * g.unsqueeze(0)
    return pback.clamp_min(eps).log()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="wsd_fed")
    ap.add_argument("--clusters", default="all",
                    help="comma list or 'all' (discovered from converged tree)")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--variants", default="no_whiten,federated_whiten")
    ap.add_argument("--alpha", type=float, default=1.0, help="whitening strength (0=off,1=full)")
    ap.add_argument("--ridge", type=float, default=1e-3, help="relative ridge on the pooled covariance")
    ap.add_argument("--refit-iters", type=int, default=0,
                    help="federated Lloyd M-steps to re-fit the codebook in whitened space")
    ap.add_argument("--lam", type=float, default=0.1, help="global-marginal backoff weight")
    ap.add_argument("--batch", type=int, default=64)
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

    print(f"[fa_pca] dataset={args.dataset} clusters={clusters} seeds={seeds} "
          f"variants={variants} alpha={args.alpha} ridge={args.ridge} "
          f"refit_iters={args.refit_iters} lam={args.lam} K={K} device={device}")

    all_records = []
    for cluster in clusters:
        entities = resolve_clients(cfg, None, cluster)
        for seed in seeds:
            shared_stage1, pool = _load_pool(args.dataset, cluster, seed, entities, cfg, device)
            if shared_stage1 is None:
                print(f"[fa_pca] SKIP {cluster} seed{seed}: cb_only run missing/incomplete")
                continue
            deep_priors, have = pool
            del deep_priors                                  # unused: free deep-prior GPU mem
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            cb = shared_stage1.quantizer._vq.codebook.weight.detach().to(device).double()  # (K, d)
            Kc, d = cb.shape
            assert Kc == K, f"codebook size {Kc} != cfg K {K}"
            eye = torch.eye(d, dtype=torch.float64, device=device)
            zero = torch.zeros(d, dtype=torch.float64, device=device)

            # ── Federated covariance suff-stat → whitening transform ──────────
            mu, Cov, Ncov = _federated_covariance(shared_stage1, cfg, have, device, d)
            Wt_w, cond = _whitening_transform(Cov, args.alpha, args.ridge)
            cent_w = ((cb - mu) @ Wt_w.T)                    # codebook in whitened space
            if args.refit_iters > 0:
                cent_w = _federated_refit(shared_stage1, cfg, have, device, d, K,
                                          mu, Wt_w, cent_w, args.refit_iters)
            # Control (Euclidean) tokenizer params.
            mu_e, Wt_e, cent_e = zero, eye, cb

            print(f"[fa_pca] {cluster} seed{seed}: {len(have)} clients {have}; "
                  f"d={d} N_latents={Ncov:.0f} cov_cond={cond:.3g} refit={args.refit_iters}")

            # ── Pooled per-position count priors (both tokenizers) + tok-agree ─
            logp_e, logp_w, N, (C, F_, W), agree = _build_priors_and_agree(
                shared_stage1, cfg, have, device, K, args.lam,
                mu_e, Wt_e, cent_e, mu, Wt_w, cent_w)
            assert N == C * F_ * W, f"count N={N} != C*F*W={C*F_*W}"
            print(f"[fa_pca] {cluster} seed{seed}: grid C={C} F={F_} W={W} N={N}; "
                  f"tok-agree(control,whiten)={agree:.4f}")

            variant_cfg = {
                "no_whiten":        (logp_e, mu_e, Wt_e, cent_e),
                "federated_whiten": (logp_w, mu,   Wt_w, cent_w),
            }
            cluster_vus = {}
            for v in variants:
                if v not in variant_cfg:
                    raise SystemExit(f"unknown variant {v!r}; choose from {list(variant_cfg)}")
                logp, mv, Wv, centv = variant_cfg[v]
                prior = _CountPrior(logp, C, F_, W)
                scorer = WhitenCountScorer(shared_stage1, prior,
                                           mv.to(torch.float32), Wv.to(torch.float32),
                                           centv.to(torch.float32))
                out_dir = OUT_ROOT / args.dataset / cluster / f"seed{seed}" / v
                vus = []
                for e in have:
                    rep = _score_entity(shared_stage1, scorer, cfg, e)
                    rep["_tok_agree"] = agree
                    rep["_cov_cond"] = cond
                    d_ = out_dir / e; d_.mkdir(parents=True, exist_ok=True)
                    (d_ / "report.json").write_text(json.dumps(rep, indent=2))
                    rec = {"_arm": f"fa_pca_{v}", "_cluster": cluster, "_seed": seed, "_entity": e,
                           **{k: float(rep[k]) for k in ("auroc", "auprc", "vus_pr", "pate_f1")
                              if isinstance(rep.get(k), (int, float)) and np.isfinite(rep.get(k))}}
                    all_records.append(rec)
                    vp = rep.get("vus_pr", float("nan"))
                    if isinstance(vp, (int, float)) and np.isfinite(vp):
                        vus.append(float(vp))
                    print(f"  [{v} {cluster} s{seed}] {e}: "
                          f"vus_pr={vp if isinstance(vp,(int,float)) else float('nan'):.3f} "
                          f"auprc={rep.get('auprc', float('nan')):.3f} "
                          f"pate_f1={rep.get('pate_f1', float('nan')):.3f}")
                cluster_vus[v] = float(np.mean(vus)) if vus else float("nan")

            if "no_whiten" in cluster_vus and "federated_whiten" in cluster_vus:
                c0, cw = cluster_vus["no_whiten"], cluster_vus["federated_whiten"]
                print(f"[fa_pca DELTA {cluster} s{seed}] cluster-mean VUS-PR | "
                      f"control={c0:.3f} whiten={cw:.3f} delta={cw - c0:+.3f} "
                      f"(tok-agree={agree:.3f})")

            del shared_stage1, cb, logp_e, logp_w
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rec_path = OUT_ROOT / f"records_{args.dataset}.jsonl"
    with rec_path.open("a") as fh:
        for r in all_records:
            fh.write(json.dumps(r) + "\n")
    print(f"[fa_pca] appended {len(all_records)} records -> {rec_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
