"""
E9 — BYZANTINE ROBUSTNESS of Federated-Analytics count-prior aggregation.

Story: the deployable anomaly score can be a POOLED per-position count prior over
the shared K-codeword token grid (the exactly-mergeable object of ngram_prior.py),
NOT the deep MaskGIT prior. Federated Analytics ships each client's per-position
token statistic and the server aggregates them. This script asks: if ONE client is
Byzantine (its train tokens are poisoned before the statistic is built), does a
ROBUST aggregator of the *statistic* defend the detector better than the plain MEAN?

Per client k we build the per-position codeword counts  count_k[p, m]  (p = flat
token position over C*F*W, m = codeword) and window count n_k. Two aggregators
collapse the client axis:

  * MEAN  (non-robust) = SUM of raw counts:  agg[p,m] = Σ_k count_k[p,m]. This is
    the exactly-mergeable pooled count. It is non-robust because a Byzantine client
    can OVER-REPORT its volume (--poison-scale): its uniform-random counts are added
    straight in and, at enough volume, dominate the pool → the per-position prior
    collapses toward uniform and the detector loses its signal.
  * ROBUST (defended) = coordinate-wise median (or trimmed mean) of the SCALE-FREE
    per-position rate  f_k[p,m] = count_k[p,m]/n_k  (each client's per-position row
    sums to 1). The attacker's inflated volume cancels in the normalisation, so it
    is merely a rank outlier and is discarded.

The aggregated table becomes a per-position distribution with a global-marginal
BACKOFF at weight `lam` (default 0.3):
      p(m|pos) = (1-lam)·pos[p,m] + lam·global[m]
scored as per-token NLL  -log p(token|pos), then run through detect.py's REAL
machinery (rolling assembly, paper per-τ threshold, VUS-PR/AUPRC/PATE) so the
numbers sit on the same axis as the converged local/cb_only/centralized arms.

We score four variants — {clean,poison} × {mean,robust} — and report the VUS-PR
DAMAGE (clean − poison) under each aggregator. EXPECTED: robust aggregation of the
statistic limits the damage. Statistics defend better than weights.

Usage (smoke):
  CUDA_VISIBLE_DEVICES=1 python scripts/fa_robust.py \
      --dataset wsd_fed --clusters c3 --seeds 0
Full:
  CUDA_VISIBLE_DEVICES=1 python scripts/fa_robust.py \
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

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pipeline"))
sys.path.insert(0, str(REPO / "scripts"))

# Reuse the TESTED mixture_eval infra (cfg build, cb_only pool loader, real-axis
# entity scorer, the _Out score container, the converged-tree root).
from mixture_eval import _build_cfg, _load_pool, _score_entity, _Out, CONVERGED  # noqa: E402
from federated import resolve_clients  # noqa: E402
from stage2 import _flatten_token_indices  # noqa: E402
from data import make_dataloaders  # noqa: E402

OUT_ROOT = REPO / "artifacts" / "fed_eval" / "fa_robust"


# ─── The count prior + Stage2-quacking scorer ───────────────────────────────

class _CountPrior:
    """Per-position count prior. Quacks like a MaskGIT prior for detect.py: it
    exposes `.training = False` and the two scorers detect asks for, returning
    per-token NLL of shape (B,C,F,W) / (1,B,C,F,W) [n_τ = 1: a count prior has no
    masking-rate axis]."""

    def __init__(self, logp: torch.Tensor, C: int, F_: int, W: int):
        self.logp = logp                 # (P, K) log p(codeword | position)
        self.C, self.F, self.W = C, F_, W
        self.training = False

    @torch.no_grad()
    def score_tokens_per_rate(self, tokens: torch.Tensor) -> torch.Tensor:
        B, P = tokens.shape
        pos = torch.arange(P, device=tokens.device)
        nll = -self.logp[pos[None, :], tokens]          # (B, P) advanced-index
        return nll.reshape(1, B, self.C, self.F, self.W)

    @torch.no_grad()
    def score_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.score_tokens_per_rate(tokens).sum(dim=0)


class CountScorer:
    """Quacks like a Stage2System for detect._compute_entities_raw. Tokenises each
    window with the SHARED stage1, then scores each token under the count prior."""

    def __init__(self, shared_stage1, prior: _CountPrior):
        self.stage1 = shared_stage1      # detect asserts .stage1.training is False
        self.prior = prior               # detect asserts .prior.training is False

    @torch.no_grad()
    def score_batch(self, batch: dict, per_rate: bool = False) -> _Out:
        _, indices, _ = self.stage1.encode_tokens(batch["inputs"])
        tokens = _flatten_token_indices(indices).long()
        if per_rate:
            return _Out(self.prior.score_tokens_per_rate(tokens))
        return _Out(self.prior.score_tokens(tokens))


# ─── Per-client per-position count statistics (with optional poisoning) ──────

@torch.no_grad()
def _client_position_counts(stage1, cfg, entity, device, K, poison=None, rng=None):
    """Tokenise a client's TRAIN windows with the shared stage1 and accumulate a
    per-position codeword count matrix counts[P,K] plus window count n. When
    `poison` in {'uniform','shuffle'} the tokens are corrupted BEFORE counting
    (the Byzantine client). Returns (counts (P,K) float, n int, (C,F_,W))."""
    ce = copy.deepcopy(cfg); ce.dataset.entity_id = entity
    loader = make_dataloaders(ce, stage="stage2").train_loader
    counts = None
    pos_idx = None
    C = F_ = W = None
    n = 0
    for batch in loader:
        x = batch["inputs"].to(device, non_blocking=True)
        _, idx, lat = stage1.encode_tokens(x)
        tokens = _flatten_token_indices(idx).long()          # (B, P)
        B, P = tokens.shape
        if counts is None:
            C = int(idx.shape[1]); F_ = int(lat[0]); W = int(lat[1])
            assert C * F_ * W == P, f"C*F*W={C*F_*W} != P={P}"
            counts = torch.zeros(P, K, device=device)
            pos_idx = torch.arange(P, device=device)
        if poison == "uniform":
            tokens = torch.randint(0, K, (B, P), device=device,
                                   generator=rng if rng is not None else None)
        elif poison == "shuffle":
            perm = torch.argsort(torch.rand(B, P, device=device,
                                            generator=rng if rng is not None else None), dim=1)
            tokens = torch.gather(tokens, 1, perm)           # permute positions within each window
        flat = (pos_idx[None, :].expand(B, P) * K + tokens).reshape(-1)
        counts += torch.bincount(flat, minlength=P * K).float().reshape(P, K)
        n += B
    if counts is None:
        return None, 0, (None, None, None)
    return counts, n, (C, F_, W)


# ─── Aggregation + backoff → log p(codeword | position) ─────────────────────

def _robust_aggregate(rates: torch.Tensor, how: str, trim: float) -> torch.Tensor:
    """rates: (Kc, P, K) per-client per-position codeword rate (each client's
    per-position row sums to 1 — scale-free, so an over-reporting Byzantine client
    gains no leverage). Collapse the client axis coordinate-wise. Returns (P, K)."""
    if how == "median":
        return rates.median(dim=0).values
    if how == "trimmed":
        Kc = rates.shape[0]
        srt, _ = torch.sort(rates, dim=0)
        cut = int(np.floor(trim * Kc))
        if 2 * cut >= Kc:                                    # too few clients → median
            return rates.median(dim=0).values
        return srt[cut:Kc - cut].mean(dim=0)
    raise ValueError(f"unknown robust aggregator {how!r}")


def _build_logp(agg: torch.Tensor, lam: float, K: int) -> torch.Tensor:
    """agg: (P,K) non-negative aggregated per-position codeword mass. Returns
    (P,K) log p with a global-marginal backoff at weight `lam`. A small uniform
    floor on the global marginal guarantees finite NLL for any observed token."""
    eps = 1e-12
    pos = agg / agg.sum(dim=1, keepdim=True).clamp_min(eps)          # (P,K)
    g = agg.sum(dim=0)                                               # (K,)
    g = g / g.sum().clamp_min(eps)
    g = 0.999 * g + 0.001 * (1.0 / K)                               # uniform floor
    g = g / g.sum()
    pback = (1.0 - lam) * pos + lam * g[None, :]
    return pback.clamp_min(eps).log()


# ─── Main ───────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="wsd_fed")
    ap.add_argument("--clusters", default="all",
                    help="comma list or 'all' (discovered from converged tree)")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--lam", type=float, default=0.3, help="global-marginal backoff weight")
    ap.add_argument("--poison-idx", type=int, default=0,
                    help="index (in the present-client list) of the Byzantine client")
    ap.add_argument("--poison-mode", choices=["uniform", "shuffle"], default="uniform")
    ap.add_argument("--poison-scale", type=float, default=10.0,
                    help="Byzantine over-reported volume multiplier: how much mass the "
                         "poisoned client injects into the non-robust SUM pool (data-volume "
                         "attack). 1.0 = honest volume. The robust aggregator is scale-free, "
                         "so this lever affects only the non-robust arm.")
    ap.add_argument("--robust-agg", choices=["median", "trimmed"], default="median")
    ap.add_argument("--trim", type=float, default=0.2, help="tail fraction for --robust-agg trimmed")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = _build_cfg(args.dataset, args.batch)
    K = cfg.quantizer.codebook_size
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    if args.clusters == "all":
        clusters = sorted(p.name for p in (CONVERGED / args.dataset).iterdir() if p.is_dir())
    else:
        clusters = [c.strip() for c in args.clusters.split(",") if c.strip()]

    variants = ["clean_mean", "clean_robust", "poison_mean", "poison_robust"]
    print(f"[fa_robust] dataset={args.dataset} clusters={clusters} seeds={seeds} "
          f"lam={args.lam} poison={args.poison_mode}@idx{args.poison_idx} "
          f"robust={args.robust_agg} K={K} device={device}")

    all_records = []
    for cluster in clusters:
        entities = resolve_clients(cfg, None, cluster)
        for seed in seeds:
            # cb_only pool = presence gate + the SHARED stage1 (same token language).
            shared_stage1, pool = _load_pool(args.dataset, cluster, seed, entities, cfg, device)
            if shared_stage1 is None:
                print(f"[fa_robust] SKIP {cluster} seed{seed}: cb_only run missing/incomplete")
                continue
            deep_priors, have = pool
            del deep_priors                                  # unused: free the deep-prior GPU mem
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            pidx = args.poison_idx if 0 <= args.poison_idx < len(have) else 0
            rng = torch.Generator(device=device).manual_seed(1234 + seed)

            # Per-client CLEAN counts (all real) + the Byzantine client's poisoned counts.
            clean_counts, n_list = [], []
            poison_counts = None
            byz_pos = None
            C = F_ = W = None
            for i, e in enumerate(have):
                cnt, n, shape = _client_position_counts(shared_stage1, cfg, e, device, K)
                if cnt is None:
                    continue
                C, F_, W = shape
                if i == pidx:
                    byz_pos = len(clean_counts)
                    pcnt, _, _ = _client_position_counts(
                        shared_stage1, cfg, e, device, K, poison=args.poison_mode, rng=rng)
                    poison_counts = pcnt
                clean_counts.append(cnt); n_list.append(max(1, n))
            if len(clean_counts) < 2 or poison_counts is None or byz_pos is None:
                print(f"[fa_robust] SKIP {cluster} seed{seed}: <2 usable clients")
                del shared_stage1
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            n_t = torch.tensor(n_list, device=device, dtype=torch.float32)        # (Kc,)
            counts_stack = torch.stack(clean_counts, dim=0)                        # (Kc,P,K) raw counts

            # NON-ROBUST aggregator = SUM of raw counts (the exactly-mergeable pooled
            # count; a Byzantine client's over-reported volume leaks straight in).
            clean_sum = counts_stack.sum(dim=0)                                    # (P,K)
            poison_sum = (clean_sum - counts_stack[byz_pos]
                          + args.poison_scale * poison_counts)                     # (P,K)

            # ROBUST aggregator = coordinate-wise median / trimmed-mean over SCALE-FREE
            # per-position rates (each client's per-position row sums to 1, so the
            # attacker's inflated volume gives it no leverage; it is a rank outlier).
            rates = counts_stack / n_t[:, None, None]                              # (Kc,P,K)
            poison_rates = rates.clone()
            poison_rates[byz_pos] = poison_counts / float(n_list[byz_pos])         # ~uniform (scale-free)

            logps = {
                "clean_mean":    _build_logp(clean_sum, args.lam, K),
                "poison_mean":   _build_logp(poison_sum, args.lam, K),
                "clean_robust":  _build_logp(_robust_aggregate(rates, args.robust_agg, args.trim), args.lam, K),
                "poison_robust": _build_logp(_robust_aggregate(poison_rates, args.robust_agg, args.trim), args.lam, K),
            }
            print(f"[fa_robust] {cluster} seed{seed}: {len(clean_counts)} clients {have}; "
                  f"poison={have[pidx]}(mode={args.poison_mode},scale={args.poison_scale}) "
                  f"grid C={C} F={F_} W={W}")

            cluster_vus = {v: [] for v in variants}
            for v in variants:
                prior = _CountPrior(logps[v], C, F_, W)
                scorer = CountScorer(shared_stage1, prior)
                out_dir = OUT_ROOT / args.dataset / cluster / f"seed{seed}" / v
                for e in have:
                    rep = _score_entity(shared_stage1, scorer, cfg, e)
                    d = out_dir / e; d.mkdir(parents=True, exist_ok=True)
                    (d / "report.json").write_text(json.dumps(rep, indent=2))
                    rec = {"_arm": f"fa_robust_{v}", "_cluster": cluster, "_seed": seed, "_entity": e,
                           **{k: float(rep[k]) for k in ("auroc", "auprc", "vus_pr", "pate_f1")
                              if isinstance(rep.get(k), (int, float)) and np.isfinite(rep.get(k))}}
                    all_records.append(rec)
                    vp = rep.get("vus_pr", float("nan"))
                    if isinstance(vp, (int, float)) and np.isfinite(vp):
                        cluster_vus[v].append(float(vp))
                    print(f"  [{v} {cluster} s{seed}] {e}: "
                          f"vus_pr={vp if isinstance(vp,(int,float)) else float('nan'):.3f} "
                          f"auprc={rep.get('auprc', float('nan')):.3f} "
                          f"pate_f1={rep.get('pate_f1', float('nan')):.3f}")

            def _m(v):
                return float(np.mean(cluster_vus[v])) if cluster_vus[v] else float("nan")
            cm, cr, pm, pr = _m("clean_mean"), _m("clean_robust"), _m("poison_mean"), _m("poison_robust")
            print(f"[fa_robust DAMAGE {cluster} s{seed}] cluster-mean VUS-PR | "
                  f"MEAN: clean={cm:.3f} poison={pm:.3f} damage={cm - pm:+.3f} || "
                  f"ROBUST: clean={cr:.3f} poison={pr:.3f} damage={cr - pr:+.3f}")

            del shared_stage1, clean_counts, counts_stack, rates, poison_rates, logps
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rec_path = OUT_ROOT / f"records_{args.dataset}.jsonl"
    with rec_path.open("a") as fh:
        for r in all_records:
            fh.write(json.dumps(r) + "\n")
    print(f"[fa_robust] appended {len(all_records)} records -> {rec_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
