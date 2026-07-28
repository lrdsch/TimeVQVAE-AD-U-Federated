"""
E5 — COLD-START / DATA SCARCITY (flagship Federated-Analytics win).

Simulate a data-poor silo: for each entity we cap its TRAIN windows to a
fraction `frac ∈ {0.1, 0.25, 0.5}` (random, seeded Subset of the entity's own
train_dataset) and ask what a fresh silo can build from that little data, with
vs without help from the cluster's already-federated tokenizer + neighbours.

Three things per (entity, frac):

  (a) local_scarce  — a FULL fresh stack (own stage1 codebook + stage2 prior)
                      trained ONLY on the capped data. The "no-help" baseline.

  (b) FA_assisted   — reuse the cluster's `federated_cb_only` SHARED stage1
                      (frozen, the already-federated tokenizer) + train a stage2
                      prior on the capped data + BACK OFF (λ=0.3, E1-style) to a
                      POOLED per-position count prior built from the OTHER
                      clients' FULL train data (tokenised with the shared
                      tokenizer). This is data the silo can obtain at NO extra
                      training cost — only additive token counts cross the wire.

  (c) oracle        — the converged `local` (FULL data) report.json. Upper bound.

Deployed anomaly score is PURE prior token-NLL, scored through detect.py's real
rolling/threshold/VUS machinery (`_score_entity`), so every number is on the
same axis as the converged local / cb_only / centralized reports.

EXPECTED: FA_assisted >> local_scarce, and the gap WIDENS as `frac` shrinks —
the flagship cold-start FA win.

Smoke:
  CUDA_VISIBLE_DEVICES=1 python scripts/fa_coldstart.py \
      --dataset wsd_fed --clusters c3 --seeds 0 --fracs 0.25 \
      --s1-epochs 2 --s2-epochs 2

Full (wsd):
  CUDA_VISIBLE_DEVICES=1 python scripts/fa_coldstart.py \
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
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pipeline"))
sys.path.insert(0, str(REPO / "scripts"))

# Tested infra (read scripts/mixture_eval.py / pipeline/federated_eval.py).
from mixture_eval import _build_cfg, _load_pool, _score_entity, _Out, CONVERGED  # noqa: E402
from federated import resolve_clients  # noqa: E402
from federated_eval import (  # noqa: E402
    _materialized_stage1, _train_stage1, _build_train_stage2, _loaders_for,
)
from data import _build_loader  # noqa: E402
from stage2 import _flatten_token_indices  # noqa: E402
from utils import seed_everything  # noqa: E402

OUT_ROOT = REPO / "artifacts" / "fed_eval" / "fa_coldstart"


# ─── FA_assisted scorer: deep prior backed off to a pooled count prior ────────

class BackoffStage2:
    """Quacks like a Stage2System for detect._compute_entities_raw.

    Tokenises with the SHARED (frozen) stage1, scores each token under the
    silo's own deep prior, and mixes — in PROBABILITY space, per masking-rate τ —
    with a pooled per-position count prior over the K codewords built from the
    OTHER clients' full data:

        p_mix(tok) = (1-λ)·p_deep,τ(tok) + λ·p_count(tok)
        nll_mix,τ  = -log p_mix                         (per τ, then τ-summed)

    Working per-τ keeps deep and count on the SAME single-NLL scale (the deep
    `score_tokens` output is a SUM over τ rates, so mixing that summed score with
    a single-τ count NLL would be scale-mismatched and let the count prior swamp
    the deep one)."""

    def __init__(self, shared_stage1, prior, count_logp: torch.Tensor, lam: float,
                 C: int, Fdim: int, W: int):
        self.stage1 = shared_stage1              # detect asserts .stage1.training is False
        self.prior = prior                       # detect asserts .prior.training is False
        self.count_logp = count_logp             # (L, K) log p(token | position)
        self.lam = float(lam)
        self._C, self._F, self._W = C, Fdim, W
        self._la = math.log(1.0 - lam) if lam < 1.0 else -math.inf
        self._lb = math.log(lam) if lam > 0.0 else -math.inf

    @torch.no_grad()
    def _count_nll(self, tokens: torch.Tensor) -> torch.Tensor:
        B, L = tokens.shape
        pos = torch.arange(L, device=tokens.device)
        logp = self.count_logp[pos[None, :], tokens]      # (B, L)
        return (-logp).reshape(B, self._C, self._F, self._W)

    @torch.no_grad()
    def score_batch(self, batch: dict, per_rate: bool = False) -> _Out:
        _, indices, _ = self.stage1.encode_tokens(batch["inputs"])
        tokens = _flatten_token_indices(indices).long()   # (B, L)
        deep = self.prior.score_tokens_per_rate(tokens)   # (n_τ, B, C, F, W) per-token NLL
        if self.lam <= 0.0:
            return _Out(deep if per_rate else deep.sum(dim=0))
        cnll = self._count_nll(tokens)                    # (B, C, F, W)
        cnll_b = cnll.unsqueeze(0).expand_as(deep)        # broadcast across τ
        # -log[(1-λ)e^{-deep} + λ e^{-count}] = -logaddexp(la - deep, lb - count)
        mix = -torch.logaddexp(self._la - deep, self._lb - cnll_b)
        return _Out(mix if per_rate else mix.sum(dim=0))


# ─── pooled per-position count prior (the exactly-mergeable neighbour help) ────

@torch.no_grad()
def _position_counts(shared_stage1, loader, device, K: int) -> torch.Tensor:
    """RAW per-position codeword counts (L, K) over ALL windows in `loader`,
    tokenised with the shared codebook. Additive across clients ⇒ exact merge."""
    counts = None
    for batch in loader:
        x = batch["inputs"].to(device, non_blocking=True)
        _, idx, _ = shared_stage1.encode_tokens(x)
        tokens = _flatten_token_indices(idx).long()       # (B, L)
        oh = F.one_hot(tokens, num_classes=K).float()     # (B, L, K)
        counts = oh.sum(0) if counts is None else counts + oh.sum(0)
    return counts                                         # (L, K)


def _backoff_logp(other_counts: list[torch.Tensor], K: int) -> torch.Tensor:
    """Sum neighbour counts, Laplace-smooth, normalise → (L, K) log-prob."""
    total = sum(other_counts) + 1.0                       # +1 Laplace
    return (total / total.sum(dim=1, keepdim=True)).log()


# ─── data capping ─────────────────────────────────────────────────────────────

def _capped_loader(cfg, train_dataset, frac: float, seed: int, entity: str):
    """Subset of `train_dataset` to `frac·len` random (seeded) windows."""
    n = len(train_dataset)
    k = max(1, int(round(frac * n)))
    g = torch.Generator().manual_seed(hash((seed, entity, round(frac, 4))) & 0x7FFFFFFF)
    idx = torch.randperm(n, generator=g)[:k].tolist()
    sub = torch.utils.data.Subset(train_dataset, idx)
    # num_workers=0: capped sets are tiny and this avoids worker churn across the
    # many short-lived loaders one run creates.
    loader = _build_loader(sub, cfg.dataset.batch_size_stage1, 0, shuffle=True)
    return loader, k, n


# ─── report / record IO ───────────────────────────────────────────────────────

_METRICS = ("auroc", "auprc", "vus_pr", "pate_f1")


def _write_report(ds: str, cluster: str, seed: int, variant: str, entity: str, rep: dict) -> None:
    d = OUT_ROOT / ds / cluster / f"seed{seed}" / variant / entity
    d.mkdir(parents=True, exist_ok=True)
    (d / "report.json").write_text(json.dumps(rep, indent=2))


def _append_record(recfh, ds: str, variant: str, cluster: str, seed: int, entity: str,
                   rep: dict, frac, kind: str) -> dict:
    rec = {"_arm": f"fa_coldstart_{variant}", "_cluster": cluster, "_seed": seed,
           "_entity": entity, "_frac": frac, "_kind": kind}
    for k in _METRICS:
        v = rep.get(k)
        if isinstance(v, (int, float)) and np.isfinite(v):
            rec[k] = float(v)
    recfh.write(json.dumps(rec) + "\n")
    recfh.flush()
    return rec


def _pr(variant: str, cluster: str, seed: int, entity: str, rep: dict) -> None:
    print(f"  [{variant} {cluster} s{seed}] {entity}: "
          f"vus_pr={rep.get('vus_pr', float('nan')):.3f} "
          f"auprc={rep.get('auprc', float('nan')):.3f} "
          f"pate_f1={rep.get('pate_f1', float('nan')):.3f}", flush=True)


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="wsd_fed")
    ap.add_argument("--clusters", default="all",
                    help="comma list or 'all' (discovered from the converged tree)")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--fracs", default="0.1,0.25,0.5",
                    help="train-window keep fractions to sweep")
    ap.add_argument("--lam", type=float, default=0.3, help="count-prior backoff weight")
    ap.add_argument("--s1-epochs", type=int, default=20)
    ap.add_argument("--s2-epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-entities", type=int, default=0,
                    help="cap SCORED entities per (cluster,seed) for smoke tests (0 = all); "
                         "neighbour count priors still use every client")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = _build_cfg(args.dataset, args.batch)
    K = cfg.quantizer.codebook_size
    lr = cfg.training.lr
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    fracs = [float(f) for f in args.fracs.split(",") if f.strip()]

    if args.clusters == "all":
        clusters = sorted(p.name for p in (CONVERGED / args.dataset).iterdir() if p.is_dir())
    else:
        clusters = [c.strip() for c in args.clusters.split(",") if c.strip()]

    print(f"[fa_coldstart] dataset={args.dataset} clusters={clusters} seeds={seeds} "
          f"fracs={fracs} lam={args.lam} s1e={args.s1_epochs} s2e={args.s2_epochs} device={device}",
          flush=True)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    recfh = (OUT_ROOT / f"records_{args.dataset}.jsonl").open("a")

    for cluster in clusters:
        entities = resolve_clients(cfg, None, cluster)
        for seed in seeds:
            # Shared (already-federated) tokenizer from cb_only. Skip if missing.
            shared_stage1, pool = _load_pool(args.dataset, cluster, seed, entities, cfg, device)
            if shared_stage1 is None:
                print(f"[fa_coldstart] SKIP {cluster} seed{seed}: cb_only run missing/incomplete",
                      flush=True)
                continue
            priors, have = pool
            del priors                                       # not needed: we retrain our own prior
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[fa_coldstart] {cluster} seed{seed}: shared tokenizer over {have}", flush=True)

            # Full loaders for every present client (used for both capping source
            # and for building the neighbour count prior).
            data_by_e = _loaders_for(cfg, have)

            # Precompute each client's RAW per-position counts ONCE (from full data,
            # shared tokenizer). Neighbour help for entity e = sum over o != e.
            client_counts = {e: _position_counts(shared_stage1, data_by_e[e].train_loader, device, K)
                             for e in have}

            scored = have if args.max_entities <= 0 else have[:args.max_entities]
            for e in scored:
                # (c) oracle — converged local (full data) report.json.
                orep_path = CONVERGED / args.dataset / cluster / f"seed{seed}" / "local" / e / "report.json"
                if orep_path.exists():
                    orep = json.loads(orep_path.read_text())
                    _write_report(args.dataset, cluster, seed, "oracle", e, orep)
                    _append_record(recfh, args.dataset, "oracle", cluster, seed, e, orep, None, "oracle")
                    _pr("oracle", cluster, seed, e, orep)
                else:
                    print(f"  [oracle {cluster} s{seed}] {e}: MISSING local report.json", flush=True)

                # Neighbour backoff log-prob for this entity (others' full counts).
                others = [client_counts[o] for o in have if o != e]
                count_logp = _backoff_logp(others, K) if others else None

                for frac in fracs:
                    tag = f"f{int(round(frac * 100))}"
                    cap_loader, kept, ntot = _capped_loader(cfg, data_by_e[e].train_dataset, frac, seed, e)
                    ex = next(iter(cap_loader))["inputs"][:1]
                    print(f"  -- {e} {tag}: {kept}/{ntot} train windows", flush=True)

                    # (a) local_scarce — fresh full stack on the capped data.
                    seed_everything(seed)
                    s1 = _materialized_stage1(cfg, ex, device, collect_stats=False)
                    _train_stage1(s1, cap_loader, args.s1_epochs, lr, device, tag=f"{e}-{tag}-loc")
                    s2 = _build_train_stage2(cfg, s1, cap_loader, args.s2_epochs, lr, device, ex,
                                             tag=f"{e}-{tag}-loc")
                    s2.stage1.eval(); s2.prior.eval()
                    variant = f"local_scarce_{tag}"
                    rep = _score_entity(s2.stage1, s2, cfg, e)
                    _write_report(args.dataset, cluster, seed, variant, e, rep)
                    _append_record(recfh, args.dataset, variant, cluster, seed, e, rep, frac, "local_scarce")
                    _pr(variant, cluster, seed, e, rep)
                    del s1, s2
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    # (b) FA_assisted — shared frozen tokenizer + capped-data prior + neighbour backoff.
                    seed_everything(seed)
                    s2fa = _build_train_stage2(cfg, shared_stage1, cap_loader, args.s2_epochs, lr,
                                               device, ex, tag=f"{e}-{tag}-fa")
                    s2fa.prior.eval()
                    Cd, Fd, Wd = (s2fa.prior.latent_channels, s2fa.prior.latent_freq,
                                  s2fa.prior.latent_time)
                    if count_logp is None:            # single-client cluster: no neighbour help
                        scorer = BackoffStage2(shared_stage1, s2fa.prior, None, 0.0, Cd, Fd, Wd)
                    else:
                        scorer = BackoffStage2(shared_stage1, s2fa.prior, count_logp, args.lam, Cd, Fd, Wd)
                    variant = f"fa_assisted_{tag}"
                    rep = _score_entity(shared_stage1, scorer, cfg, e)
                    _write_report(args.dataset, cluster, seed, variant, e, rep)
                    _append_record(recfh, args.dataset, variant, cluster, seed, e, rep, frac, "fa_assisted")
                    _pr(variant, cluster, seed, e, rep)
                    del s2fa, scorer
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            del shared_stage1, data_by_e, client_counts
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    recfh.close()
    print(f"[fa_coldstart] done -> {OUT_ROOT / f'records_{args.dataset}.jsonl'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
