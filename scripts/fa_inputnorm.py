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
E12 — FEDERATED INPUT NORMALIZATION (Federated Analytics).

The deployed detector z-scores every entity by its OWN train mean/std
(`per_entity_standard`: `PerEntityScaler` fits one (mean, std) per entity on that
entity's train segment and applies it to its own train/val/test). This script
does NOT touch weights or the codebook. It probes a FEDERATED / shared input
normalization: a single pooled (mean, std) computed across ALL clients' train
data — an exactly-mergeable sufficient statistic (Σx, Σx², n additive over
clients). Each entity's train+test is then re-normalized with that SHARED stat,
re-tokenized with the SHARED `federated_cb_only` stage-1, and re-scored with the
entity's OWN deep prior (pure prior token-NLL — the deployed score).

  * per_entity     : control — each entity keeps its own (mean, std). With the
                     shared stage-1 + own deep prior this reproduces the
                     converged `federated_cb_only` numbers (sanity anchor).
  * federated_norm : every entity is z-scored by the pooled (mean, std).

Question this gates: does a shared normalization statistic help or hurt? The bet
is that it HURTS rich clients (their own scale is already well-matched to their
prior) but MAY help cold-start / low-data clients whose per-entity moments are
noisy. We report VUS-PR (+ AUPRC / PATE / AUROC) and a `token_shift` diagnostic:
the fraction of test-window token positions whose argmax code index CHANGES when
the input is renormalized with the shared stat instead of the per-entity stat —
a direct measure of how disruptive the shared normalization is to the token grid
the deep prior was trained on.

Everything downstream (rolling assembly, paper threshold, VUS-PR/AUPRC/PATE) is
the EXACT detect.py machinery via mixture_eval._score_entity — so the numbers are
directly comparable to the converged local / cb_only / centralized reports.
  ^^^ [RETRACTED 2026-07-27 — FALSE. See the banner at the top of this file: cb_only shares only
      the codebook, so the common tokenizer this sentence assumes does not exist.]

Usage (smoke):
  CUDA_VISIBLE_DEVICES=1 python scripts/fa_inputnorm.py \
      --dataset wsd_fed --clusters c3 --seeds 0 --limit-entities 2
Usage (full, bounded):
  CUDA_VISIBLE_DEVICES=1 python scripts/fa_inputnorm.py \
      --dataset wsd_fed --clusters c0,c3 --seeds 0,1
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

import data as _data_mod  # noqa: E402  (monkeypatch target: PerEntityScaler)
from config import Config  # noqa: E402
from data import make_dataloaders, load_scaled_records  # noqa: E402
from stage2 import _flatten_token_indices  # noqa: E402
from federated import resolve_clients  # noqa: E402

# Reuse the tested infra from mixture_eval verbatim.
from mixture_eval import _build_cfg, _load_pool, _score_entity, _Out, CONVERGED  # noqa: E402

OUT_ROOT = REPO / "artifacts" / "fed_eval" / "fa_inputnorm"

# ── Shared-normalization monkeypatch ─────────────────────────────────────────
# `make_dataloaders` / `load_scaled_records` do `PerEntityScaler().fit(train)`
# resolving the class name from data.py's module globals at call time. Swapping
# `data.PerEntityScaler` therefore redirects the fit used INSIDE
# mixture_eval._score_entity, with no signature changes anywhere. The federated
# variant overrides every entity's (mean, std) with a single pooled stat.
_ORIG_SCALER = _data_mod.PerEntityScaler
_SHARED = {"mean": None, "std": None}


class _FederatedScaler(_ORIG_SCALER):
    """Per-entity scaler that ignores per-entity moments and assigns the POOLED
    (mean, std) to whatever entity ids appear in the fitted records."""

    def fit(self, records):
        m, s = _SHARED["mean"], _SHARED["std"]
        if m is None or s is None:
            raise RuntimeError("shared (mean, std) not set before federated fit")
        for r in records:
            if r.metadata is None:
                raise ValueError("record missing entity_id")
            self.stats[r.metadata.entity_id] = (m, s)
        return self


def _use_federated(mean: np.ndarray, std: np.ndarray) -> None:
    _SHARED["mean"], _SHARED["std"] = mean, std
    _data_mod.PerEntityScaler = _FederatedScaler


def _use_per_entity() -> None:
    _data_mod.PerEntityScaler = _ORIG_SCALER


# ── Pure-prior scorer (no weights touched) ───────────────────────────────────
class PlainStage2:
    """Quacks like a Stage2System for detect._compute_entities_raw. Tokenises the
    (already-normalized) window with the SHARED stage-1 and scores it under the
    entity's OWN deep prior — the pure-prior deployed score, identical to how the
    `federated_cb_only` reports were produced. All experimental signal comes from
    HOW the input was normalized before it reached `encode_tokens`."""

    def __init__(self, shared_stage1, prior):
        self.stage1 = shared_stage1     # detect asserts .stage1.training is False
        self.prior = prior              # this entity's OWN cb_only deep prior, eval()

    @torch.no_grad()
    def score_batch(self, batch: dict, per_rate: bool = False) -> _Out:
        _, indices, _ = self.stage1.encode_tokens(batch["inputs"])
        tokens = _flatten_token_indices(indices).long()          # (B, N)
        if per_rate:
            return _Out(self.prior.score_tokens_per_rate(tokens))  # (n_τ, B, C, F, W)
        return _Out(self.prior.score_tokens(tokens))               # (B, C, F, W)


# ── Federated pooled normalization statistic (exactly mergeable) ─────────────
def _pooled_stats(cfg: Config, entities: list[str]):
    """POOLED per-channel (mean, std) over ALL clients' TRAIN records, tokenizing
    nothing — a pure Federated-Analytics object. Uses the SAME train records that
    `PerEntityScaler` fits on (load_scaled_records with scaling='none' reproduces
    make_dataloaders' train/val resolution). Accumulates Σx, Σx², n additively
    across clients (order-independent, exactly mergeable), then
    mean = Σx/n, std = sqrt(Σx²/n − mean²) clamped to min_std (matches
    PerEntityScaler's ddof=0 numpy moments and 1e-4 floor)."""
    tot_sum = tot_sq = None
    tot_n = 0
    for e in entities:
        ce = copy.deepcopy(cfg)
        ce.dataset.entity_id = e
        ce.dataset.scaling = "none"                 # raw records, un-normalized
        train_recs, _val, _test = load_scaled_records(ce)
        for r in train_recs:
            X = r.X.astype(np.float64)              # (T, C)
            s = X.sum(axis=0)
            sq = (X * X).sum(axis=0)
            tot_sum = s if tot_sum is None else tot_sum + s
            tot_sq = sq if tot_sq is None else tot_sq + sq
            tot_n += int(X.shape[0])
    if tot_n == 0 or tot_sum is None:
        raise ValueError("no train records to build the pooled normalization statistic")
    mean = tot_sum / tot_n
    var = np.maximum(tot_sq / tot_n - mean * mean, 0.0)
    std = np.maximum(np.sqrt(var), 1e-4)            # PerEntityScaler.min_std
    return mean.astype(np.float32), std.astype(np.float32), tot_n


# ── Token-shift diagnostic ───────────────────────────────────────────────────
@torch.no_grad()
def _tokenize_test(shared_stage1, cfg: Config, entity: str, device) -> torch.Tensor:
    """Tokenise this entity's TEST windows (in order) with the shared stage-1
    under whatever scaler is currently active. shuffle=False on the test loader,
    so the window order is identical regardless of the normalization → the token
    grids from two calls align 1:1 for a per-position comparison."""
    ce = copy.deepcopy(cfg)
    ce.dataset.entity_id = entity
    loader = make_dataloaders(ce, stage="eval").test_loader
    chunks = []
    for batch in loader:
        x = batch["inputs"].to(device, non_blocking=True)
        _, idx, _ = shared_stage1.encode_tokens(x)
        chunks.append(_flatten_token_indices(idx).long().cpu())
    return torch.cat(chunks, dim=0)                 # (n_windows, N)


def _token_shift(shared_stage1, cfg: Config, entity: str, mean, std, device) -> float:
    """Fraction of test-window token positions whose code index changes when the
    input is renormalized with the pooled stat instead of the per-entity stat."""
    _use_per_entity()
    toks_pe = _tokenize_test(shared_stage1, cfg, entity, device)
    _use_federated(mean, std)
    toks_fed = _tokenize_test(shared_stage1, cfg, entity, device)
    _use_per_entity()
    if toks_pe.shape != toks_fed.shape:
        # Should never happen (same records/windowing); guard anyway.
        return float("nan")
    return float((toks_pe != toks_fed).float().mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="wsd_fed")
    ap.add_argument("--clusters", default="all",
                    help="comma list or 'all' (discovered from converged tree)")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--variants", default="per_entity,federated_norm",
                    help="comma list from {per_entity, federated_norm}")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit-entities", type=int, default=0,
                    help="cap entities per (cluster,seed) for a fast smoke; 0 = all")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = _build_cfg(args.dataset, args.batch)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    known = {"per_entity", "federated_norm"}
    bad = [v for v in variants if v not in known]
    if bad:
        raise SystemExit(f"unknown variant(s) {bad}; choose from {sorted(known)}")

    if args.clusters == "all":
        clusters = sorted(p.name for p in (CONVERGED / args.dataset).iterdir()
                          if p.is_dir())
    else:
        clusters = [c.strip() for c in args.clusters.split(",") if c.strip()]

    print(f"[fa_inputnorm] dataset={args.dataset} clusters={clusters} seeds={seeds} "
          f"variants={variants} device={device}")

    all_records = []
    for cluster in clusters:
        entities = resolve_clients(cfg, None, cluster)
        for seed in seeds:
            _use_per_entity()  # always start from the control scaler
            shared_stage1, pool = _load_pool(args.dataset, cluster, seed, entities, cfg, device)
            if shared_stage1 is None:
                print(f"[fa_inputnorm] SKIP {cluster} seed{seed}: cb_only run missing/incomplete")
                continue
            priors, have = pool
            if args.limit_entities > 0:
                have = have[:args.limit_entities]
                priors = priors[:args.limit_entities]
            prior_by_entity = dict(zip(have, priors))

            # POOLED normalization statistic over the clients present (built ONCE).
            mean, std, n_train = _pooled_stats(cfg, have)
            print(f"[fa_inputnorm] {cluster} seed{seed}: {len(have)} clients over {have} | "
                  f"pooled mean={np.round(mean, 4).tolist()} std={np.round(std, 4).tolist()} "
                  f"(n_train={n_train})")

            # Per-entity token-shift under the shared stat (independent of variant).
            token_shift = {e: _token_shift(shared_stage1, cfg, e, mean, std, device)
                           for e in have}

            for variant in variants:
                if variant == "federated_norm":
                    _use_federated(mean, std)
                else:
                    _use_per_entity()
                out_dir = OUT_ROOT / args.dataset / cluster / f"seed{seed}" / variant
                for e in have:
                    scorer = PlainStage2(shared_stage1, prior_by_entity[e])
                    rep = _score_entity(shared_stage1, scorer, cfg, e)
                    rep["token_shift"] = token_shift[e] if variant == "federated_norm" else 0.0
                    rep["pooled_mean"] = mean.tolist()
                    rep["pooled_std"] = std.tolist()
                    d = out_dir / e
                    d.mkdir(parents=True, exist_ok=True)
                    (d / "report.json").write_text(json.dumps(rep, indent=2))
                    rec = {"_arm": f"fa_inputnorm_{variant}", "_cluster": cluster,
                           "_seed": seed, "_entity": e,
                           "token_shift": float(token_shift[e]) if variant == "federated_norm" else 0.0,
                           **{k: float(rep[k]) for k in ("auroc", "auprc", "vus_pr", "pate_f1")
                              if isinstance(rep.get(k), (int, float)) and np.isfinite(rep.get(k))}}
                    all_records.append(rec)
                    print(f"  [{variant} {cluster} s{seed}] {e}: "
                          f"vus_pr={rep.get('vus_pr', float('nan')):.3f} "
                          f"auprc={rep.get('auprc', float('nan')):.3f} "
                          f"pate_f1={rep.get('pate_f1', float('nan')):.3f} "
                          f"tok_shift={token_shift[e] if variant == 'federated_norm' else 0.0:.3f}")

            _use_per_entity()  # restore control before releasing the pool
            del shared_stage1, priors
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Refuse to truncate the ledger to nothing (same guard as mixture_eval:295-303). The file
    # is opened in "w", so a run that scored zero entities would OVERWRITE a good records file
    # with 0 bytes — and a 0-byte ledger is indistinguishable from "never run" for every
    # downstream reducer.
    if not all_records:
        raise SystemExit(
            "[fa_inputnorm] produced 0 records — refusing to write an empty ledger.\n"
            "  Every (cluster, seed) was skipped: the per-arm checkpoints are missing.\n"
            "  Fix the inputs; do not let this overwrite an existing records file."
        )

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rec_path = OUT_ROOT / f"records_{args.dataset}.jsonl"
    with rec_path.open("w") as fh:
        for r in all_records:
            fh.write(json.dumps(r) + "\n")
    print(f"[fa_inputnorm] wrote {len(all_records)} records -> {rec_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
