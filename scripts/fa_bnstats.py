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
E11 — FEDERATED ENCODER BATCHNORM STATS (Federated-Analytics inference probe).

The stage1 encoder is a stack of grouped-conv blocks, each with a BatchNorm2d.
In the `federated_cb_only` arm the *codebook* is federated (Prop.1 sufficient-
statistic merge) but everything else — including every encoder BatchNorm's
running mean/var — is trained LOCALLY per client (this is exactly what FedBN
prescribes: keep BN buffers local). So each client tokenizes test windows through
its OWN normalization, and its deep MaskGIT prior was trained on THAT tokenization.

This probe asks the counterfactual FedBN warns against: what if we had ALSO
federated the BN running stats? For each (cluster, seed) we:

  1. load every client's cb_only stage1 (own encoder + own BN buffers, shared cb),
  2. merge the 12 encoder BatchNorm running stats across clients as a pure
     Federated-Analytics object — a COUNT-WEIGHTED (by n_k = #train windows)
     pool of the per-client first/second moments (law of total variance):
         μ*   = Σ_k w_k μ_k
         σ²*  = Σ_k w_k (σ²_k + μ_k²) − μ*²        (w_k = n_k / Σ n_k)
     (`--bn-merge mean` uses a plain weighted average of σ²_k instead — the
      naive foil that ignores between-client mean shift.)
  3. build, per client, a modified stage1 = its OWN encoder/decoder/codebook with
     ONLY the encoder BN running stats swapped for the merged ones,
  4. re-tokenize + re-score the test/train windows with the client's OWN deep
     prior (which never saw the merged tokenization).

Variants:
  * local_bn      : control = cb_only (own BN buffers). Reproduces the converged
                    federated_cb_only report; token_shift_rate ≡ 0.
  * federated_bn  : merged BN buffers. We report `token_shift_rate` (fraction of
                    test-token positions whose index changed vs local BN) and the
                    detection impact.

Since the priors were trained on the local-BN tokenization, a large token shift
that HURTS detection is the finding: federating BN stats corrupts the tokenizer
each prior was fit to — i.e. FedBN (keep BN local) is right.

Everything downstream (rolling assembly, paper threshold, VUS-PR/AUPRC/PATE) is
the EXACT detect.py machinery via mixture_eval._score_entity, so numbers are
directly comparable to the converged local / cb_only / centralized reports.
  ^^^ [RETRACTED 2026-07-27 — FALSE. See the banner at the top of this file: cb_only shares only
      the codebook, so the common tokenizer this sentence assumes does not exist.]

Usage (smoke):
  CUDA_VISIBLE_DEVICES=1 python scripts/fa_bnstats.py \
      --dataset wsd_fed --clusters c3 --seeds 0
Usage (full):
  CUDA_VISIBLE_DEVICES=1 python scripts/fa_bnstats.py \
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
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pipeline"))
sys.path.insert(0, str(REPO / "scripts"))

from config import Config  # noqa: E402
from data import make_dataloaders  # noqa: E402
from stage1 import load_stage1  # noqa: E402
from stage2 import _flatten_token_indices  # noqa: E402
from federated import resolve_clients  # noqa: E402

# Reuse the tested infra from mixture_eval verbatim.
from mixture_eval import _build_cfg, _load_pool, _score_entity, _Out, CONVERGED  # noqa: E402

OUT_ROOT = REPO / "artifacts" / "fed_eval" / "fa_bnstats"


class SinglePriorStage2:
    """Quacks like a Stage2System for detect._compute_entities_raw: tokenizes with
    ONE (possibly BN-modified) stage1 and scores under ONE deep prior (the client's
    own cb_only prior). Pure prior token-NLL — identical to the deployed cb_only
    score except the encoder BN buffers may have been swapped for merged ones."""

    def __init__(self, stage1, prior):
        self.stage1 = stage1          # detect asserts .stage1.training is False
        self.prior = prior            # this entity's OWN cb_only deep prior, eval()

    @torch.no_grad()
    def score_batch(self, batch: dict, per_rate: bool = False) -> _Out:
        _, indices, _ = self.stage1.encode_tokens(batch["inputs"])
        tokens = _flatten_token_indices(indices).long()               # (B, N)
        s = (self.prior.score_tokens_per_rate(tokens) if per_rate
             else self.prior.score_tokens(tokens))
        return _Out(s)


def _encoder_bn_modules(encoder: nn.Module) -> dict[str, nn.Module]:
    """Name -> BatchNorm module for every BN in the ENCODER (the only BN that
    affects tokenization: encode_tokens runs transform -> encoder -> quantizer,
    the decoder BN never touches the tokens). Names are stable across clients
    (identical architecture) so they align 1:1 for the merge."""
    return {n: m for n, m in encoder.named_modules()
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.BatchNorm1d))
            and m.running_mean is not None and m.running_var is not None}


def _pool_bn_stats(bn_by_e: dict[str, dict[str, nn.Module]], have: list[str],
                   weights: torch.Tensor, merge: str) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Federated merge of the encoder BN running stats across clients, weighted by
    w_k (= n_k / Σ n_k). For each BN position returns (pooled_mean, pooled_var).

      pooled: μ* = Σ w_k μ_k ; σ²* = Σ w_k (σ²_k + μ_k²) − μ*²  (exact pooled 2nd
              moment — a sufficient-statistic merge; captures cross-client shift).
      mean:   plain weighted average of σ²_k (ignores between-client mean shift).
    """
    names = list(bn_by_e[have[0]].keys())
    w = weights.view(-1, 1)                                    # (Kc, 1)
    pooled: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name in names:
        means = torch.stack([bn_by_e[e][name].running_mean.detach().float() for e in have])  # (Kc, D)
        vars = torch.stack([bn_by_e[e][name].running_var.detach().float() for e in have])    # (Kc, D)
        pooled_mean = (w * means).sum(dim=0)
        if merge == "mean":
            pooled_var = (w * vars).sum(dim=0).clamp_min(0.0)
        else:  # "pooled" — law of total variance
            pooled_ex2 = (w * (vars + means.pow(2))).sum(dim=0)
            pooled_var = (pooled_ex2 - pooled_mean.pow(2)).clamp_min(0.0)
        pooled[name] = (pooled_mean, pooled_var)
    return pooled


def _apply_bn_stats(stage1, pooled: dict[str, tuple[torch.Tensor, torch.Tensor]]):
    """Deep-copy `stage1` and overwrite ONLY its encoder BN running_mean/var with
    the merged stats. Conv weights, codebook, decoder — all stay LOCAL."""
    m = copy.deepcopy(stage1)
    bn = _encoder_bn_modules(m.encoder)
    with torch.no_grad():
        for name, (pm, pv) in pooled.items():
            bn[name].running_mean.copy_(pm.to(bn[name].running_mean.dtype))
            bn[name].running_var.copy_(pv.to(bn[name].running_var.dtype))
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@torch.no_grad()
def _token_shift_rate(s1_local, s1_fed, loader, device) -> float:
    """Fraction of test-token positions whose codebook index CHANGES when the
    encoder BN buffers are swapped from local to merged (same encoder weights)."""
    ndiff, ntot = 0, 0
    for batch in loader:
        x = batch["inputs"].to(device, non_blocking=True)
        _, il, _ = s1_local.encode_tokens(x)
        _, iff, _ = s1_fed.encode_tokens(x)
        tl = _flatten_token_indices(il).long()
        tf = _flatten_token_indices(iff).long()
        ndiff += int((tl != tf).sum().item())
        ntot += int(tl.numel())
    return ndiff / max(1, ntot)


def _client_nwin(cfg: Config, entities: list[str]) -> dict[str, int]:
    """n_k = number of stage1 train windows per client (the BN running-stat weight —
    what each client's BN buffers were accumulated over during stage1 training)."""
    out = {}
    for e in entities:
        ce = copy.deepcopy(cfg); ce.dataset.entity_id = e
        out[e] = int(len(make_dataloaders(ce, stage="stage1").train_dataset))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="wsd_fed")
    ap.add_argument("--clusters", default="all",
                    help="comma list or 'all' (discovered from converged tree)")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--bn-merge", default="pooled", choices=["pooled", "mean"],
                    help="federated BN variance merge: 'pooled' (law of total variance, "
                         "exact 2nd-moment) or 'mean' (naive weighted avg of var).")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = _build_cfg(args.dataset, args.batch)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    if args.clusters == "all":
        clusters = sorted(p.name for p in (CONVERGED / args.dataset).iterdir()
                          if p.is_dir())
    else:
        clusters = [c.strip() for c in args.clusters.split(",") if c.strip()]

    print(f"[fa_bnstats] dataset={args.dataset} clusters={clusters} seeds={seeds} "
          f"bn_merge={args.bn_merge} device={device}")

    variants = ["local_bn", "federated_bn"]
    all_records = []
    for cluster in clusters:
        entities = resolve_clients(cfg, None, cluster)
        for seed in seeds:
            shared_stage1, pool = _load_pool(args.dataset, cluster, seed, entities, cfg, device)
            if shared_stage1 is None:
                print(f"[fa_bnstats] SKIP {cluster} seed{seed}: cb_only run missing/incomplete")
                continue
            priors, have = pool
            prior_by_e = dict(zip(have, priors))
            cb = CONVERGED / args.dataset / cluster / f"seed{seed}" / "federated_cb_only"

            # Per-client stage1 (own encoder + own BN buffers). Reuse the already-loaded
            # have[0] stage1 from _load_pool; load the rest with the same (1,C,W) example.
            c0 = copy.deepcopy(cfg); c0.dataset.entity_id = have[0]
            example = next(iter(make_dataloaders(c0, stage="eval").train_loader))["inputs"][:1].cpu()
            stage1_by_e = {have[0]: shared_stage1}
            for e in have[1:]:
                s1 = load_stage1(cb / e / "stage1.ckpt", cfg, example, device=device)
                s1.eval()
                stage1_by_e[e] = s1

            # n_k weights + federated BN merge.
            nwin = _client_nwin(cfg, have)
            n = torch.tensor([nwin[e] for e in have], dtype=torch.float32, device=device)
            weights = n / n.sum()
            bn_by_e = {e: _encoder_bn_modules(stage1_by_e[e].encoder) for e in have}
            n_bn = len(bn_by_e[have[0]])
            pooled = _pool_bn_stats(bn_by_e, have, weights, args.bn_merge)
            print(f"[fa_bnstats] {cluster} seed{seed}: {len(have)} clients {have} "
                  f"n_k={[nwin[e] for e in have]} bn_layers={n_bn}")

            for variant in variants:
                out_dir = OUT_ROOT / args.dataset / cluster / f"seed{seed}" / variant
                for e in have:
                    if variant == "local_bn":
                        s1_use = stage1_by_e[e]
                        shift = 0.0
                    else:
                        s1_use = _apply_bn_stats(stage1_by_e[e], pooled)
                        ce = copy.deepcopy(cfg); ce.dataset.entity_id = e
                        test_loader = make_dataloaders(ce, stage="eval").test_loader
                        shift = _token_shift_rate(stage1_by_e[e], s1_use, test_loader, device)

                    scorer = SinglePriorStage2(s1_use, prior_by_e[e])
                    rep = _score_entity(s1_use, scorer, cfg, e)
                    rep["token_shift_rate"] = float(shift)
                    rep["bn_merge"] = args.bn_merge

                    d = out_dir / e; d.mkdir(parents=True, exist_ok=True)
                    (d / "report.json").write_text(json.dumps(rep, indent=2))
                    rec = {"_arm": f"fa_bnstats_{variant}", "_cluster": cluster,
                           "_seed": seed, "_entity": e, "token_shift_rate": float(shift),
                           **{k: float(rep[k]) for k in ("auroc", "auprc", "vus_pr", "pate_f1")
                              if isinstance(rep.get(k), (int, float)) and np.isfinite(rep.get(k))}}
                    all_records.append(rec)
                    print(f"  [{variant} {cluster} s{seed}] {e}: "
                          f"vus_pr={rep.get('vus_pr', float('nan')):.3f} "
                          f"auprc={rep.get('auprc', float('nan')):.3f} "
                          f"pate_f1={rep.get('pate_f1', float('nan')):.3f} "
                          f"tok_shift={shift:.3f}")

                    if variant == "federated_bn":
                        del s1_use

            del stage1_by_e, priors, bn_by_e, pooled
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Refuse to truncate the ledger to nothing (same guard as mixture_eval:295-303). The file
    # is opened in "w", so a run that scored zero entities would OVERWRITE a good records file
    # with 0 bytes — and a 0-byte ledger is indistinguishable from "never run" for every
    # downstream reducer.
    if not all_records:
        raise SystemExit(
            "[fa_bnstats] produced 0 records — refusing to write an empty ledger.\n"
            "  Every (cluster, seed) was skipped: the per-arm checkpoints are missing.\n"
            "  Fix the inputs; do not let this overwrite an existing records file."
        )

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rec_path = OUT_ROOT / f"records_{args.dataset}.jsonl"
    with rec_path.open("w") as fh:
        for r in all_records:
            fh.write(json.dumps(r) + "\n")
    print(f"[fa_bnstats] wrote {len(all_records)} records -> {rec_path}")

    # Finding summary: mean token shift + Δvus_pr(federated_bn - local_bn).
    loc = {(r["_cluster"], r["_seed"], r["_entity"]): r for r in all_records
           if r["_arm"] == "fa_bnstats_local_bn"}
    fed = [r for r in all_records if r["_arm"] == "fa_bnstats_federated_bn"]
    if fed:
        shifts = [r["token_shift_rate"] for r in fed]
        dv = [r.get("vus_pr", np.nan) - loc.get((r["_cluster"], r["_seed"], r["_entity"]), {}).get("vus_pr", np.nan)
              for r in fed]
        dv = [x for x in dv if np.isfinite(x)]
        print(f"[fa_bnstats] FINDING: mean token_shift_rate={np.mean(shifts):.3f} "
              f"(max={np.max(shifts):.3f}); mean Δvus_pr(fed_bn - local_bn)={np.mean(dv):+.4f}"
              if dv else f"[fa_bnstats] FINDING: mean token_shift_rate={np.mean(shifts):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
