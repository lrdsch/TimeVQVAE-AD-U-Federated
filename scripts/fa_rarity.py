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
E2 — FEDERATED RARITY WEIGHTS (Federated Analytics).

The deployed anomaly score is the pure MaskGIT prior token-NLL. Globally-rare
tokens are anomaly-adjacent: a token that is rare across the WHOLE federation is
exactly the kind of pattern a per-client detector should be most sensitive to.

Inverse-sqrt-frequency loss weighting (rare token → higher training weight, à la
`Stage2System.set_token_weights_from_tokens`) normally derives its frequencies
from a client's OWN local token counts. Here we federate the count statistic:

  * local_weights     (CONTROL) : weights from THIS client's local token counts.
  * federated_weights (TREATMENT): weights from the POOLED token counts summed
    across ALL clients in the cluster (an exactly-mergeable additive statistic,
    like the shared-codebook suff-stat merge). Globally-rare tokens get upweighted
    everywhere, even on clients that rarely see them.

In BOTH arms each client's deep prior is RETRAINED from scratch on its own data,
on the SHARED (federated_cb_only) frozen tokenizer, so all clients speak the same
token language. Only the per-token loss-weight vector differs between arms.

NOTE: the deployed prior is MaskGITPrior3DPos, which is NOT a subclass of the
weight-aware MaskGITPrior — its forward() uses plain (unweighted) cross-entropy
and `set_token_weights_from_tokens` is a no-op for it. So we replicate its
forward exactly (same masking, same _logits) and inject the SAME weighted-CE
convention the base class uses: F.cross_entropy(logits, target, weight=w).

Everything downstream (rolling assembly, paper threshold, VUS-PR / AUPRC / PATE)
is the EXACT detect.py machinery via `_score_entity`, so the numbers land on the
same axis as the converged local / cb_only / centralized reports.
  ^^^ [RETRACTED 2026-07-27 — FALSE. See the banner at the top of this file: cb_only shares only
      the codebook, so the common tokenizer this sentence assumes does not exist.]

Usage (full scope):
  CUDA_VISIBLE_DEVICES=1 python scripts/fa_rarity.py \
      --dataset wsd_fed --clusters all --seeds 0,1,2 --s2-epochs 18
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

# Tested infra reused verbatim.
from mixture_eval import _build_cfg, _load_pool, _score_entity, CONVERGED  # noqa: E402
from federated import resolve_clients  # noqa: E402
from stage2 import Stage2System, _flatten_token_indices  # noqa: E402
from data import make_dataloaders  # noqa: E402
from utils import seed_everything  # noqa: E402

OUT_ROOT = REPO / "artifacts" / "fed_eval" / "fa_rarity"


# ─── Rarity weight vector (identical convention to set_token_weights_from_tokens) ─

def weights_from_counts(counts: torch.Tensor, K: int) -> torch.Tensor:
    """Inverse-sqrt-frequency weights from a K-length count vector.

    Matches Stage2System.set_token_weights_from_tokens: active tokens get
    1/sqrt(freq) renormalised to mean 1; unseen tokens get weight 0."""
    counts = counts.float()
    freq = counts / counts.sum().clamp(min=1)
    active = freq > 0
    w = torch.ones(K)
    w[active] = 1.0 / freq[active].sqrt()
    w[active] = w[active] / w[active].mean()
    w[~active] = 0.0
    return w


@torch.no_grad()
def client_token_counts(shared_stage1, cfg, entity: str, device, K: int) -> torch.Tensor:
    """K-length token occupancy over a client's stage2 train windows, tokenised
    with the SHARED frozen codebook. Additive across clients (exact merge)."""
    ce = copy.deepcopy(cfg)
    ce.dataset.entity_id = entity
    loader = make_dataloaders(ce, stage="stage2").train_loader
    counts = torch.zeros(K, device=device)
    for batch in loader:
        x = batch["inputs"].to(device, non_blocking=True)
        _, idx, _ = shared_stage1.encode_tokens(x)
        t = _flatten_token_indices(idx).long().reshape(-1)
        counts += torch.bincount(t, minlength=K).float()
    return counts.cpu()


# ─── Train a fresh deep prior on ONE client with a given rarity-weight vector ──

def train_prior(shared_stage1, cfg, entity: str, weights: torch.Tensor,
                device, n_epochs: int, base_seed: int) -> Stage2System:
    """Retrain a MaskGITPrior3DPos on `entity`'s local data over the shared frozen
    stage1. `weights` is the per-token loss-weight vector (local or pooled). Both
    arms take the identical seed → identical init/masks/data-order, isolating the
    weight vector as the only difference. Returns the eval-mode Stage2System."""
    seed_everything(base_seed)

    model = Stage2System(cfg, shared_stage1)     # freezes + evals stage1 internally
    model.to(device)

    ce = copy.deepcopy(cfg)
    ce.dataset.entity_id = entity
    data = make_dataloaders(ce, stage="stage2")
    example = next(iter(data.train_loader))["inputs"][:1]
    model.materialize(example.to(device))        # builds prior weights on `device`
    model.to(device)
    model.stage1.eval()
    weights = weights.to(device)

    opt = torch.optim.AdamW(
        model.prior.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        fused=(torch.device(device).type == "cuda"),
    )
    batches_per_epoch = max(1, len(data.train_loader))
    max_steps = max(1, n_epochs * batches_per_epoch)
    warmup_steps = max(1, int(max_steps * cfg.training.warmup_rate))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        prog = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # fp16 on sm_75 (Turing has no bf16 tensor cores); AMP off → bit-exact fp32.
    use_amp = bool(getattr(cfg.training, "amp", False)) and torch.cuda.is_available()
    amp_dtype = (torch.bfloat16 if use_amp and torch.cuda.get_device_capability()[0] >= 8
                 else torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype is torch.float16)

    ls = float(getattr(model.prior, "label_smoothing", 0.0))

    # Re-seed: materialize() ran a throwaway prior.forward that consumed RNG.
    seed_everything(base_seed)

    best_val = float("inf")
    best_state = None
    step = 0
    for epoch in range(n_epochs):
        model.prior.train()
        for batch in data.train_loader:
            batch["inputs"] = batch["inputs"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                tokens = model.tokens_from_batch(batch)          # frozen stage1 encode
                model._inform_latent_shape()
                masked, mask = model.prior._mask_tokens(tokens)
                logits = model.prior._logits(masked)
                target = tokens[mask]
                masked_logits = logits[mask]
                if masked_logits.numel() > 0:
                    loss = F.cross_entropy(masked_logits, target,
                                           weight=weights, label_smoothing=ls)
                else:
                    loss = logits.sum() * 0.0
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            prev_scale = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            if scaler.get_scale() >= prev_scale:    # LR advances only on a real step
                sched.step()
            step += 1

        # Model-selection val loss: UNWEIGHTED prior NLL (fp32), same for both arms.
        model.prior.eval()
        vloss = []
        with torch.no_grad():
            for batch in data.val_loader:
                batch["inputs"] = batch["inputs"].to(device, non_blocking=True)
                tokens = model.tokens_from_batch(batch)
                model._inform_latent_shape()
                out = model.prior(tokens)
                vloss.append(out.loss.item())
        vl = float(np.mean(vloss)) if vloss else float("nan")
        if vl == vl and vl < best_val:
            best_val = vl
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.prior.state_dict().items()}
        print(f"      [train {entity}] epoch={epoch:3d} step={step:5d} "
              f"val/nll={vl:.4f} lr={opt.param_groups[0]['lr']:.2e}")

    if best_state is not None:
        model.prior.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="wsd_fed")
    ap.add_argument("--clusters", default="all",
                    help="comma list or 'all' (discovered from converged tree)")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--variants", default="local_weights,federated_weights",
                    help="subset of {local_weights, federated_weights}")
    ap.add_argument("--s2-epochs", type=int, default=18, help="stage2 epochs per client (smoke: 2)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-entities", type=int, default=0,
                    help="cap #entities trained/scored per (cluster,seed) for smoke; 0 = all")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = _build_cfg(args.dataset, args.batch)
    K = cfg.quantizer.codebook_size
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variants:
        if v not in ("local_weights", "federated_weights"):
            raise SystemExit(f"unknown variant {v!r}")

    if args.clusters == "all":
        clusters = sorted(p.name for p in (CONVERGED / args.dataset).iterdir() if p.is_dir())
    else:
        clusters = [c.strip() for c in args.clusters.split(",") if c.strip()]

    print(f"[fa_rarity] dataset={args.dataset} clusters={clusters} seeds={seeds} "
          f"variants={variants} s2_epochs={args.s2_epochs} K={K} device={device}")

    all_records = []
    for cluster in clusters:
        entities = resolve_clients(cfg, None, cluster)
        for seed in seeds:
            shared_stage1, pool = _load_pool(args.dataset, cluster, seed, entities, cfg, device)
            if shared_stage1 is None:
                print(f"[fa_rarity] SKIP {cluster} seed{seed}: cb_only run missing/incomplete")
                continue
            priors, have = pool
            del priors                                  # we retrain fresh; free the loaded deep priors
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[fa_rarity] {cluster} seed{seed}: shared tokenizer over {len(have)} clients {have}")

            # POOLED token counts = additive over ALL clients (federated statistic),
            # plus each client's own counts (for the local-weights control).
            per_client = {e: client_token_counts(shared_stage1, cfg, e, device, K) for e in have}
            pooled_counts = torch.stack([per_client[e] for e in have], 0).sum(0)
            pooled_weights = weights_from_counts(pooled_counts, K)
            n_active_pool = int((pooled_counts > 0).sum())
            print(f"[fa_rarity] {cluster} seed{seed}: pooled active tokens={n_active_pool}/{K}")

            train_ents = have if args.max_entities <= 0 else have[:args.max_entities]
            for vi, e in enumerate(train_ents):
                base_seed = 1000 * seed + 31 * vi + 7    # same across arms for this (seed,entity)
                local_weights = weights_from_counts(per_client[e], K)
                for variant in variants:
                    w = local_weights if variant == "local_weights" else pooled_weights
                    model = train_prior(shared_stage1, cfg, e, w, device,
                                        args.s2_epochs, base_seed)
                    rep = _score_entity(shared_stage1, model, cfg, e)
                    d = OUT_ROOT / args.dataset / cluster / f"seed{seed}" / variant / e
                    d.mkdir(parents=True, exist_ok=True)
                    (d / "report.json").write_text(json.dumps(rep, indent=2))
                    rec = {"_arm": f"fa_rarity_{variant}", "_cluster": cluster,
                           "_seed": seed, "_entity": e,
                           **{k: float(rep[k]) for k in ("auroc", "auprc", "vus_pr", "pate_f1")
                              if isinstance(rep.get(k), (int, float)) and np.isfinite(rep.get(k))}}
                    all_records.append(rec)
                    print(f"  [{variant} {cluster} s{seed}] {e}: "
                          f"vus_pr={rep.get('vus_pr', float('nan')):.3f} "
                          f"auprc={rep.get('auprc', float('nan')):.3f} "
                          f"pate_f1={rep.get('pate_f1', float('nan')):.3f}")
                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            del shared_stage1
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rec_path = OUT_ROOT / f"records_{args.dataset}.jsonl"
    with rec_path.open("a") as fh:
        for r in all_records:
            fh.write(json.dumps(r) + "\n")
    print(f"[fa_rarity] appended {len(all_records)} records -> {rec_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
