"""
#3 — FedDF: one-shot ensemble distillation in distribution space.

Each client keeps its OWN deep prior (the converged `federated_cb_only` priors —
no weight averaging ever happens, so the monotone-harmful operation is avoided).
We distill the ENSEMBLE of their masked-prediction distributions into a single
global student prior on a shared UNLABELED probe (pooled train windows), matching
teacher soft-logits by KL. No hard targets → the student never sees pooled labels;
it only learns what the teacher ensemble predicts (federated-legal FedDF).

The ensemble of conditionals carries cross-client structure no single teacher has
and can, in principle, exceed every teacher — the property weight-averaging lacks.
The student is one deployable model (unlike the #1 mixture, which keeps all K
priors at inference). If #1 shows mixture ≫ cb_only, this compresses that gain.

Student scored through the EXACT detect.py machinery (VUS-PR / AUPRC / PATE),
so its numbers sit on the same axis as local / cb_only / centralized.

Usage:
  CUDA_VISIBLE_DEVICES=1 python scripts/feddf_distill.py \
      --dataset toy_fed_uni --clusters all --seeds 0,1,2,3 --distill-steps 1500
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pipeline"))

from data import make_dataloaders  # noqa: E402
from stage1 import load_stage1  # noqa: E402
from stage2 import Stage2System, load_stage2, _flatten_token_indices  # noqa: E402
from federated import resolve_clients  # noqa: E402

# Reuse the exact cfg builder + detect-core scorer written for #1.
from mixture_eval import _build_cfg, _score_entity, _load_pool, CONVERGED  # noqa: E402

OUT_ROOT = REPO / "artifacts" / "fed_eval" / "feddf"


@torch.no_grad()
def _build_probe_tokens(cfg, entities, shared_stage1, device, max_windows: int) -> torch.Tensor:
    """Pool train windows across clients, tokenize ONCE with the shared codebook.
    Returns (N, L) long tokens on CPU. The unlabeled public probe for FedDF."""
    toks = []
    per_client = max(1, max_windows // max(1, len(entities)))
    for e in entities:
        c = copy.deepcopy(cfg); c.dataset.entity_id = e
        loader = make_dataloaders(c, stage="stage2").train_loader
        got = 0
        for batch in loader:
            x = batch["inputs"].to(device, non_blocking=True)
            _, idx, _ = shared_stage1.encode_tokens(x)
            t = _flatten_token_indices(idx).long().cpu()
            toks.append(t)
            got += t.shape[0]
            if got >= per_client:
                break
    allt = torch.cat(toks, dim=0)
    if allt.shape[0] > max_windows:
        allt = allt[torch.randperm(allt.shape[0])[:max_windows]]
    return allt


def _distill(student: Stage2System, teachers, probe_tokens, device,
             steps: int, lr: float, batch: int, seed: int) -> list[float]:
    """Pure soft-KL distillation of the teacher ensemble into the student prior.
    Masks each probe batch (student's own scheme), matches the ensemble mean of
    teacher softmax at masked positions. Student only; teachers frozen."""
    student.prior.train()
    for t in teachers:
        t.eval()
    opt = torch.optim.AdamW(student.prior.parameters(), lr=lr,
                            fused=(device.type == "cuda"))
    N = probe_tokens.shape[0]
    g = torch.Generator().manual_seed(seed * 101 + 1)
    losses = []
    for s in range(steps):
        sel = torch.randint(0, N, (batch,), generator=g)
        tokens = probe_tokens[sel].to(device)
        masked, mask = student.prior._mask_tokens(tokens)          # student scheme
        if mask.sum() == 0:
            continue
        with torch.no_grad():
            p_bar = None
            for t in teachers:
                tl = t._logits(masked)                             # (B, L, K)
                p = F.softmax(tl.float(), dim=-1)
                p_bar = p if p_bar is None else p_bar + p
            p_bar = p_bar / len(teachers)                          # ensemble mean prob
        student_logits = student.prior._logits(masked)            # train (dropout on ctx)
        log_q = F.log_softmax(student_logits.float(), dim=-1)
        per_pos = -(p_bar * log_q).sum(-1)                        # (B, L) CE to soft target
        loss = per_pos[mask].mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        lv = float(loss.detach())
        if np.isfinite(lv):
            losses.append(lv)
        if s < 2 or (s + 1) % 250 == 0 or s == steps - 1:
            print(f"    [distill] step {s+1}/{steps} kl={lv:.4f}")
    return losses


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="toy_fed_uni")
    ap.add_argument("--clusters", default="all")
    ap.add_argument("--seeds", default="0,1,2,3")
    ap.add_argument("--distill-steps", type=int, default=1500)
    ap.add_argument("--distill-lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--probe-windows", type=int, default=4096)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = _build_cfg(args.dataset, args.batch)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    if args.clusters == "all":
        clusters = sorted(p.name for p in (CONVERGED / args.dataset).iterdir() if p.is_dir())
    else:
        clusters = [c.strip() for c in args.clusters.split(",") if c.strip()]

    print(f"[feddf] dataset={args.dataset} clusters={clusters} seeds={seeds} "
          f"steps={args.distill_steps} device={device}")
    all_records = []
    for cluster in clusters:
        entities = resolve_clients(cfg, None, cluster)
        for seed in seeds:
            shared_stage1, pool = _load_pool(args.dataset, cluster, seed, entities, cfg, device)
            if shared_stage1 is None:
                print(f"[feddf] SKIP {cluster} seed{seed}: cb_only run missing/incomplete")
                continue
            teacher_priors, have = pool
            print(f"[feddf] {cluster} seed{seed}: {len(teacher_priors)} teachers over {have}")

            # Fresh student on the SAME shared codebook.
            c0 = copy.deepcopy(cfg); c0.dataset.entity_id = have[0]
            example = next(iter(make_dataloaders(c0, stage="eval").train_loader))["inputs"][:1].cpu()
            student = Stage2System(cfg, shared_stage1)
            student.prior.to(device)
            student.materialize(example.to(device))
            student.to(device)

            probe = _build_probe_tokens(cfg, have, shared_stage1, device, args.probe_windows)
            print(f"[feddf] probe windows={probe.shape[0]} seq_len={probe.shape[1]}")
            _distill(student, teacher_priors, probe, device,
                     steps=args.distill_steps, lr=args.distill_lr, batch=args.batch, seed=seed)

            student.stage1.eval(); student.prior.eval()
            out_dir = OUT_ROOT / args.dataset / cluster / f"seed{seed}"
            for e in have:
                rep = _score_entity(shared_stage1, student, cfg, e)   # student IS a Stage2System
                d = out_dir / e; d.mkdir(parents=True, exist_ok=True)
                (d / "report.json").write_text(json.dumps(rep, indent=2))
                rec = {"_arm": "feddf", "_cluster": cluster, "_seed": seed, "_entity": e,
                       **{k: float(rep[k]) for k in ("auroc", "auprc", "vus_pr", "pate_f1")
                          if isinstance(rep.get(k), (int, float)) and np.isfinite(rep.get(k))}}
                all_records.append(rec)
                print(f"  [feddf {cluster} s{seed}] {e}: "
                      f"vus_pr={rep.get('vus_pr', float('nan')):.3f} "
                      f"auprc={rep.get('auprc', float('nan')):.3f} "
                      f"pate_f1={rep.get('pate_f1', float('nan')):.3f}")

            del student, teacher_priors, shared_stage1, probe
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rec_path = OUT_ROOT / f"records_{args.dataset}.jsonl"
    with rec_path.open("w") as fh:
        for r in all_records:
            fh.write(json.dumps(r) + "\n")
    print(f"[feddf] wrote {len(all_records)} records -> {rec_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
