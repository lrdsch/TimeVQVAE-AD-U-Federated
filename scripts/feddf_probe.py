"""
FedDF probe (i) — does the distilled student actually REACH the ensemble mean p̄?

The audited fact: FedDF distillation converged (cross-entropy flat from ~step 250)
yet feddf VUS-PR (0.182) < mixture (0.264), and the mixture score −log p̄ is the
EXACT NLL of the arithmetic mean p̄ = mean_k softmax(teacher_k) that FedDF
distills. So a perfect student (q = p̄) would reproduce the mixture. It doesn't.

Two candidate causes, both from the same masked-token forward:
  (i)  capacity  — the single student CANNOT represent p̄: residual KL(p̄‖q) > 0
                   at the converged (flat-loss) floor.
  (ii) proxy≠test — q ≈ p̄ on train contexts but the test distribution differs.

This script isolates (i): after distillation it measures, over masked positions
on a held-out proxy sample,
    H(p̄)      = -Σ p̄ log p̄            (the cross-entropy floor / teacher entropy)
    CE(p̄,q)   = -Σ p̄ log q            (what distillation minimises)
    KL(p̄‖q)   = CE - H                 (how far the student still is from p̄)
KL≈0 ⇒ student reaches p̄ ⇒ the feddf<mixture gap is NOT capacity → it is (ii),
scoring/proxy shift (follow-up: measure the same on tokenised TEST windows).
KL≫0 ⇒ capacity floor: the mixture does not compress into one same-size prior.

Usage:
  CUDA_VISIBLE_DEVICES=1 python scripts/feddf_probe.py \
      --dataset wsd_fed --clusters all --seeds 0,1 --distill-steps 1500
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
from federated import resolve_clients  # noqa: E402
from stage2 import Stage2System  # noqa: E402
from mixture_eval import CONVERGED, _build_cfg, _load_pool  # noqa: E402
from feddf_distill import _build_probe_tokens, _distill  # noqa: E402

OUT_ROOT = REPO / "artifacts" / "fed_eval" / "feddf_probe"


@torch.no_grad()
def _kl_floor(student, teachers, probe_tokens, device, batch, seed, n_batches=60):
    """Mean H(p̄), CE(p̄,q), KL(p̄‖q) over masked positions on a fresh proxy sample."""
    student.prior.eval()
    for t in teachers:
        t.eval()
    N = probe_tokens.shape[0]
    g = torch.Generator().manual_seed(seed * 13 + 7)
    tot_H = tot_CE = 0.0
    tot_n = 0
    for _ in range(n_batches):
        sel = torch.randint(0, N, (batch,), generator=g)
        tokens = probe_tokens[sel].to(device)
        masked, mask = student.prior._mask_tokens(tokens)
        if int(mask.sum()) == 0:
            continue
        p_bar = None
        for t in teachers:
            p = F.softmax(t._logits(masked).float(), dim=-1)
            p_bar = p if p_bar is None else p_bar + p
        p_bar = p_bar / len(teachers)
        log_pbar = p_bar.clamp_min(1e-12).log()
        log_q = F.log_softmax(student.prior._logits(masked).float(), dim=-1)
        H = -(p_bar * log_pbar).sum(-1)[mask]     # teacher-ensemble entropy
        CE = -(p_bar * log_q).sum(-1)[mask]       # distillation cross-entropy
        tot_H += float(H.sum()); tot_CE += float(CE.sum()); tot_n += int(mask.sum())
    H = tot_H / max(1, tot_n); CE = tot_CE / max(1, tot_n)
    return {"H_pbar": H, "CE_pbar_q": CE, "KL_pbar_q": CE - H, "n_masked": tot_n}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="wsd_fed")
    ap.add_argument("--clusters", default="all")
    ap.add_argument("--seeds", default="0,1")
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

    print(f"[feddf_probe] dataset={args.dataset} clusters={clusters} seeds={seeds} "
          f"steps={args.distill_steps} device={device}")
    records = []
    for cluster in clusters:
        entities = resolve_clients(cfg, None, cluster)
        for seed in seeds:
            shared_stage1, pool = _load_pool(args.dataset, cluster, seed, entities, cfg, device)
            if shared_stage1 is None:
                print(f"[feddf_probe] SKIP {cluster} seed{seed}: cb_only missing")
                continue
            teachers, have = pool
            c0 = copy.deepcopy(cfg); c0.dataset.entity_id = have[0]
            example = next(iter(make_dataloaders(c0, stage="eval").train_loader))["inputs"][:1].cpu()
            student = Stage2System(cfg, shared_stage1)
            student.prior.to(device); student.materialize(example.to(device)); student.to(device)

            probe = _build_probe_tokens(cfg, have, shared_stage1, device, args.probe_windows)
            losses = _distill(student, teachers, probe, device,
                              steps=args.distill_steps, lr=args.distill_lr, batch=args.batch, seed=seed)
            # A second, independently-sampled proxy pool ⇒ the KL floor is measured
            # on windows not tied to the distill minibatch stream.
            eval_probe = _build_probe_tokens(cfg, have, shared_stage1, device, args.probe_windows)
            m = _kl_floor(student, teachers, eval_probe, device, args.batch, seed)
            m.update({"_cluster": cluster, "_seed": seed, "n_teachers": len(teachers),
                      "distill_final_ce": float(losses[-1]) if losses else float("nan")})
            records.append(m)
            print(f"[feddf_probe {cluster} s{seed}] teachers={len(teachers)} "
                  f"H(p̄)={m['H_pbar']:.4f} CE(p̄,q)={m['CE_pbar_q']:.4f} "
                  f"KL(p̄‖q)={m['KL_pbar_q']:.4f}  (distill_final={m['distill_final_ce']:.4f})")
            del student, teachers, shared_stage1, probe, eval_probe
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rec_path = OUT_ROOT / f"records_{args.dataset}.jsonl"
    with rec_path.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    # Verdict summary.
    if records:
        mk = np.mean([r["KL_pbar_q"] for r in records])
        mh = np.mean([r["H_pbar"] for r in records])
        print(f"\n[feddf_probe] MEAN over {len(records)} runs: "
              f"H(p̄)={mh:.4f}  KL(p̄‖q)={mk:.4f}  "
              f"→ {'CAPACITY floor (i): student cannot reach p̄' if mk > 0.15 else 'student ≈ reaches p̄ → gap is (ii) proxy/scoring, not capacity'}")
    print(f"[feddf_probe] wrote {len(records)} records -> {rec_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
