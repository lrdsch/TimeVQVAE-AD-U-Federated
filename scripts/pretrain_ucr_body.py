#!/usr/bin/env python3
"""
pretrain_ucr_body.py — pretrain a COHERENT, FEDERATION-LEGAL shared body on the
public UCR corpus, and drop it into the converged tree as the `ucr_pretrained` arm
so the EXISTING flare/prism analytic head can read it (L5 probe).

The body (stage1 tokenizer + MaskGIT prior) is trained ONCE on the pooled UCR
`train` segments via the SAME `train_centralized` code path used for the wsd
skyline — identical architecture (window=256, K=64), identical AMP/optimizer. It
never sees a client's private data, so it is federation-legal; it is a single
pooled model, so it is coherent. We then copy the one pretrained (stage1, stage2)
pair under EVERY target entity dir of each requested (cluster, seed), because
flare's `_load_single_shared_body` loads entity[0]'s body and reuses it for all —
but requires >=2 entities to have the ckpts (to collect per-client Gram stats).

Usage (GPU1 house rule):
    CUDA_VISIBLE_DEVICES=1 python scripts/pretrain_ucr_body.py \
        --target-dataset wsd_fed --target-clusters c3 --target-seeds 0 \
        --s1-epochs 35 --s2-epochs 35 --pool-cap 100000 --batch 64 --seed 0

Then evaluate:
    CUDA_VISIBLE_DEVICES=1 python scripts/flare_eval.py --dataset wsd_fed \
        --clusters c3 --seeds 0 --arms flare_ucrbody --lams 0.1,1,10
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
import sys

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pipeline"))
sys.path.insert(0, str(REPO / "scripts"))

from mixture_eval import _build_cfg, CONVERGED                       # noqa: E402
from federated_eval import _loaders_for, train_centralized           # noqa: E402
from federated import resolve_clients                                # noqa: E402
from stage1 import save_stage1_checkpoint                            # noqa: E402
from stage2 import save_stage2_checkpoint                            # noqa: E402
from utils import seed_everything                                    # noqa: E402


def _ucr_entities(cfg) -> list[str]:
    root = Path(cfg.paths.raw_data) if os.path.isabs(str(cfg.paths.raw_data)) else REPO / cfg.paths.raw_data
    root = root / cfg.dataset.name / "train"
    ents = sorted(Path(p).stem for p in glob.glob(str(root / "*.npy")))
    if not ents:
        raise SystemExit(f"No ucr_pool entities under {root}. Run scripts/build_ucr_pool.py first.")
    return ents


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ucr_pool", help="pretraining corpus name")
    ap.add_argument("--target-dataset", default="wsd_fed", help="where the body is deployed as a body arm")
    ap.add_argument("--target-clusters", default="c3")
    ap.add_argument("--target-seeds", default="0")
    ap.add_argument("--arm-dir", default="ucr_pretrained")
    ap.add_argument("--s1-epochs", type=int, default=35)
    ap.add_argument("--s2-epochs", type=int, default=35)
    ap.add_argument("--pool-cap", type=int, default=100000,
                    help="diverse-subsample the pooled UCR windows to this many (0=all)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = _build_cfg(args.dataset, args.batch)
    entities = _ucr_entities(cfg)
    print(f"[pretrain_ucr] corpus={args.dataset} entities={len(entities)} device={device} "
          f"s1_epochs={args.s1_epochs} s2_epochs={args.s2_epochs} pool_cap={args.pool_cap} seed={args.seed}",
          flush=True)

    print("[pretrain_ucr] building per-entity loaders ...", flush=True)
    data_by_e = _loaders_for(cfg, entities)

    seed_everything(args.seed)                         # deterministic pool subsample + training
    models = train_centralized(cfg, entities, data_by_e, args.s1_epochs, args.s2_epochs,
                               device, pool_cap=args.pool_cap)
    s1, s2 = models[entities[0]]

    # Deploy: copy the one pretrained body under every target entity dir.
    clusters = [c.strip() for c in args.target_clusters.split(",") if c.strip()]
    seeds = [int(s) for s in args.target_seeds.split(",") if s.strip()]
    tgt_cfg = _build_cfg(args.target_dataset, args.batch)
    n_written = 0
    for cluster in clusters:
        tents = resolve_clients(tgt_cfg, None, cluster)
        for seed in seeds:
            base = CONVERGED / args.target_dataset / cluster / f"seed{seed}" / args.arm_dir
            for e in tents:
                d = base / e
                d.mkdir(parents=True, exist_ok=True)
                save_stage1_checkpoint(d / "stage1.ckpt", s1, cfg, 0, args.s1_epochs)
                save_stage2_checkpoint(d / "stage2.ckpt", s2, cfg, 0, args.s2_epochs)
                n_written += 1
            print(f"[pretrain_ucr] deployed body -> {base} ({len(tents)} entities)", flush=True)
    print(f"[pretrain_ucr] DONE: wrote {n_written} ckpt pairs under arm '{args.arm_dir}'.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
