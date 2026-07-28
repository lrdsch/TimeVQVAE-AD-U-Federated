"""Leakage-free cold-start onboarding (leave-one-out, per cluster).

The existing fa_onboard reuses the shared codebook that was federated-trained
across ALL clients of the cluster — INCLUDING the held-out one — so the tokenizer
has already seen the "new" client. That inflates the onboarding number.

This harness runs the controlled comparison at a FIXED training budget R:
  * LEAKY  — train cb_only on ALL N clients once; onboard each held-out h with a
             mixture of the OTHER priors over that (held-out-contaminated) codebook.
  * CLEAN  — for each held-out h, RETRAIN cb_only (codebook + priors) on the N-1
             donors only; onboard h with a mixture of the donor priors over that
             leakage-free codebook.
The leaky-vs-clean delta at the same R is the pure codebook-leakage effect.
Reference: the held-out client's own trained-local vus_pr (the "if it had data"
ceiling) and the prevalence floor ("nothing").

Training is delegated to the validated pipeline/federated_eval.py (arm
federated_cb_only) via subprocess; scoring reuses scripts/mixture_eval.py.

    CUDA_VISIBLE_DEVICES=1 python scripts/onboard_clean.py \
        --dataset toy_fed_uni --cluster M1_rotary --seed 0 --rounds 18
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

from utils import resolve_path
from stage1 import load_stage1
from stage2 import load_stage2
from data import make_dataloaders
from federated import resolve_clients
from cross_eval import build_cfg
from mixture_eval import MixtureStage2, _score_entity

ROOT = resolve_path(".")
OUT = resolve_path("artifacts/fed_eval/onboard_clean")
PYEXE = sys.executable


def train_cb_only(ds, clients, seed, rounds, out_dir, gpu_env):
    """Train federated_cb_only on `clients`; skip if already present."""
    done = all((out_dir / f"seed{seed}" / "federated_cb_only" / c / "stage2.ckpt").exists()
               for c in clients)
    if done:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [PYEXE, "pipeline/federated_eval.py",
           "--dataset", ds, "--clients", ",".join(clients),
           "--arms", "federated_cb_only", "--seeds", str(seed),
           "--s1-rounds", str(rounds), "--s2-rounds", str(rounds),
           "--local-epochs", "2", "--batch", "16",
           "--out-dir", str(out_dir)]
    log = out_dir / f"train_seed{seed}.log"
    with open(log, "w") as lf:
        subprocess.run(cmd, check=True, stdout=lf, stderr=subprocess.STDOUT, cwd=str(ROOT))


def load_pool(ds, out_dir, seed, clients, cfg, device):
    """shared codebook (from clients[0]) + each client's prior, from a trained dir."""
    base = out_dir / f"seed{seed}" / "federated_cb_only"
    ex_cfg = copy.deepcopy(cfg); ex_cfg.dataset.entity_id = clients[0]
    example = next(iter(make_dataloaders(ex_cfg, stage="eval").train_loader))["inputs"][:1].cpu()
    s1p0 = base / clients[0] / "stage1.ckpt"
    shared = load_stage1(s1p0, cfg, example, device=device); shared.eval()
    priors = {}
    for c in clients:
        ce = copy.deepcopy(cfg); ce.dataset.entity_id = c
        s2 = load_stage2(base / c / "stage2.ckpt", ce, stage1_ckpt=base / c / "stage1.ckpt",
                         stage1_example_inputs=example, device=device)
        s2.prior.eval(); priors[c] = s2.prior
    return shared, priors


def local_ref(ds, cluster, seed, entity):
    p = resolve_path(f"artifacts/fed_eval/{ds}/{cluster}/seed{seed}/local/{entity}/report.json")
    if p.exists():
        return json.load(open(p)).get("vus_pr")
    return None


def prevalence(ds, cluster, seed, entity, cfg, device):
    # floor: a random detector's AUPRC ~ anomaly prevalence
    z = resolve_path(f"artifacts/fed_eval/cross/scorevecs_{ds}_s{seed}.npz")
    if z.exists():
        d = np.load(z, allow_pickle=True)
        k = f"y|{entity}"
        if k in d.files:
            return float((d[k] == 1).mean())
    return None


def run(ds, cluster, seed, rounds):
    cfg = build_cfg(ds, seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ents = resolve_clients(cfg, None, cluster)
    OUT.mkdir(parents=True, exist_ok=True)
    rec_path = OUT / f"records_{ds}.jsonl"
    scratch = OUT / ds / cluster / f"R{rounds}"
    print(f"[onboard_clean] {ds}/{cluster} seed{seed} R={rounds}: {len(ents)} clients {ents}", flush=True)

    # ---- LEAKY: one full-cluster training, reused for every held-out ----
    full_dir = scratch / "full"
    train_cb_only(ds, ents, seed, rounds, full_dir, None)
    shared_full, priors_full = load_pool(ds, full_dir, seed, ents, cfg, device)

    for h in ents:
        donors = [e for e in ents if e != h]
        loc = local_ref(ds, cluster, seed, h)
        prev = prevalence(ds, cluster, seed, h, cfg, device)

        # leaky onboard: full (contaminated) codebook + donor priors
        mix_leaky = MixtureStage2(shared_full, [priors_full[d] for d in donors], "mixture")
        rep_leaky = _score_entity(shared_full, mix_leaky, cfg, h)

        # clean onboard: retrain on donors only (codebook + priors)
        hd = scratch / f"holdout_{h}"
        train_cb_only(ds, donors, seed, rounds, hd, None)
        shared_clean, priors_clean = load_pool(ds, hd, seed, donors, cfg, device)
        mix_clean = MixtureStage2(shared_clean, [priors_clean[d] for d in donors], "mixture")
        rep_clean = _score_entity(shared_clean, mix_clean, cfg, h)
        del shared_clean, priors_clean
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        rec = {"dataset": ds, "cluster": cluster, "seed": seed, "rounds": rounds,
               "held_out": h, "n_donors": len(donors),
               "vus_leaky": rep_leaky.get("vus_pr"), "vus_clean": rep_clean.get("vus_pr"),
               "local_vus": loc, "floor_prevalence": prev,
               "auprc_leaky": rep_leaky.get("auprc"), "auprc_clean": rep_clean.get("auprc")}
        with open(rec_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        lk, cl = rec["vus_leaky"], rec["vus_clean"]
        lo = f"{loc:.3f}" if isinstance(loc, float) else "n/a"
        print(f"  [{cluster} s{seed}] held={h}: leaky={lk:.3f} clean={cl:.3f} "
              f"Δleak={lk-cl:+.3f} | local={lo} floor={prev:.3f}" if isinstance(prev, float)
              else f"  [{cluster} s{seed}] held={h}: leaky={lk:.3f} clean={cl:.3f} Δleak={lk-cl:+.3f} local={lo}",
              flush=True)
    del shared_full, priors_full
    print(f"[onboard_clean] DONE {ds}/{cluster} seed{seed} -> {rec_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--cluster", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=18)
    args = ap.parse_args()
    run(args.dataset, args.cluster, args.seed, args.rounds)


if __name__ == "__main__":
    main()
