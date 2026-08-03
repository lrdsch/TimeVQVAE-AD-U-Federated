#!/usr/bin/env python3
"""Scambio controllato di shard: lo stesso checkpoint, shard di train diversi.

Generalizza `scripts/repro_probe.py` (che era cablato su ucr_086) a qualunque serie e coppia
di client, perche' il meccanismo va verificato dove il predittore dice che morde di piu', non
solo dove l'ho scoperto.

COSA DIMOSTRA. Tiene fisso il checkpoint e cambia solo l'entity, cioe' solo quale fetta di
train viene caricata. Il test e' byte-identico fra i client (verificato: stesso md5), quindi
qualunque differenza nel punteggio viene dal train. La causa nota e' `PerEntityScaler`
(`data.py:464`), fittato sul train e applicato al test: fette con deviazione standard diversa
producono normalizzazioni diverse dello STESSO test.

    $PY scripts/shard_swap_probe.py --series 170 --ckpt-client 0 --train-clients 0,1 --window 674

⚠️ `detect` scrive `detect_score_cache.npz` dentro artifacts/runs anche con `output_dir` altrove
(il path e' ancorato al checkpoint). Innocuo su celle chiuse — il fingerprint include
`entity_id`, quindi al piu' provoca un cache-miss — ma da evitare su una cella viva.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/leonardo/PhD/TimeVQVAE-AD-U-Federated")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pipeline"))

from config import Config, apply_dataset_overrides, apply_env_overrides  # noqa: E402
import detect  # noqa: E402

OUT = Path("/tmp/claude-1016/-home-leonardo-PhD-TimeVQVAE-AD-U-Federated"
           "/d8ea9ea2-e5c6-4e25-b2b3-33cdd5f3b8e0/scratchpad/swap")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", required=True)               # es. 170
    ap.add_argument("--arm", default="centralized")
    ap.add_argument("--tag", default=None)                   # default ucr<serie>_v1
    ap.add_argument("--ckpt-client", type=int, default=0)    # il checkpoint resta SEMPRE questo
    ap.add_argument("--train-clients", default="0,1")        # gli shard da provare
    ap.add_argument("--window", type=int, required=True)
    ap.add_argument("--tol", type=int, default=64)
    a = ap.parse_args()

    s = a.series
    tag = a.tag or f"ucr{s}_v1"
    arm_dir = REPO / f"artifacts/runs/{tag}/ckpt/ucr_split_w2p/ucr_{s}/seed0/{a.arm}"
    ck = arm_dir / f"ucr_{s}_p{a.ckpt_client}"

    base = Config()
    base.dataset.name = "ucr_split_w2p"
    apply_dataset_overrides(base)
    apply_env_overrides(base)
    base.dataset.window_length = a.window
    base.evaluation.paper_metrics_tolerance = a.tol

    S, orig = {}, {}
    for p in [int(x) for x in a.train_clients.split(",")]:
        cfg = copy.deepcopy(base)
        cfg.dataset.entity_id = f"ucr_{s}_p{p}"
        d = OUT / f"ucr{s}_ck{a.ckpt_client}_tr{p}"
        d.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*70}\n  checkpoint di p{a.ckpt_client}  +  train di p{p}\n{'='*70}", flush=True)
        detect.detect(cfg, stage1_ckpt=ck / "stage1.ckpt",
                      stage2_ckpt=ck / "stage2.ckpt", output_dir=d)
        S[p] = np.load(d / "scores.npz")["test_scores"]
        orig[p] = np.load(arm_dir / f"ucr_{s}_p{p}" / "scores.npz")["test_scores"]
        lab = np.load(d / "scores.npz")["test_labels"]

    seg = np.flatnonzero(lab)
    print(f"\n{'='*70}\n  ESITO — segmento anomalo {seg.min()}..{seg.max()}\n{'='*70}")
    for p, v in S.items():
        am = int(v.argmax())
        dist = 0 if seg.min() - a.tol <= am <= seg.max() + a.tol else min(abs(am - seg.min()),
                                                                         abs(am - seg.max()))
        rho = float(np.corrcoef(v, orig[p])[0, 1])
        print(f"  train p{p}: argmax={am:6d}  {'TROVATA' if dist == 0 else f'mancata di {dist}'}"
              f"   | rho con la corsa originale di p{p} = {rho:.6f}")
    for x, y in itertools.combinations(S, 2):
        print(f"  train p{x} vs p{y}: rho={float(np.corrcoef(S[x], S[y])[0,1]):+.4f}  "
              f"max|Δ|={np.abs(S[x]-S[y]).max():.4g}")


if __name__ == "__main__":
    main()
