#!/usr/bin/env python3
"""E se valutassimo con UNO scaler condiviso invece di cinque?

DOMANDA. L'artefatto e' che `PerEntityScaler` (`data.py:464`) e' fittato sul train del client e
applicato al test condiviso, quindi i 5 client normalizzano lo stesso test in 5 modi diversi.
Viene naturale chiedersi se basti aggregare i 5 scaler in uno solo e fare UNA valutazione.

PERCHE' NON E' UN FIX. Il modello e' stato ADDESTRATO con gli scaler per-client. Cambiare solo
la valutazione crea uno scarto fra la distribuzione di addestramento e quella di test: si
sostituisce un artefatto di misura con uno di spostamento di distribuzione. Per `local`, dove
ogni client ha il proprio modello addestrato a una certa scala, il danno e' quasi garantito.

PERCHE' L'ESPERIMENTO VALE COMUNQUE. Per `centralized` il modello e' UNO, quindi «una sola
valutazione» e' concettualmente piu' corretta di cinque. E i 5 shard sono fette disgiunte dello
stesso train originale: **lo scaler aggregato coincide con quello degli autori**. Quindi la
domanda diventa verificabile — col loro stesso scaler, il nostro modello trova l'anomalia?

Tre valutazioni dello STESSO checkpoint, cambiando solo la normalizzazione del test:
    (a) scaler del client di riferimento   (b) scaler di un altro client   (c) scaler AGGREGATO

⚠️ `detect` scrive `detect_score_cache.npz` dentro artifacts/runs anche con `output_dir` altrove.
Innocuo su celle chiuse (fingerprint per entity_id), da evitare su celle vive.
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/leonardo/PhD/TimeVQVAE-AD-U-Federated")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pipeline"))

import data as datamod  # noqa: E402
from config import Config, apply_dataset_overrides, apply_env_overrides  # noqa: E402
import detect  # noqa: E402

OUT = Path("/tmp/claude-1016/-home-leonardo-PhD-TimeVQVAE-AD-U-Federated"
           "/d8ea9ea2-e5c6-4e25-b2b3-33cdd5f3b8e0/scratchpad/pooled")


def pooled_stats(series: str) -> tuple[float, float]:
    """Media e std sulla CONCATENAZIONE dei 5 shard = il train originale intero,
    cioe' esattamente cio' su cui gli autori fittano il loro scaler."""
    xs = [np.load(REPO / f"data/raw/ucr_split_w2p/train/{series}_p{p}.npy").ravel()
          for p in range(5)]
    allx = np.concatenate(xs)
    return float(allx.mean()), float(allx.std())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", required=True)
    ap.add_argument("--window", type=int, required=True)
    ap.add_argument("--arm", default="centralized")
    ap.add_argument("--ckpt-client", type=int, default=0)
    ap.add_argument("--other-client", type=int, default=1)
    ap.add_argument("--tol", type=int, default=64)
    a = ap.parse_args()

    s = f"ucr_{a.series}"
    arm_dir = REPO / f"artifacts/runs/ucr{a.series}_v1/ckpt/ucr_split_w2p/{s}/seed0/{a.arm}"
    ck = arm_dir / f"{s}_p{a.ckpt_client}"
    mu, sd = pooled_stats(s)
    print(f"scaler AGGREGATO (= train originale intero): mean={mu:+.4f} std={sd:.4f}")
    for p in range(5):
        x = np.load(REPO / f"data/raw/ucr_split_w2p/train/{s}_p{p}.npy").ravel()
        print(f"  scaler di p{p}: mean={x.mean():+.4f} std={x.std():.4f}")

    base = Config()
    base.dataset.name = "ucr_split_w2p"
    apply_dataset_overrides(base)
    apply_env_overrides(base)
    base.dataset.window_length = a.window
    base.evaluation.paper_metrics_tolerance = a.tol

    vero_fit = datamod.PerEntityScaler.fit

    def run(tag: str, entity: str, forza_pool: bool):
        """`forza_pool` sostituisce le statistiche del client con quelle aggregate DOPO il fit,
        lasciando intatto tutto il resto del percorso dati."""
        if forza_pool:
            def fit_pool(self, records):
                vero_fit(self, records)
                for k in self.stats:
                    n_ch = len(self.stats[k][0])
                    self.stats[k] = (np.full(n_ch, mu, dtype=np.float64),
                                     np.full(n_ch, sd, dtype=np.float64))
                return self
            datamod.PerEntityScaler.fit = fit_pool
        else:
            datamod.PerEntityScaler.fit = vero_fit
        cfg = copy.deepcopy(base); cfg.dataset.entity_id = entity
        d = OUT / tag; d.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*70}\n  {tag}\n{'='*70}", flush=True)
        detect.detect(cfg, stage1_ckpt=ck / "stage1.ckpt",
                      stage2_ckpt=ck / "stage2.ckpt", output_dir=d)
        datamod.PerEntityScaler.fit = vero_fit
        z = np.load(d / "scores.npz")
        return z["test_scores"], z["test_labels"]

    res = {}
    res[f"scaler p{a.ckpt_client}"] = run(f"{s}_scaler_p{a.ckpt_client}",
                                          f"{s}_p{a.ckpt_client}", False)
    res[f"scaler p{a.other_client}"] = run(f"{s}_scaler_p{a.other_client}",
                                           f"{s}_p{a.other_client}", False)
    res["scaler AGGREGATO"] = run(f"{s}_scaler_pooled", f"{s}_p{a.ckpt_client}", True)

    lab = res["scaler AGGREGATO"][1]
    seg = np.flatnonzero(lab)
    print(f"\n{'='*70}\n  ESITO — checkpoint di p{a.ckpt_client}, segmento {seg.min()}..{seg.max()}"
          f"\n{'='*70}")
    for k, (v, _) in res.items():
        am = int(v.argmax())
        ok = seg.min() - a.tol <= am <= seg.max() + a.tol
        dist = 0 if ok else min(abs(am - seg.min()), abs(am - seg.max()))
        print(f"  {k:20s} argmax={am:6d}  {'TROVATA' if ok else f'mancata di {dist}'}")


if __name__ == "__main__":
    main()
