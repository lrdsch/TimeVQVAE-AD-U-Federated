#!/usr/bin/env python3
"""ESPERIMENTO DECISIVO: `centralized` addestrato con lo scaler FEDERATO a priori.

DOMANDA. Su `ucr_170` il nostro `centralized` manca l'anomalia che gli autori trovano. Sappiamo
che (a) i 5 client normalizzano il test in 5 modi, e (b) `centralized` mette in pool record gia'
normalizzati da scaler diversi, quindi e' addestrato su una **miscela di 5 riscalature**.
Sappiamo anche che correggere solo la VALUTAZIONE non basta: con lo scaler aggregato il modello
gia' addestrato manca comunque (`scripts/pooled_scaler_probe.py`, argmax 16656).

Resta da verificare l'ipotesi che conta: **se lo scaler condiviso lo si applica PRIMA di
addestrare, il modello ritrova l'anomalia come gli autori?** Se si', la diagnosi e' chiusa e il
fix e' validato; se no, la causa e' altrove e va cercata.

COME. `PerEntityScaler.fit` viene sostituito in modo che ogni entity riceva le statistiche
aggregate sui 5 shard. La concatenazione delle 5 fette **e' il train originale intero** (sono
fette disgiunte), e media/varianza su di essa coincidono con l'aggregazione delle statistiche
sufficienti additive (n, Sx, Sx^2) che un vero scaler federato calcolerebbe in un round. Quindi
questo e' lo scaler federato, non un'approssimazione — ed e' anche lo scaler degli autori.

Il resto della riga di comando e' copiato da `scripts/launch.sh:344-349` (protocol converged,
300 round, 10 epoche locali, pazienza 6, batch 64, seed 0), cosi' l'UNICA differenza rispetto
alla cella della campagna e' la normalizzazione.

    $PY scripts/fedscaler_train_probe.py --series 170 --window 674

⚠️ Scrive in `artifacts/runs/ucr<serie>_fedscaler/`, tag separato: non tocca le 306 celle.
"""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/leonardo/PhD/TimeVQVAE-AD-U-Federated")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pipeline"))

import data as datamod  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", required=True)
    ap.add_argument("--window", type=int, required=True)
    ap.add_argument("--arm", default="centralized")
    ap.add_argument("--tol", type=int, default=64)
    a = ap.parse_args()
    s = f"ucr_{a.series}"

    xs = [np.load(REPO / f"data/raw/ucr_split_w2p/train/{s}_p{p}.npy").ravel() for p in range(5)]
    allx = np.concatenate(xs)
    mu, sd = float(allx.mean()), float(allx.std())
    print(f"[fedscaler] statistiche aggregate sui 5 shard: mean={mu:+.4f} std={sd:.4f}")
    for p, x in enumerate(xs):
        print(f"[fedscaler]   p{p}: mean={x.mean():+.4f} std={x.std():.4f}")

    vero_fit = datamod.PerEntityScaler.fit

    def fit_federato(self, records):
        """Fitta come sempre (per avere le chiavi e la forma giuste), poi sovrascrive le
        statistiche di OGNI entity con quelle aggregate. Cosi' train, val e test di tutti e 5 i
        client passano per la stessa identica normalizzazione."""
        vero_fit(self, records)
        for k in self.stats:
            n_ch = len(self.stats[k][0])
            self.stats[k] = (np.full(n_ch, mu, dtype=np.float64),
                             np.full(n_ch, sd, dtype=np.float64))
        return self

    datamod.PerEntityScaler.fit = fit_federato
    print("[fedscaler] PerEntityScaler.fit sostituito: scaler UNICO per tutti i client\n")

    out = REPO / f"artifacts/runs/ucr{a.series}_fedscaler"
    sys.argv = [
        "federated_eval.py",
        "--dataset", "ucr_split_w2p", "--cluster", s, "--arms", a.arm,
        "--protocol", "converged", "--s1-rounds", "300", "--s2-rounds", "300",
        "--local-epochs", "10", "--fed-patience-rounds", "6", "--batch", "64",
        "--seeds", "0",
        "--window-length", str(a.window), "--metrics-tolerance", str(a.tol),
        "--out-dir", str(out / "ckpt/ucr_split_w2p"),
        "--out-json", str(out / f"{a.arm}.json"),
    ]
    (out / "ckpt/ucr_split_w2p").mkdir(parents=True, exist_ok=True)
    runpy.run_path(str(REPO / "pipeline/federated_eval.py"), run_name="__main__")


if __name__ == "__main__":
    main()
