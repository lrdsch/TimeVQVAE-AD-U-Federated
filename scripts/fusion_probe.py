#!/usr/bin/env python3
"""Fusione in FUNCTION-SPACE dei client di un cluster, e diversita' degli errori.

Perche' esiste (2026-08-08). Su ucr_170 la media dei profili di score dei 5 modelli
`local` da' AUPRC 0,698 contro 0,303 della media per-client -- e sopra il MIGLIOR client
singolo (0,588). Il motivo, misurato: con encoder locale i falsi positivi dei client sono
IDIOSINCRATICI (corr media fra profili 0,558; i top-picchi cadono in punti diversi) mentre
l'anomalia vera e' comune, quindi mediare cancella i FP. Con l'encoder mediato (A2) i 5
modelli sono lo stesso modello -- corr 0,981, stessa classifica di picchi su 5/5 -- e il
guadagno da fusione e' esattamente +0,000.

Ne segue che ogni cella che cambia QUANTO encoder si condivide va letta su DUE numeri, non
uno: la media per-client (la metrica standard delle celle) e la fusione. La correlazione
fra i profili e' la variabile di meccanismo che le lega.

⚠️ Il guadagno dipende dalla REGOLA di fusione: su ucr_170/local la media da' 0,698 e la
trimmed 0,690, ma la rank-mean 0,277 (sotto la media per-client) e la mediana 0,363. Vive
nell'AMPIEZZA dei picchi, non nell'ordinamento -- va dichiarato quando si riporta.

⚠️ Sul dataset ucr_split_w2p il TEST e' identico fra i client (sono gli SHARD DI TRAIN a
essere disgiunti): la fusione e' quindi una fusione di MODELLI sugli stessi dati, e resta
federation-legal (nessun dato attraversa la rete, solo i modelli/score). L'unita' di analisi
resta il CLUSTER: la fusione produce UN numero, la media per-client un altro -- non sono la
stessa quantita' e non vanno confrontate con le soglie di rumore per-cella.

Uso:
    python3.10 scripts/fusion_probe.py --run artifacts/runs/zn_main --series ucr_170
    python3.10 scripts/fusion_probe.py --run artifacts/runs/zn_170_neck --series ucr_170 \
        --baseline artifacts/runs/zn_a2
    python3.10 scripts/fusion_probe.py --run artifacts/runs/zn_main --all-series --json out.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score


def _load_cluster(arm_dir: str, series: str):
    """Profili di test dei client di un cluster. -> (S [n_client, T], y [T]) o (None, None)."""
    paths = sorted(glob.glob(os.path.join(arm_dir, f"{series}_p*", "scores.npz")))
    if not paths:
        return None, None
    scores, labels = [], None
    for p in paths:
        z = np.load(p)
        scores.append(z["test_scores"])
        labels = z["test_labels"]
    lens = {len(s) for s in scores}
    if len(lens) != 1:
        raise SystemExit(f"client con lunghezze di test diverse in {arm_dir}: {lens}")
    return np.asarray(scores), (labels > 0).astype(int)


def _z(S: np.ndarray) -> np.ndarray:
    sd = S.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (S - S.mean(axis=1, keepdims=True)) / sd


def analyse(arm_dir: str, series: str) -> dict | None:
    S, y = _load_cluster(arm_dir, series)
    if S is None or y is None or y.sum() == 0:
        return None
    per = [float(average_precision_score(y, s)) for s in S]
    Z = _z(S)
    R = np.asarray([rankdata(s) / len(s) for s in S])
    trimmed = np.sort(Z, axis=0)[1:-1].mean(axis=0) if len(S) > 2 else Z.mean(axis=0)
    fus = {
        "mean_z": float(average_precision_score(y, Z.mean(axis=0))),
        "median_z": float(average_precision_score(y, np.median(Z, axis=0))),
        "trimmed_z": float(average_precision_score(y, trimmed)),
        "mean_rank": float(average_precision_score(y, R.mean(axis=0))),
    }
    corr = np.corrcoef(Z)
    iu = np.triu_indices(len(S), 1)
    return {
        "arm_dir": arm_dir,
        "series": series,
        "n_client": len(S),
        "per_client_auprc": per,
        "mean_auprc": float(np.mean(per)),
        "best_auprc": float(np.max(per)),
        "fusion": fus,
        "gain_vs_mean": fus["mean_z"] - float(np.mean(per)),
        "gain_vs_best": fus["mean_z"] - float(np.max(per)),
        "corr_mean": float(corr[iu].mean()) if len(S) > 1 else float("nan"),
        "corr_min": float(corr[iu].min()) if len(S) > 1 else float("nan"),
    }


def _arm_dirs(run: str, series: str, dataset: str, seed: int) -> list[str]:
    root = os.path.join(run, "ckpt", dataset, series, f"seed{seed}")
    return sorted(d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d))


def main() -> None:
    ap_ = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--run", required=True, help="artifacts/runs/<tag>")
    ap_.add_argument("--series", default=None, help="es. ucr_170")
    ap_.add_argument("--all-series", action="store_true")
    ap_.add_argument("--dataset", default="ucr_split_w2p")
    ap_.add_argument("--seed", type=int, default=0)
    ap_.add_argument("--baseline", default=None, help="altro run-dir da confrontare")
    ap_.add_argument("--json", default=None, help="scrivi i risultati grezzi qui")
    a = ap_.parse_args()

    if not a.series and not a.all_series:
        ap_.error("serve --series o --all-series")
    if a.all_series:
        root = os.path.join(a.run, "ckpt", a.dataset)
        series = sorted(os.path.basename(p) for p in glob.glob(os.path.join(root, "ucr_*")))
    else:
        series = [a.series]

    rows = []
    for run in filter(None, [a.run, a.baseline]):
        for s in series:
            for d in _arm_dirs(run, s, a.dataset, a.seed):
                r = analyse(d, s)
                if r:
                    r["run"] = run
                    rows.append(r)

    if not rows:
        raise SystemExit("nessun cluster con scores.npz trovato (run/serie/seed giusti?)")

    hdr = f"{'run':<26}{'arm':<44}{'serie':<9}{'media':>7}{'best':>7}{'FUSA':>7}{'d.med':>7}{'d.best':>8}{'corr':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{os.path.basename(r['run']):<26}{os.path.basename(r['arm_dir'])[:43]:<44}"
              f"{r['series']:<9}{r['mean_auprc']:>7.3f}{r['best_auprc']:>7.3f}"
              f"{r['fusion']['mean_z']:>7.3f}{r['gain_vs_mean']:>+7.3f}"
              f"{r['gain_vs_best']:>+8.3f}{r['corr_mean']:>7.3f}")

    print("\nregole di fusione (AUPRC) — la media e' quella riportata sopra:")
    for r in rows:
        f = r["fusion"]
        print(f"  {os.path.basename(r['arm_dir'])[:43]:<44}{r['series']:<9}"
              f"media {f['mean_z']:.3f} · trimmed {f['trimmed_z']:.3f} · "
              f"mediana {f['median_z']:.3f} · rank {f['mean_rank']:.3f}")

    if a.json:
        with open(a.json, "w") as fh:
            json.dump(rows, fh, indent=1)
        print(f"\nscritto {a.json}")


if __name__ == "__main__":
    main()
