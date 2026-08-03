#!/usr/bin/env python3
"""Il punteggio di `detect` è riproducibile a modello e input identici?

DA DOVE NASCE. Il 2026-08-03 ho verificato che i 5 client di `centralized` — che condividono
UN modello e UN test — producono punteggi diversi su tutte e 9 le serie, e su 3 il top-1 passa
da 0 a 1. Verificato ingresso per ingresso: pesi `max|Δw|=0` su 785 tensori, file di test
byte-identici, val identici, `window`/`period` identici, seed 0 ovunque, dropout spento e
asserito, BatchNorm tutta dentro `stage1` che detect asserisce in eval, nessun campionamento
nel percorso di scoring, cache con `entity_id` nel fingerprint. L'unica cosa che differisce
fra le 5 esecuzioni è lo **shard di train**.

Restano due spiegazioni, e questo script le separa:

  (a) NON DETERMINISMO — due esecuzioni identiche danno già punteggi diversi (kernel scelti a
      tempo da cudnn.benchmark, riduzioni atomiche non deterministiche, ...).
  (b) ACCOPPIAMENTO TRAIN→TEST — due esecuzioni identiche coincidono, e il punteggio cambia
      solo quando cambia lo shard di train. Sarebbe un difetto: il test non deve dipenderne.

COSA FA. Tre esecuzioni di `detect` sugli STESSI checkpoint:
    A1, A2 = entity ucr_086_p0  (stesso train, due volte)
    B      = entity ucr_086_p3  (train diverso, stesso test)
Poi confronta i `test_scores`. A1 vs A2 risponde a (a); A vs B misura l'effetto dello shard.

⚠️ I checkpoint sono aperti in sola lettura e i report vanno in una cartella temporanea, MA
`detect` scrive comunque `detect_score_cache.npz` **dentro artifacts/runs**: `_detect_cache_path`
ancora il file al checkpoint (`anchor.parent.parent`), non a `output_dir`. Qui è innocuo — il
fingerprint include `entity_id`, quindi al più si provoca un cache-miss, e le celle di
`ucr086_v1` sono già chiuse — ma su una cella VIVA andrebbe evitato.

ESITO (2026-08-03 05:00): (b). A1 e A2 bit-identici; A vs B rho=0,119 con argmax 36965 -> 48067,
e B riproduce il risultato originale di p3. Il punteggio sul test è funzione dello shard di train.
"""
from __future__ import annotations

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

ARM = REPO / "artifacts/runs/ucr086_v1/ckpt/ucr_split_w2p/ucr_086/seed0/centralized"
OUT = Path("/tmp/claude-1016/-home-leonardo-PhD-TimeVQVAE-AD-U-Federated"
           "/d8ea9ea2-e5c6-4e25-b2b3-33cdd5f3b8e0/scratchpad/repro")
# Presi dal log della corsa vera (`window_length -> 536`, `paper_metrics_tolerance PINNED to 64`).
WINDOW, TOL = 536, 64


def base_cfg() -> Config:
    """La stessa sequenza di federated_eval.py:1210-1275, nell'ordine che conta:
    apply_dataset_overrides PRIMA, poi window e tolleranza espliciti che vincono su di esso."""
    cfg = Config()
    cfg.dataset.name = "ucr_split_w2p"
    apply_dataset_overrides(cfg)
    apply_env_overrides(cfg)
    cfg.dataset.window_length = WINDOW
    cfg.evaluation.paper_metrics_tolerance = TOL
    return cfg


def run(tag: str, entity: str) -> np.ndarray:
    cfg = copy.deepcopy(base_cfg())
    cfg.dataset.entity_id = entity
    d = OUT / tag
    d.mkdir(parents=True, exist_ok=True)
    # ⚠ I checkpoint sono quelli di p0 SEMPRE: e' il punto dell'esperimento. Cambia solo
    # l'entity, cioe' quale shard di train viene caricato.
    detect.detect(cfg, stage1_ckpt=ARM / "ucr_086_p0/stage1.ckpt",
                  stage2_ckpt=ARM / "ucr_086_p0/stage2.ckpt", output_dir=d)
    return np.load(d / "scores.npz")["test_scores"]


def main() -> None:
    runs = {"A1_p0": "ucr_086_p0", "A2_p0": "ucr_086_p0", "B_p3": "ucr_086_p3"}
    S = {}
    for tag, ent in runs.items():
        print(f"\n{'='*70}\n  {tag}: entity={ent}, checkpoint di p0\n{'='*70}", flush=True)
        S[tag] = run(tag, ent)

    print(f"\n{'='*70}\n  ESITO\n{'='*70}")
    for a, b in itertools.combinations(S, 2):
        x, y = S[a], S[b]
        ident = np.array_equal(x, y)
        rho = float(np.corrcoef(x, y)[0, 1])
        print(f"  {a:6s} vs {b:6s}: identici={str(ident):5s}  rho={rho:7.4f}  "
              f"max|Δ|={np.abs(x - y).max():.4g}  argmax {int(x.argmax())} vs {int(y.argmax())}")

    same = np.array_equal(S["A1_p0"], S["A2_p0"])
    print("\n  VERDETTO:")
    if not same:
        print("  (a) NON DETERMINISMO — due esecuzioni identiche divergono gia' da sole.")
        print("      Lo shard di train non e' (necessariamente) la causa; il punteggio")
        print("      semplicemente non e' riproducibile.")
    elif np.array_equal(S["A1_p0"], S["B_p3"]):
        print("  NESSUNO DEI DUE — cambiare shard di train non cambia il punteggio.")
        print("      Allora la differenza fra i 5 client viene da altro: rileggere il caso.")
    else:
        print("  (b) ACCOPPIAMENTO TRAIN->TEST — le repliche coincidono, ma cambiare lo shard")
        print("      di train sposta il punteggio sul test. E' un difetto della pipeline.")


if __name__ == "__main__":
    main()
