#!/usr/bin/env python3.10
"""Aggiunge `paper_top{k}_acc_at_100` ai `report.json` gia' su disco, rileggendo i punteggi.

    $PY scripts/backfill_protocol_topk.py --dry
    nice -n 19 $PY scripts/backfill_protocol_topk.py

Perche' esiste. `detect.PROTOCOL_TOLERANCE` fa emettere le due colonne a ogni lancio NUOVO,
ma i run gia' fatti hanno solo quella alla tolleranza fissata (64). Rigirarli costerebbe
giorni di GPU per un numero che si ricava dai `scores.npz` gia' salvati: il top-k e' una
funzione pura di (labels, scores, tol). Qui si ricalcola e si aggiunge, senza toccare nulla
di esistente.

⚠️ NON cambiare `metrics_tolerance` nelle coorti per ottenere lo stesso effetto: sta dentro
il fingerprint (`scripts/cohort.py:168`), quindi orfanerebbe ogni cella gia' su disco.

⚠️ Il FLOOR non e' recuperabile per questa via: i suoi record non salvano i punteggi grezzi,
solo le metriche gia' calcolate. Va rigirato (i floor nuovi prendono le due colonne da soli).

Sicurezza: additivo e idempotente. Le chiavi esistenti non vengono mai toccate; se le chiavi
@100 ci sono gia' il file viene saltato. Scrittura atomica via file temporaneo + rename, cosi'
un'interruzione non lascia un report.json troncato.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO), str(REPO / "pipeline")]

from detect import PROTOCOL_TOLERANCE, _topk_at  # noqa: E402

KS = (1, 3, 5)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(REPO / "artifacts" / "runs"))
    ap.add_argument("--tol", type=int, default=PROTOCOL_TOLERANCE)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    done = skip = miss = 0
    changed: list[tuple[str, str, float, float]] = []
    for rp in sorted(Path(a.runs).glob("*/ckpt/*/*/seed0/*/*/report.json")):
        try:
            rep = json.loads(rp.read_text())
        except Exception:
            continue
        if f"paper_top1_acc_at_{a.tol}" in rep:
            skip += 1
            continue
        sp = rp.parent / "scores.npz"
        if not sp.exists():
            miss += 1
            continue
        z = np.load(sp, allow_pickle=True)
        if "test_scores" not in z or "test_labels" not in z:
            miss += 1
            continue
        sc = np.asarray(z["test_scores"], dtype=float)
        pos = np.flatnonzero(np.asarray(z["test_labels"]) > 0)
        new = ({f"paper_top{k}_acc_at_{a.tol}": float("nan") for k in KS}
               if sc.size == 0 or pos.size == 0 else _topk_at(pos, sc, a.tol, list(KS)))
        # confronto con la colonna alla tolleranza fissata, per vedere se il righello muove
        old1 = next((v for k, v in rep.items()
                     if k.startswith("paper_top1_acc_at_") and not k.endswith(str(a.tol))), None)
        n1 = new[f"paper_top1_acc_at_{a.tol}"]
        if old1 is not None and old1 == old1 and n1 == n1 and old1 != n1:
            changed.append((str(rp.relative_to(a.runs)), rp.parent.name, old1, n1))
        if not a.dry:
            rep.update(new)
            tmp = rp.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(rep, indent=2))
            os.replace(tmp, rp)          # atomico: nessun report.json a meta'
        done += 1

    verb = "da aggiornare" if a.dry else "aggiornati"
    print(f"{verb}: {done}   gia' presenti: {skip}   senza scores.npz: {miss}")
    print(f"celle in cui top-1 CAMBIA passando a tol={a.tol}: {len(changed)}")
    for p, ent, o, n in changed[:20]:
        print(f"   {o:.0f} -> {n:.0f}   {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
