#!/usr/bin/env python3
"""authors_check.py — confronto permanente `centralized` vs paper originale (TimeVQVAE-AD).

Gli autori rilasciano le accuratezze top-1/3/5 di tutte 250 le serie NEL NOME dei PNG
(`.released_results/visualizations/<idx>-joint_anomaly_score-acc_<t1>,<t3>,<t5>.png`,
protocollo ±100; ricalcolate dai nomi: 70,8/77,6/82,4% = paper). L'indice `ucr_NNN` ≡ il
loro `dataset_idx` (periodi coincidenti 180/180, verificato 2026-08-03).

Per ogni serie con una cella `centralized` completata confrontiamo il cluster (top-k @100
dai report per-client) con il loro risultato. Bandiere:
  ⚠️ MISS  loro t1=1 e noi t1<0,5  → cella da GUARDARE (config/finestra), non da tabella
  ~ SPLIT  client non unanimi      → risorge la dipendenza dallo shard (non dovrebbe, post z-norm)

  python3.10 scripts/authors_check.py                 # campagna c50 (default)
  python3.10 scripts/authors_check.py --tag zn_es     # altro tag
  python3.10 scripts/authors_check.py --dev           # 10 serie di sviluppo (zn_es→zn_main)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

REL = "/home/leonardo/PhD/TimeVQVAE-AnomalyDetection/.released_results/visualizations"
DEV = ["ucr_001", "ucr_011", "ucr_014", "ucr_043", "ucr_082",
       "ucr_083", "ucr_086", "ucr_170", "ucr_222", "ucr_229"]


def authors_acc() -> dict[int, tuple[int, int, int]]:
    acc = {}
    for p in glob.glob(os.path.join(REL, "*-joint_anomaly_score-acc_*.png")):
        m = re.search(r"(\d+)-joint_anomaly_score-acc_([01]),([01]),([01])\.png$",
                      os.path.basename(p))
        if m:
            acc[int(m.group(1))] = tuple(int(x) for x in (m.group(2), m.group(3), m.group(4)))
    return acc


def our_topk(tag: str, series: str) -> tuple[list[float], int] | None:
    reps = sorted(glob.glob(
        f"artifacts/runs/{tag}/ckpt/ucr_split_w2p/{series}/seed0/centralized*/{series}_p*/report.json"))
    per = [[], [], []]
    for p in reps:
        r = json.load(open(p))
        for i, k in enumerate(("paper_top1_acc_at_100", "paper_top3_acc_at_100",
                               "paper_top5_acc_at_100")):
            per[i].append(float(r[k]))
    if not per[0]:
        return None
    return [sum(x) / len(x) for x in per], len(per[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="c50_central")
    ap.add_argument("--dev", action="store_true", help="10 serie di sviluppo, zn_es poi zn_main")
    a = ap.parse_args()

    rel = authors_acc()
    if not rel:
        raise SystemExit(f"nessun PNG rilasciato trovato in {REL}")

    if a.dev:
        cells = []
        for s in DEV:
            tag = next((t for t in ("zn_es", "zn_main")
                        if os.path.exists(f"artifacts/runs/{t}/ucr_split_w2p/{s}__centralized.json")), None)
            if tag:
                cells.append((tag, s))
    else:
        cells = [(a.tag, os.path.basename(j).split("__")[0])
                 for j in sorted(glob.glob(f"artifacts/runs/{a.tag}/ucr_split_w2p/*__centralized.json"))]

    if not cells:
        print(f"(nessuna cella centralized completata per {'dev' if a.dev else a.tag})")
        return

    print(f"{'serie':<10}{'tag':<12}{'loro t1,3,5':>12}{'nostri t1,3,5':>16}   flag")
    tot_rel = [0, 0, 0]
    tot_our = [0.0, 0.0, 0.0]
    n = flags = 0
    for tag, s in cells:
        idx = int(s.split("_")[1])
        if idx not in rel:
            print(f"{s:<10}{tag:<12}{'(non rilasciata)':>12}")
            continue
        ours = our_topk(tag, s)
        if ours is None:
            print(f"{s:<10}{tag:<12}{','.join(map(str, rel[idx])):>12}{'(report mancanti)':>16}")
            continue
        (t1, t3, t5), nc = ours
        flag = ""
        if rel[idx][0] == 1 and t1 < 0.5:
            flag += " ⚠️ MISS"
        if any(0.0 < v < 1.0 for v in (t1, t3, t5)):
            flag += " ~ SPLIT"
        if flag.strip():
            flags += 1
        n += 1
        for i, v in enumerate(rel[idx]):
            tot_rel[i] += v
        for i, v in enumerate((t1, t3, t5)):
            tot_our[i] += v
        print(f"{s:<10}{tag:<12}{','.join(map(str, rel[idx])):>12}"
              f"{f'{t1:.1f},{t3:.1f},{t5:.1f}':>16}  {flag}")
    if n:
        print(f"\nsul sottoinsieme coperto (n={n}):  loro "
              f"{tot_rel[0]/n:.3f}/{tot_rel[1]/n:.3f}/{tot_rel[2]/n:.3f}   noi "
              f"{tot_our[0]/n:.3f}/{tot_our[1]/n:.3f}/{tot_our[2]/n:.3f}   celle flaggate: {flags}")
        print("(riferimento paper su tutte 250: 0,708/0,776/0,824 — il sottoinsieme può differire)")


if __name__ == "__main__":
    main()
