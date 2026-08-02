#!/usr/bin/env python3.10
"""Tabella paper top-1/3/5 per client, arm e build — deep E floor insieme.

    $PY scripts/topk_table.py ucr_001
    $PY scripts/topk_table.py ucr_011 --tag ucr011_v1 --floor-tag ucr011_floor

Perche' esiste. `federated_eval.METRIC_KEYS` e' `vus_pr, auprc, auroc, pate_f1`, quindi il
top-k NON compare nel summary del deep -- anche se la pipeline lo calcola eccome e lo
scrive nel `report.json` di ogni client. Il floor lo mette nei suoi record. Senza questo
script il confronto sulla metrica ufficiale di UCR -- la sola aggregabile fra serie, vedi
la nota su VUS-PR in UCR_WINDOW_ABLATION_SET.md -- semplicemente non esiste.

I numeri NON vengono ricalcolati: si leggono dove la pipeline li ha scritti
(`<arm>/<client>/report.json` per il deep, `records_<build>.jsonl` per il floor). Entrambi
escono da `detect._paper_metrics`, quindi le due meta' della tabella sono confrontabili
per costruzione. Definizione: top-1 = argmax dello score; top-K = i K massimi locali
distanti almeno `tol` (`find_peaks`); hit se una predizione cade entro `tol` da un
positivo. `_paper_metrics` ricade su argmax quando i picchi sono meno di K.
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

ARMS = ["local", "centralized", "federated_cb_only", "federated_cb_only_ema",
        "federated_fedavg_cb_only", "federated", "federated_shared",
        "federated_fedavg_cb_sharedprior"]
KS = (1, 3, 5)


def cell(v) -> str:
    return "—" if v is None or v != v else ("**1**" if v else "0")


def row(name: str, vals: list, width: int) -> str:
    good = [v for v in vals if v is not None and v == v]
    m = f"**{statistics.fmean(good):.2f}**" if good else "—"
    cells = " | ".join(cell(v) for v in vals)
    return f"| {name} | {cells} | {m} |"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cluster")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--floor-tag", default=None)
    ap.add_argument("--cohort", default=None)
    a = ap.parse_args()

    cluster = a.cluster
    cohort = a.cohort or cluster.replace("_", "")
    tag = a.tag or f"{cohort}_v1"
    ftag = a.floor_tag or f"{cohort}_floor"
    deep = REPO / "artifacts/runs" / tag / "ckpt"
    floor = REPO / "artifacts/runs" / ftag / "floor"

    c = json.loads((REPO / f"cohorts/{cohort}.json").read_text())
    builds = []
    for ds, v in c["datasets"].items():
        w = v["window"] if v["window_mode"] == "fixed" else v["windows"].get(cluster)
        builds.append((ds, int(w), int(v["metrics_tolerance"])))
    builds.sort(key=lambda t: t[1])

    print(f"# `{cluster}` — paper top-1 / top-3 / top-5, per client\n")
    print("Metrica ufficiale UCR. **1 = anomalia trovata, 0 = mancata.** Top-1 = l'argmax "
          "dello score; top-K = i K massimi locali distanti almeno `tol`; hit se una "
          "predizione cade entro `tol` dal segmento anomalo.\n")
    print("Letti dove la pipeline li ha scritti — `report.json` per client (deep) e i "
          "record del floor — non ricalcolati. Entrambi da `detect._paper_metrics`.\n")

    for ds, w, tol in builds:
        print(f"\n## `{ds}` — W={w}, tolleranza={tol}\n")
        # deep: {arm: {entity: {k: val}}}
        got: dict[str, dict[str, dict]] = {}
        ents: set[str] = set()
        for arm in ARMS:
            base = deep / ds / cluster / "seed0" / arm
            if not base.is_dir():
                continue
            per = {}
            for rp in sorted(base.glob("*/report.json")):
                r = json.loads(rp.read_text())
                e = rp.parent.name
                per[e] = {k: r.get(f"paper_top{k}_acc_at_{tol}") for k in KS}
                ents.add(e)
            if per:
                got[arm] = per
        if not ents:
            print("*(nessun report.json trovato — run non ancora completo)*")
            continue
        ents_l = sorted(ents)

        hdr = " | ".join(e.split("_")[-1] for e in ents_l)
        print(f"| arm | k | {hdr} | media |")
        print("|---|---|" + "---:|" * (len(ents_l) + 1))
        for arm in ARMS:
            if arm not in got:
                continue
            for k in KS:
                vals = [got[arm].get(e, {}).get(k) for e in ents_l]
                print(row(f"`{arm}` | top-{k}", vals, len(ents_l)))

        # floor: l'arm migliore su top-1, piu' il conteggio di quanti ne trovano almeno una
        fp = floor / f"records_{ds}.jsonl"
        if fp.exists():
            fr: dict[str, dict[str, dict]] = collections.defaultdict(dict)
            for line in fp.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                fr[r["_arm"]][r["_entity"]] = {
                    k: r.get(f"paper_top{k}_acc_at_{tol}") for k in KS}
            best, bestv = None, -1.0
            for arm, per in fr.items():
                v = [per.get(e, {}).get(1) for e in ents_l]
                v = [x for x in v if x is not None and x == x]
                if v and statistics.fmean(v) > bestv:
                    best, bestv = arm, statistics.fmean(v)
            if best:
                for k in KS:
                    vals = [fr[best].get(e, {}).get(k) for e in ents_l]
                    nm = (f"**floor migliore** (`{best.replace('floor_', '')[:22]}`)"
                          if k == 1 else "↑")
                    print(row(f"{nm} | top-{k}", vals, len(ents_l)))
                nz = sum(1 for arm in fr
                         if any(fr[arm].get(e, {}).get(5) for e in ents_l))
                print(f"\n*floor: mostrato l'arm migliore su top-1 fra {len(fr)}. "
                      f"Arm del floor che trovano l'anomalia in top-5 su almeno un client: "
                      f"**{nz}/{len(fr)}**.*")
    return 0


if __name__ == "__main__":
    sys.exit(main())
