#!/usr/bin/env python3
"""ucrsplit_predictions_check.py -- verifica le previsioni pre-registrate di ucr_split.

Le previsioni sono state congelate il 2026-07-27 con n=24/82 cluster chiusi
(documentation/ucrsplit_prereg.json, doc umano in documentation/UCRSPLIT_PREDICTIONS.md).
Questo script NON le ricalcola: le legge e le confronta con i dati su disco, così il
confronto resta una verifica e non una ricostruzione a posteriori.

    python3.10 scripts/ucrsplit_predictions_check.py
    python3.10 scripts/ucrsplit_predictions_check.py --metric auprc      # metrica secondaria
    python3.10 scripts/ucrsplit_predictions_check.py --json out.json     # esito macchina

L'unita' di analisi e' IL CLUSTER, non il client: i 5 client di un cluster condividono lo
stesso test set (ICC 0.82, n_eff ~27 per 115 client), quindi n=115 e' pseudo-replicazione.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREREG = os.path.join(REPO, "documentation", "ucrsplit_prereg.json")
RESDIR = os.path.join(REPO, "artifacts", "ucrsplit")
CKPT = os.path.join(RESDIR, "ckpt")
BASE = "local"


def load_results(metric):
    """cluster -> arm -> {entity: score}"""
    by = defaultdict(dict)
    for f in glob.glob(os.path.join(RESDIR, "*.json")):
        stem = os.path.basename(f)[:-5]
        if "__" not in stem:
            continue
        cl, arm = stem.split("__", 1)
        recs = json.load(open(f)).get("records", [])
        vals = {r["_entity"]: r[metric] for r in recs if metric in r}
        if vals:
            by[cl][arm] = vals
    return by


def cluster_deltas(by, clusters, arm):
    """media entro cluster dei delta appaiati per-client -- una osservazione per cluster."""
    return np.array([
        np.mean([by[c][arm][e] - by[c][BASE][e] for e in by[c][arm] if e in by[c][BASE]])
        for c in clusters
    ])


def ci95(v):
    n = len(v)
    if n < 2:
        return float("nan"), float("nan")
    se = v.std(ddof=1) / np.sqrt(n)
    return v.mean() - 1.96 * se, v.mean() + 1.96 * se


def convergence_audit():
    """hard rule 2026-07-24: una riga TRUNCATED non e' riportabile."""
    files = sorted(glob.glob(os.path.join(CKPT, "**", "fed_history.json"), recursive=True))
    trunc, rounds = [], []
    for f in files:
        h = json.load(open(f)).get("stage1", [])
        if not h:
            continue
        rounds.append(h[-1].get("round", 0))
        if h[-1].get("truncated"):
            trunc.append(os.path.relpath(f, CKPT))
    return len(files), trunc, rounds


def verdict_for(arm, obs, lo, hi, pred, p_lo, p_hi, n_new):
    """le regole fissate in UCRSPLIT_PREDICTIONS.md sec.5, applicate meccanicamente.

    n_new = cluster chiusi DOPO il congelamento. A n_new=0 l'osservato coincide per
    costruzione col valore congelato, quindi ogni verdetto sarebbe circolare.
    """
    if n_new == 0:
        return "n/d -- nessun cluster nuovo dal congelamento"
    inside = p_lo <= obs <= p_hi
    if arm == "centralized":
        if obs > 0.25:
            return "SOPRA IL RANGE -- controllare i livelli assoluti, non solo i delta"
        if inside:
            return "CONFERMATA (dentro l'IC previsto)"
        if abs(obs - 0.0862) < 0.02:
            return "MODELLO FALSIFICATO -- il gap e' costante, i 3 corti erano un caso"
        return "FUORI dall'IC previsto"
    crossover = lo > 0
    if crossover:
        return "CROSSOVER CONFERMATO (Δ>0, IC esclude lo zero)"
    if hi < 0:
        return "resta SOTTO local (IC esclude lo zero)"
    return "IC contiene lo zero -- nessun crossover dimostrabile" + ("" if inside else " [e fuori dalla previsione]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default=None, help="default: la metrica congelata nel prereg")
    ap.add_argument("--json", default=None, help="scrive l'esito in un file json")
    args = ap.parse_args()

    pre = json.load(open(PREREG))
    metric = args.metric or pre.get("metric", "vus_pr")
    arms = pre["arms"]
    tl = pre["train_total"]
    frozen_done = set(pre["clusters_done"])
    pending = pre["clusters_pending"]

    by = load_results(metric)
    full = sorted((c for c in tl if all(a in by.get(c, {}) for a in arms)), key=lambda c: tl[c])
    new = [c for c in full if c not in frozen_done]
    still = [c for c in pending if c not in set(full)]

    print(f"=== ucr_split: verifica previsioni pre-registrate  (metrica {metric}) ===")
    print(f"  congelate a n={pre['frozen_at_n']}   ora chiusi n={len(full)}/82"
          f"   nuovi dal congelamento: {len(new)}   ancora pendenti: {len(still)}")
    if still:
        print(f"  !! RUN INCOMPLETO -- le righe sotto NON sono l'esito finale")

    nfiles, trunc, rounds = convergence_audit()
    print(f"\n=== audit di convergenza (hard rule 2026-07-24) ===")
    print(f"  {len(trunc)} TRUNCATED / {nfiles} storie   round mediano "
          f"{int(np.median(rounds)) if rounds else '-'}  max {max(rounds) if rounds else '-'}")
    for t in trunc[:10]:
        print(f"    TRUNCATED, NON RIPORTABILE: {t}")

    if len(full) < 2:
        print("\ntroppi pochi cluster completi per un confronto")
        return

    print(f"\n=== previsto vs osservato  (Δ = arm - {BASE}, unita' = cluster, n={len(full)}) ===")
    print(f"{'arm':13s} {'n24':>8s} {'pred n82':>9s} {'IC95 pred':>17s} | "
          f"{'OSSERVATO':>10s} {'IC95 oss':>17s}  {'p':>6s}  verdetto")
    rows = {}
    for arm in arms:
        if arm == BASE:
            continue
        pr = pre["predictions"][arm]
        v = cluster_deltas(by, full, arm)
        lo, hi = ci95(v)
        p = stats.wilcoxon(v).pvalue if len(v) >= 6 else float("nan")
        p_lo, p_hi = pr["ci95"]
        vd = verdict_for(arm, float(v.mean()), lo, hi, pr["pred_pooled_n82"], p_lo, p_hi, len(new))
        print(f"{arm:13s} {pr['observed_n24']:+8.3f} {pr['pred_pooled_n82']:+9.3f} "
              f"[{p_lo:+.3f},{p_hi:+.3f}] | {v.mean():+10.3f} [{lo:+.3f},{hi:+.3f}] "
              f"{p:6.3f}  {vd}")
        rows[arm] = {"pred": pr["pred_pooled_n82"], "pred_ci95": [p_lo, p_hi],
                     "observed": float(v.mean()), "ci95": [lo, hi], "p_wilcoxon": float(p),
                     "n": len(v), "verdict": vd}

    # il meccanismo: il gap dipende ancora dalla lunghezza?
    x = np.log10([tl[c] for c in full])
    print(f"\n=== meccanismo: Δ ~ log10(train totale)   (al congelamento era "
          f"{pre['design_caveat']['slope_all24']:+.3f}/decade, p={pre['design_caveat']['slope_all24_p']:.3f}) ===")
    slopes = {}
    for arm in arms:
        if arm == BASE:
            continue
        r = stats.linregress(x, cluster_deltas(by, full, arm))
        flag = "  <-- significativo" if r.pvalue < 0.05 else ""
        print(f"  {arm:13s} slope={r.slope:+.4f}/decade  p={r.pvalue:.3f}  r2={r.rvalue**2:.2f}{flag}")
        slopes[arm] = {"slope": float(r.slope), "p": float(r.pvalue), "r2": float(r.rvalue ** 2)}

    # il buco del disegno si e' chiuso?
    plo, phi = pre["design_caveat"]["pending_range"]
    inside = [c for c in full if plo <= tl[c] <= phi]
    print(f"\n=== il buco bimodale del disegno ===")
    print(f"  cluster chiusi nel range [{plo}, {phi}]: {len(inside)}  (era 0 al congelamento)")
    if len(inside) >= 6:
        vi = cluster_deltas(by, inside, "centralized")
        vh = cluster_deltas(by, [c for c in full if tl[c] > phi], "centralized")
        lo_i, hi_i = ci95(vi)
        print(f"  Δ(central-local) nel range medio: {vi.mean():+.3f} [{lo_i:+.3f},{hi_i:+.3f}] (n={len(vi)})")
        print(f"  Δ(central-local) clump lungo    : {vh.mean():+.3f} (n={len(vh)})")
        print(f"  atteso dai 3 corti al congelamento: +{pre['design_caveat']['delta_central_short_clump']:.3f}")
    else:
        print(f"  ancora insufficiente: la pendenza resta guidata dai 3 cluster corti "
              f"({pre['design_caveat']['leverage_share_short_clump']:.0%} della leva)")

    # livelli assoluti: il controllo che distingue "gap reale" da "local che collassa"
    print(f"\n=== livelli assoluti (media per cluster) -- distingue gap reale da collasso di {BASE} ===")
    for arm in (BASE, "centralized"):
        lv = np.array([np.mean(list(by[c][arm].values())) for c in full])
        pr = pre["levels_pred_on_short"].get(arm)
        print(f"  {arm:13s} media {lv.mean():.3f}  mediana {np.median(lv):.3f}"
              + (f"   (predetta sui corti: {pr:.3f})" if pr else ""))

    if args.json:
        json.dump({"metric": metric, "n_done": len(full), "n_pending": len(still),
                   "truncated": trunc, "arms": rows, "slopes": slopes,
                   "n_inside_design_gap": len(inside)},
                  open(args.json, "w"), indent=2)
        print(f"\nesito scritto in {args.json}")


if __name__ == "__main__":
    main()
