#!/usr/bin/env python3.10
"""paper_table.py — la tabella del paper: paper top-k, aggregata FRA serie.

    $PY scripts/paper_table.py                       # tutte le serie trovate su disco
    $PY scripts/paper_table.py --series ucr_001,ucr_011
    $PY scripts/paper_table.py --k 1 --delta-null    # normalizzato sul controllo nullo

Perche' esiste, e perche' NON usa VUS-PR. `ucr_series_report.py` gira su VUS-PR, che su una
serie e' legittimo ma fra serie UCR non e' aggregabile: sugli stessi punteggi AUPRC 0.994 puo'
diventare VUS-PR 0.211, perche' la normalizzazione dipende dalla lunghezza del segmento
anomalo, che fra serie UCR varia di ordini di grandezza. Mediare VUS-PR su 10 serie produce un
numero che non significa niente. `paper_top1` invece e' binario per (serie, client) -- l'anomalia
l'hai trovata o no -- quindi la media E' una frequenza, ed e' la metrica ufficiale di UCR.

Unita' di analisi. I 5 client di una serie CONDIVIDONO il test set (ICC 0.82): sono 5 modelli
valutati sugli stessi dati, non 5 misure indipendenti. Quindi si mediano i client DENTRO la
serie -- ottenendo l'accuratezza attesa di un client a caso -- e la serie e' l'unita' per
qualunque statistica fra serie. Contare i client come n gonfia i p di sqrt(5).

I numeri non si ricalcolano: si leggono da `<arm>/<client>/report.json`, dove la pipeline ha
gia' scritto `paper_top{k}_acc_at_{tol}` via `detect._paper_metrics`.
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "artifacts" / "runs"

# Ordine di presentazione. Gli arm del fattoriale 2x2 prima, poi il blocco encoder.
ARM_ORDER = [
    "local", "centralized",
    "federated_cb_only", "federated_cb_only_ema", "federated_fedavg_cb_only",
    "federated", "federated_shared", "federated_fedavg_cb_sharedprior",
    "federated_enc_commoninit", "federated_enc_fedavg",
    "federated_enc_fedprox", "federated_enc_fedproto",
]
# Il controllo nullo del blocco encoder: stessa init, nessuna comunicazione. Δ-null misura
# quanto rende la federazione al netto di cio' che si ottiene gratis dall'init condivisa.
NULL_ARM = "federated_enc_commoninit"


def strip_knobs(arm: str) -> str:
    """`federated_enc_fedproto_lam1_uniform` -> `federated_enc_fedproto`. Le manopole
    finiscono nel nome della dir; l'arm logico e' il prefisso."""
    for base in sorted(ARM_ORDER, key=len, reverse=True):
        if arm == base or arm.startswith(base + "_"):
            return base
    return arm


def variant(arm: str, tag: str) -> str:
    """Etichetta la riga quando lo stesso arm compare in piu' ablazioni: senza questo,
    `federated_cb_only` a K=64 e a K=128 collasserebbero sulla stessa riga."""
    bits = []
    if arm != strip_knobs(arm):
        bits.append(arm[len(strip_knobs(arm)) + 1:])
    for key, lab in (("cb128", "K=128"), ("proto_count", "agg=count")):
        if tag.endswith(key):
            bits.append(lab)
    return " ".join(bits)


def collect(series: list[str] | None) -> tuple[dict, dict, list, list]:
    """-> rows[(build, arm_label)][serie] = media sui client; tol[build]; serie; build."""
    rows: dict = collections.defaultdict(dict)
    tols: dict = {}
    seen_series, seen_builds = set(), set()
    for tag_dir in sorted(RUNS.iterdir()):
        ck = tag_dir / "ckpt"
        if not ck.is_dir():
            continue
        tag = tag_dir.name
        for rp in ck.glob("*/*/seed0/*/*/report.json"):
            ds, cl, _, arm_dir, ent = rp.parts[-6:-1]
            if series and cl not in series:
                continue
            try:
                r = json.loads(rp.read_text())
            except Exception:
                continue
            tol = next((int(k.rsplit("_", 1)[1]) for k in r
                        if k.startswith("paper_top1_acc_at_")), None)
            if tol is None:
                continue
            base = strip_knobs(arm_dir)
            v = variant(arm_dir, tag)
            label = f"{base} [{v}]" if v else base
            key = (ds, label)
            rows[key].setdefault(cl, {}).setdefault(ent, {})
            rows[key][cl][ent] = {k: r.get(f"paper_top{k}_acc_at_{tol}") for k in (1, 3, 5)}
            tols[ds] = tol
            seen_series.add(cl)
            seen_builds.add(ds)
    return rows, tols, sorted(seen_series), sorted(seen_builds)


def per_series(cell: dict, k: int) -> float | None:
    """Media sui client di UNA serie. I client condividono il test set, quindi questa e'
    l'accuratezza attesa di un client a caso -- non una media di misure indipendenti."""
    v = [d.get(k) for d in cell.values()]
    v = [x for x in v if x is not None and x == x]
    return statistics.fmean(v) if v else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default=None, help="es. ucr_001,ucr_011 (default: tutte)")
    ap.add_argument("--k", type=int, default=1, choices=(1, 3, 5))
    ap.add_argument("--delta-null", action="store_true",
                    help=f"sottrai `{NULL_ARM}` della stessa serie e build")
    a = ap.parse_args()
    sel = a.series.split(",") if a.series else None

    rows, tols, series, builds = collect(sel)
    if not series:
        print("nessun report.json trovato", file=sys.stderr)
        return 1
    k = a.k

    print(f"# Tabella paper — `paper_top{k}`, aggregata fra serie\n")
    print(f"Metrica ufficiale UCR: **1 = anomalia trovata, 0 = mancata**. Ogni cella è la "
          f"media sui 5 client di quella serie — i client condividono il test set (ICC 0.82), "
          f"quindi la media è l'accuratezza attesa di un client a caso, e **l'unità per "
          f"qualunque statistica fra serie è la serie, non il client**.\n")
    print(f"VUS-PR non compare apposta: [non è aggregabile fra serie UCR]"
          f"(UCR_WINDOW_ABLATION_SET.md). Letti da `report.json`, non ricalcolati.\n")
    if a.delta_null:
        print(f"**Δ-null**: sottratto `{NULL_ARM}` (stessa init, zero comunicazione) della "
              f"stessa serie e build. Isola quanto rende la federazione al netto "
              f"dell'inizializzazione condivisa.\n")

    for ds in builds:
        ws = {s: None for s in series}
        print(f"\n## `{ds}` — tolleranza {tols.get(ds, '?')}\n")
        labels = [lab for (d, lab) in rows if d == ds]
        labels.sort(key=lambda L: (ARM_ORDER.index(strip_knobs(L.split(" [")[0]))
                                   if strip_knobs(L.split(" [")[0]) in ARM_ORDER else 99, L))
        null = {s: per_series(rows.get((ds, NULL_ARM), {}).get(s, {}), k)
                for s in series} if a.delta_null else {}
        print("| arm | " + " | ".join(f"`{s}`" for s in series) + " | media | n |")
        print("|---|" + "---:|" * (len(series) + 2))
        for lab in labels:
            if a.delta_null and lab == NULL_ARM:
                continue
            vals = []
            for s in series:
                v = per_series(rows[(ds, lab)].get(s, {}), k)
                if a.delta_null and v is not None:
                    n = null.get(s)
                    v = None if n is None else v - n
                vals.append(v)
            got = [v for v in vals if v is not None]
            if not got:
                continue
            cells = " | ".join("—" if v is None else
                               (f"{v:+.2f}" if a.delta_null else f"{v:.2f}") for v in vals)
            m = statistics.fmean(got)
            ms = f"**{m:+.3f}**" if a.delta_null else f"**{m:.3f}**"
            print(f"| `{lab}` | {cells} | {ms} | {len(got)} |")
        if len(series) < 3:
            print(f"\n⚠️ **{len(series)} serie**: la colonna «media» è una media di "
                  f"{len(series)} punti, non una stima. Nessun ordinamento fra arm è "
                  f"distinguibile dal rumore a questo n.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
