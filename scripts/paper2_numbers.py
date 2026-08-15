#!/usr/bin/env python3.10
"""paper2_numbers.py — ricalcola e VERIFICA ogni numero di documentation/paper2.tex
che non era gia' letto da un report della pipeline.

    $PY scripts/paper2_numbers.py                    # verifica, exit 1 se qualcosa non torna
    $PY scripts/paper2_numbers.py --json evidence/paper2_numbers.json

Perche' esiste. I numeri del paper hanno due provenienze diverse e la differenza conta:

  (1) LETTI dalla pipeline — le tabelle II/III/V escono da `summary.<arm>.auprc.mean` nei
      json di `artifacts/runs/<tag>/ucr_split_w2p/<serie>__<arm>.json` e da
      `paper_top1_acc_at_64` nei `report.json`. Quelli non si ricalcolano: si leggono.

  (2) AGGREGATI sopra la pipeline — conteggi vinte/perse, mediane, sign test, range, e
      TUTTA la fusione in function-space. Questi erano stati calcolati a mano, una volta,
      e non erano riproducibili da nessuno script. Questo file colma quel buco: ogni
      asserzione qui sotto e' una frase del paper, con il valore che ci sta scritto.

⚠️ Il top-1 della fusione NON viene da un report: la pipeline non lo scrive, perche' la
fusione non e' un arm addestrato. Lo ricalcoliamo con la regola di `detect._paper_metrics`
(argmax del profilo, hit se entro `TOL` da un timestep positivo). La validazione che
autorizza a usarlo: ricalcolato per-client sui 5 `local` deve riprodurre ESATTAMENTE la
colonna `local` della Tabella II. Se quel check fallisce, ogni numero di fusione qui e'
sospetto e lo script esce 1.

⚠️ Asimmetria dichiarata: la fusione produce UN numero per cluster, gli arm addestrati la
MEDIA sui 5 client. Il confronto fusione-vs-(g) e' quindi "artefatto dispiegabile contro
client medio", che e' la lettura giusta ma non e' un pareggio di unita'. Sul top-1 il punto
non si pone: (g) vale 0 o 1 su tutte e 10 le serie di sviluppo (nessun client dissente).
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import statistics
import sys

import numpy as np
from sklearn.metrics import average_precision_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(REPO, "artifacts", "runs")
DATASET = "ucr_split_w2p"
TOL = 64
SEED = 0

DEV = ["ucr_011", "ucr_014", "ucr_043", "ucr_170",
       "ucr_001", "ucr_083", "ucr_086", "ucr_082", "ucr_222", "ucr_229"]
DISC = ["ucr_011", "ucr_014", "ucr_043", "ucr_170"]

# arm del paper -> (tag, nome arm su disco)
G = ("zn_a2", "federated_enc_fedavg")            # (g) = A2
A1 = ("zn_a1", "federated_enc_fedavg")           # (f)
LOCAL = ("zn_main", "local")                     # (a)
CENTR = ("zn_main", "centralized")               # (b)
LOCAL_OT = ("zn_ot", "local")                    # baseline sovrallenata
CB_SUF = ("zn_main", "federated_cb_only")        # (c)
CB_FA = ("zn_main", "federated_fedavg_cb_only")  # (d)


# ---------------------------------------------------------------- lettura pipeline
def auprc_table() -> dict[tuple[str, str], dict[str, float]]:
    """(tag, arm) -> {serie: auprc medio sui client}, come lo legge il paper."""
    out: dict[tuple[str, str], dict[str, float]] = collections.defaultdict(dict)
    for f in glob.glob(os.path.join(RUNS, "*", DATASET, "*__*.json")):
        tag = f.split(os.sep)[-3]
        series = os.path.basename(f)[:-5].split("__", 1)[0]
        with open(f) as fh:
            d = json.load(fh)
        for arm, m in d.get("summary", {}).items():
            if "auprc" in m:
                out[(tag, arm)][series] = m["auprc"]["mean"]
    return out


def top1_from_reports(tag: str, series: str) -> float | None:
    """Media per-client di `paper_top1_acc_at_64`, la colonna accuracy@64 del paper."""
    vals = []
    pat = os.path.join(RUNS, tag, "ckpt", DATASET, series, f"seed{SEED}", "*", "*", "report.json")
    for f in glob.glob(pat):
        with open(f) as fh:
            d = json.load(fh)
        k = [x for x in d if x.startswith(f"paper_top1_acc_at_{TOL}")]
        if k:
            vals.append(d[k[0]])
    return statistics.mean(vals) if vals else None


# ---------------------------------------------------------------- fusione
def load_cluster(tag: str, arm: str, series: str):
    """Profili di test dei 5 client. Stessa ricetta di scripts/fusion_probe.py."""
    pat = os.path.join(RUNS, tag, "ckpt", DATASET, series, f"seed{SEED}", arm,
                       f"{series}_p*", "scores.npz")
    paths = sorted(glob.glob(pat))
    if not paths:
        return None, None
    S, y = [], None
    for p in paths:
        z = np.load(p)
        S.append(z["test_scores"])
        y = z["test_labels"]
    return np.asarray(S), (y > 0).astype(int)


def zscore(S: np.ndarray) -> np.ndarray:
    sd = S.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (S - S.mean(axis=1, keepdims=True)) / sd


def top1_hit(score: np.ndarray, pos: np.ndarray, tol: int = TOL) -> float:
    """detect._paper_metrics, k=1: argmax, hit entro tol da un positivo."""
    return float(np.min(np.abs(int(np.argmax(score)) - pos)) <= tol)


def fusion_row(tag: str, arm: str, series: str) -> dict | None:
    S, y = load_cluster(tag, arm, series)
    if S is None or y is None or y.sum() == 0:
        return None
    pos = np.flatnonzero(y)
    Z = zscore(S)
    fused = Z.mean(axis=0)
    per = [float(average_precision_score(y, s)) for s in S]
    corr = np.corrcoef(Z)
    iu = np.triu_indices(len(S), 1)
    return {
        "series": series,
        "n_client": len(S),
        "fusion_auprc": float(average_precision_score(y, fused)),
        "per_client_auprc_mean": float(np.mean(per)),
        "best_client_auprc": float(np.max(per)),
        "fusion_top1": top1_hit(fused, pos),
        "per_client_top1_mean": float(np.mean([top1_hit(s, pos) for s in S])),
        "corr_mean": float(corr[iu].mean()),
    }


# ---------------------------------------------------------------- statistica
def sign_p(w: int, l: int) -> float:
    """Sign test a DUE code, pareggi esclusi. E' la convenzione della Tabella III."""
    n = w + l
    if n == 0:
        return float("nan")
    k = min(w, l)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def wl(diffs) -> tuple[int, int, int]:
    d = list(diffs)
    w = sum(1 for x in d if x > 0)
    l = sum(1 for x in d if x < 0)
    return w, l, len(d) - w - l


# ---------------------------------------------------------------- verifica
class Check:
    def __init__(self) -> None:
        self.rows: list[tuple[bool, str, str, str]] = []

    def eq(self, label: str, got, want, tol=0.0, section="") -> None:
        if isinstance(got, float) and isinstance(want, float):
            ok = abs(got - want) <= tol
            g, w = f"{got:.4f}", f"{want:.4f}"
        else:
            ok = got == want
            g, w = str(got), str(want)
        self.rows.append((ok, section, label, f"calcolato {g} · nel paper {w}"))

    def report(self) -> int:
        bad = 0
        sec = None
        for ok, section, label, detail in self.rows:
            if section != sec:
                print(f"\n--- {section}")
                sec = section
            print(f"  [{'OK ' if ok else 'FAIL'}] {label}: {detail}")
            bad += (not ok)
        print(f"\n{len(self.rows) - bad}/{len(self.rows)} verificati" +
              ("" if not bad else f"  ⚠️  {bad} NON TORNANO"))
        return bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default=None, help="scrivi qui i valori grezzi ricalcolati")
    args = ap.parse_args()

    A = auprc_table()
    for key in (G, A1, LOCAL, CENTR, CB_SUF, CB_FA):
        missing = [s for s in DEV if s not in A.get(key, {})]
        if missing:
            sys.exit(f"mancano serie per {key}: {missing}")

    c = Check()

    # --- gate: il nostro top-1 deve riprodurre la colonna `local` della Tabella II ------
    fus = {s: fusion_row(*LOCAL, s) for s in DEV}
    paper_local_top1 = {"ucr_011": 0.40, "ucr_014": 0.80, "ucr_043": 0.20, "ucr_170": 0.20,
                        "ucr_001": 1.00, "ucr_083": 1.00, "ucr_086": 1.00,
                        "ucr_082": 0.00, "ucr_222": 0.00, "ucr_229": 0.00}
    gate_ok = True
    for s in DEV:
        got = fus[s]["per_client_top1_mean"]
        c.eq(f"top-1 per-client {s}", got, paper_local_top1[s], 1e-9,
             "GATE — la nostra regola top-1 riproduce la colonna `local` (Tab. II)")
        gate_ok &= abs(got - paper_local_top1[s]) < 1e-9

    # --- VI-A: minimo del sign test a due code su 4 serie -----------------------------
    c.eq("min p a due code, n=4", sign_p(4, 0), 0.125, 1e-9,
         "VI-A — «its minimum attainable value is 0.125»")

    # --- VI-B: centralized contro il MIGLIOR client ------------------------------------
    best_local = {s: fusion_row(*LOCAL, s)["best_client_auprc"] for s in DEV}
    d_best = [A[CENTR][s] - best_local[s] for s in DEV]
    w, l, _ = wl(d_best)
    c.eq("vinte", w, 7, section="VI-B — «exceeds it on 7 of 10 series in AUPRC (p=0.34)»")
    c.eq("p (due code)", sign_p(w, l), 0.34, 0.005, "VI-B — «exceeds it on 7 of 10 series in AUPRC (p=0.34)»")

    # --- VI-C: (g) contro local, e contro la baseline sovrallenata ---------------------
    d_g_loc = [A[G][s] - A[LOCAL][s] for s in DEV]
    w, l, _ = wl(d_g_loc)
    c.eq("vinte", w, 9, section="VI-C — «improves 9 of 10 series (p=0.021)» e le mediane")
    c.eq("p", sign_p(w, l), 0.021, 0.001, "VI-C — «improves 9 of 10 series (p=0.021)» e le mediane")
    c.eq("mediana vs baseline a pazienza", statistics.median(d_g_loc), 0.021, 0.0005,
         "VI-C — «improves 9 of 10 series (p=0.021)» e le mediane")
    d_g_ot = [A[G][s] - A[LOCAL_OT][s] for s in DEV if s in A[LOCAL_OT]]
    w_ot, l_ot, _ = wl(d_g_ot)
    c.eq("mediana vs baseline sovrallenata", statistics.median(d_g_ot), 0.022, 0.0005,
         "VI-C — «improves 9 of 10 series (p=0.021)» e le mediane")
    c.eq("conteggio vs sovrallenata", (w_ot, l_ot), (6, 4),
         section="VI-C — «improves 9 of 10 series (p=0.021)» e le mediane")

    # «the gains concentrated on three (+0.42,+0.46,+0.30) and six of them below 0.036»:
    # dei 9 guadagni si tolgono i TRE grandi, i sei restanti devono stare sotto 0.036.
    gains = sorted((x for x in d_g_loc if x > 0), reverse=True)
    S036 = "VI-C — «six of them below 0.036»"
    c.eq("guadagni positivi", len(gains), 9, section=S036)
    c.eq("i tre grandi", [round(x, 2) for x in gains[:3]], [0.46, 0.42, 0.30], section=S036)
    c.eq("i sei restanti", len(gains[3:]), 6, section=S036)
    c.eq("il piu' grande dei sei", max(gains[3:]), 0.0353, 0.0005, S036)
    c.eq("tutti e sei sotto 0.036", all(x < 0.036 for x in gains[3:]), True, section=S036)

    # --- VI-D: regola del codebook, (c) contro (d) -------------------------------------
    d_cd = [A[CB_SUF][s] - A[CB_FA][s] for s in DEV]
    w, l, _ = wl(d_cd)
    c.eq("vinte/perse", (w, l), (5, 5), section="VI-D — «(c)-(d) run from -0.946 to +0.292»")
    c.eq("minimo", min(d_cd), -0.946, 0.001, "VI-D — «(c)-(d) run from -0.946 to +0.292»")
    c.eq("massimo", max(d_cd), 0.292, 0.001, "VI-D — «(c)-(d) run from -0.946 to +0.292»")

    # --- VI-E: fusione in function-space -----------------------------------------------
    S = "VI-E — fusione: conteggi, correlazioni, testa a testa con (g)"
    c.eq("ucr_170 fusione", fus["ucr_170"]["fusion_auprc"], 0.698, 0.0005, S)
    c.eq("ucr_170 miglior client", fus["ucr_170"]["best_client_auprc"], 0.588, 0.0005, S)
    c.eq("ucr_170 (g)", A[G]["ucr_170"], 0.086, 0.0005, S)

    d_mean = [fus[s]["fusion_auprc"] - fus[s]["per_client_auprc_mean"] for s in DEV]
    w, l, _ = wl(d_mean)
    c.eq("batte la media per-client", w, 7, section=S)
    c.eq("mediana del guadagno", statistics.median(d_mean), 0.027, 0.0005, S)
    d_bst = [fus[s]["fusion_auprc"] - fus[s]["best_client_auprc"] for s in DEV]
    c.eq("batte il MIGLIOR client", wl(d_bst)[0], 1, section=S)

    corrs = [fus[s]["corr_mean"] for s in DEV]
    c.eq("corr su ucr_170", fus["ucr_170"]["corr_mean"], 0.56, 0.005, S)
    c.eq("ucr_170 e' il minimo", min(corrs), fus["ucr_170"]["corr_mean"], 1e-12, S)
    c.eq("mediana delle correlazioni", statistics.median(corrs), 0.86, 0.005, S)

    d_fg = [fus[s]["fusion_auprc"] - A[G][s] for s in DEV]
    w, l, _ = wl(d_fg)
    c.eq("fusione vs (g): vinte/perse", (w, l), (3, 7), section=S)
    c.eq("fusione vs (g): mediana", statistics.median(d_fg), -0.006, 0.0005, S)
    for s, want in [("ucr_043", -0.53), ("ucr_011", -0.39), ("ucr_014", -0.23)]:
        c.eq(f"fusione vs (g) su {s}", fus[s]["fusion_auprc"] - A[G][s], want, 0.005, S)
    # le altre 4 sconfitte sono trascurabili: il paper dice «le GRANDI sono su queste tre»
    other = [fus[s]["fusion_auprc"] - A[G][s] for s in DEV
             if s not in ("ucr_043", "ucr_011", "ucr_014") and fus[s]["fusion_auprc"] < A[G][s]]
    c.eq("le altre sconfitte sono sotto 0.02", max(abs(x) for x in other) < 0.02, True, section=S)

    g_top1 = {s: top1_from_reports(G[0], s) for s in DEV}
    c.eq("(g) e' binario su tutte le 10 (unita' confrontabile)",
         all(g_top1[s] in (0.0, 1.0) for s in DEV), True, section=S)
    d_t1 = [fus[s]["fusion_top1"] - g_top1[s] for s in DEV]
    c.eq("top-1 fusione vs (g): V/S/P", wl(d_t1), (1, 1, 8), section=S)

    # --- VI-C: il ritmo dello stage 2. Il riferimento e' il CTRL, non (g) --------------
    # Se questo contrasto venisse appaiato a zn_a2 invece che a zn_a2s2_ctrl i numeri
    # cambierebbero: e' la ragione per cui il paper nomina esplicitamente il riferimento.
    ST = "VI-C — tau64 contro il riferimento a oracolo congelato (NON contro (g))"
    TAU, CTRL, LEP = (("zn_a2s2_tau64", "federated_enc_fedavg"),
                      ("zn_a2s2_ctrl", "federated_enc_fedavg"),
                      ("zn_a2s2_lep", "federated_enc_fedavg"))
    d_tau = [A[TAU][s] - A[CTRL][s] for s in DEV if s in A.get(TAU, {}) and s in A.get(CTRL, {})]
    w, l, _ = wl(d_tau)
    c.eq("tau64 vs ctrl: vinte/perse", (w, l), (4, 6), section=ST)
    c.eq("tau64 vs ctrl: mediana", statistics.median(d_tau), -0.001, 0.0005, ST)
    for s, want in [("ucr_011", -0.198), ("ucr_043", -0.245)]:
        if s in A.get(LEP, {}):
            c.eq(f"31 epoche locali vs ctrl su {s}", A[LEP][s] - A[CTRL][s], want, 0.001, ST)

    # --- VI-C: (h) sul campione di conferma, dove il ctrl NON esiste -------------------
    SC = "VI-C — (h) sulle 48 di conferma: li' l'oracolo viaggia con tau"
    c50a2 = {s: v for s, v in A.get(("c50_a2", "federated_enc_fedavg"), {}).items()}
    c50t = {s: v for s, v in A.get(("c50_tau64", "federated_enc_fedavg"), {}).items()}
    common = sorted(set(c50a2) & set(c50t))
    c.eq("serie appaiate", len(common), 48, section=SC)
    d_c50 = [c50t[s] - c50a2[s] for s in common]
    c.eq("AUPRC vinte/perse", wl(d_c50)[:2], (25, 23), section=SC)
    c.eq("AUPRC p", sign_p(*wl(d_c50)[:2]), 0.89, 0.005, SC)
    t_a2 = {s: top1_from_reports("c50_a2", s) for s in common}
    t_t = {s: top1_from_reports("c50_tau64", s) for s in common}
    d_t = [t_t[s] - t_a2[s] for s in common]
    c.eq("top-1 V/S/P", wl(d_t), (2, 4, 42), section=SC)
    c.eq("top-1 p", sign_p(*wl(d_t)[:2]), 0.69, 0.005, SC)

    # --- Tabella II: ucr_082 e' al pavimento -------------------------------------------
    v82 = {"local": A[LOCAL]["ucr_082"], "(g)": A[G]["ucr_082"],
           "centr": A[CENTR]["ucr_082"]}
    c.eq("tutti sotto 1e-4", max(v82.values()) < 1e-4, True,
         section="Tab. II — «on ucr_082 every arm is at floor ... fifth decimal»")
    c.eq("(g) batte local alla quinta cifra", (A[G]["ucr_082"] - A[LOCAL]["ucr_082"]) > 0, True,
         section="Tab. II — «on ucr_082 every arm is at floor ... fifth decimal»")

    bad = c.report()
    if not gate_ok:
        print("\n⛔ GATE FALLITO: la regola top-1 non riproduce la colonna `local`. "
              "Ogni numero di fusione qui sopra e' da considerarsi non validato.")

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        payload = {
            "dataset": DATASET, "seed": SEED, "tolerance": TOL,
            "series": DEV, "discriminating": DISC,
            "auprc": {f"{t}:{a}": {s: A[(t, a)][s] for s in DEV}
                      for (t, a) in (G, A1, LOCAL, CENTR, CB_SUF, CB_FA)},
            "auprc_local_overtrained": {s: A[LOCAL_OT][s] for s in DEV if s in A[LOCAL_OT]},
            "fusion": {s: fus[s] for s in DEV},
            "top1_g": g_top1,
            "checks_failed": bad,
        }
        with open(args.json, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nscritto {args.json}")

    sys.exit(1 if (bad or not gate_ok) else 0)


if __name__ == "__main__":
    main()
