"""IL BUDGET DI CALCOLO DI OGNI CELLA — l'unica cosa che rende «equo» un confronto.

Finora il budget non era misurato da nessuna parte: si assumeva. Il 2026-08-06 quella
assunzione si e' rivelata sbagliata **di segno**, non di grado — credevo che A2 usasse 2,2x
gli step di `local` e usa lo 0,50-0,81x proprio sulle serie dove vince di piu'.

## Le tre unita', e perche' danno risposte diverse

  step/CLIENT   quanto vede ogni modello. E' l'asse su cui si chiede «e' allenato quanto
                l'altro?». Con 5 client, un arm federato a N step per client ne fa 5N di
                calcolo totale ma ogni modello ne vede N.
  step TOTALI   il conto del calcolo, sommato sui client. E' l'asse su cui si chiede
                «quanto e' costato?». Un arm federato costa 5x a parita' di step/client.
  ROUND         il costo di COMUNICAZIONE, che `local` ha esattamente zero. Non e'
                convertibile negli altri due e va riportato a parte.

Riportarne una sola sceglie implicitamente una tesi. Si riportano tutte e tre.

## Come si contano, e perche' non allo stesso modo

`local`/`centralized` passano da `_converged_loop`, che logga per client
`[s1 <ent>] converged protocol: ... (N batches/epoch)` e poi `ep E step S`: l'ultimo `step S`
E' il conteggio, esatto.

Gli arm federati NON ci passano per il training di round: ogni round allena `local_epochs`
epoche piene per client, quindi

    step/client = round x local_epochs x batches_per_epoch(client)

⚠️ `batches_per_epoch` e' PER CLIENT, non una costante della serie: gli shard hanno
dimensioni diverse (quantity skew) e su `ucr_011` vanno da 13 a 29. Usare la mediana
sbaglierebbe il totale del ~30%. Si legge dal log di `local` della stessa serie, che gli
stessi shard li stampa uno per uno — lecito perche' il batch di stage 1 e' 64 in entrambi
(verificato nel `meta` degli out-json).

## Il verdetto di arresto — la colonna che conta piu' di tutte

`patience` = fermato dalla regola di convergenza.  `TETTO` = fermato dal budget, cioe'
**TRUNCATED, NOT CONVERGED**, e quella riga non e' riportabile (regola dura del progetto).

Misurato il 2026-08-06: `local` e `centralized` colpiscono il tetto su 6 e 5 serie su 10,
gli arm federati **mai** (max 119 round su 300). Le due famiglie si fermavano per ragioni
diverse — convergenza da una parte, esaurimento di budget dall'altra — ed e' un handicap a
senso unico sulle baseline, che GONFIA gli arm federati.

    $PY scripts/zn_compute.py              # tutte le celle
    $PY scripts/zn_compute.py --tag zn_es  # un tag solo
"""
import argparse
import glob
import json
import os
import re
import statistics as st
from collections import defaultdict

REPO = "/home/leonardo/PhD/TimeVQVAE-AD-U-Federated"
os.chdir(REPO)
DS = "ucr_split_w2p"
SER = ["ucr_001", "ucr_011", "ucr_014", "ucr_043", "ucr_082",
       "ucr_083", "ucr_086", "ucr_170", "ucr_222", "ucr_229"]

# (tag, arm dell'out-json, etichetta). Stesso arm sotto tag diversi = configurazioni diverse.
ARMS = [("zn_main", "local", "local"), ("zn_main", "centralized", "centr"),
        ("zn_main", "federated", "feder"), ("zn_main", "federated_cb_only", "cb_only"),
        ("zn_main", "federated_cb_only_ema", "cb_ema"),
        ("zn_main", "federated_fedavg_cb_only", "fa_cb"),
        ("zn_enc", "federated_enc_fedavg", "e_avg"),
        ("zn_a1", "federated_enc_fedavg", "A1"),
        ("zn_a2", "federated_enc_fedavg", "A2"),
        ("zn_es", "local", "local_es"), ("zn_es", "centralized", "centr_es"),
        ("zn_ot", "local", "local_ot"), ("zn_ot", "centralized", "centr_ot")]

RX_PROTO = re.compile(r"\[s(\d) (\S+)\] converged protocol:.*?\((\d+) batches/epoch\)")
RX_STEP = re.compile(r"\[s(\d) (\S+)\] ep \d+ step (\d+)")
RX_STOP = re.compile(r"\[s(\d) (\S+)\] early stop @ep\d+ step(\d+)")
RX_DONE = re.compile(r"\[s(\d) (\S+)\] done: \d+ epochs, (\d+) steps")
RX_R1 = re.compile(r"^\[fed:\S+\] round (\d+):", re.M)
RX_R2 = re.compile(r"^\[fed-s2\] round (\d+):", re.M)
RX_LE = re.compile(r"local_epochs=(\d+)")
RX_CAP = re.compile(r"max_steps=(\d+)")


def leggi(tag, arm):
    """{serie: {...}} dai log. None dove il log non c'e'."""
    out = {}
    for s in SER:
        p = f"logs/runs/{tag}/{DS}__{s}__{arm}.log"
        if not os.path.exists(p):
            continue
        t = open(p, errors="ignore").read()
        bpe = {m.group(2): int(m.group(3)) for m in RX_PROTO.finditer(t) if m.group(1) == "1"}
        cap = {int(m) for m in RX_CAP.findall(t)}
        # ultimo step visto per (stage, client), piu' i verdetti espliciti
        last = defaultdict(int)
        for m in list(RX_STEP.finditer(t)) + list(RX_DONE.finditer(t)):
            last[(m.group(1), m.group(2))] = max(last[(m.group(1), m.group(2))], int(m.group(3)))
        stopped = {(m.group(1), m.group(2)) for m in RX_STOP.finditer(t)}
        avviati = {k for k in last if k[0] == "1"}
        r1 = [int(x) for x in RX_R1.findall(t)]
        r2 = [int(x) for x in RX_R2.findall(t)]
        le = int(RX_LE.search(t).group(1)) if RX_LE.search(t) else None
        out[s] = {"bpe": bpe, "s1": {k[1]: v for k, v in last.items() if k[0] == "1"},
                  "s2": {k[1]: v for k, v in last.items() if k[0] == "2"},
                  "conv1": {k[1] for k in stopped if k[0] == "1"},
                  "n1": len(avviati), "rounds1": max(r1) + 1 if r1 else None,
                  "rounds2": max(r2) + 1 if r2 else None, "local_epochs": le,
                  "cap": max(cap) if cap else None}
    return out


ap = argparse.ArgumentParser()
ap.add_argument("--tag")
a = ap.parse_args()
D = {lab: leggi(tag, arm) for tag, arm, lab in ARMS if not a.tag or tag == a.tag}
D = {k: v for k, v in D.items() if v}

# `batches_per_epoch` per client: solo `local` lo stampa (gli arm federati non passano da
# `_converged_loop` nel loop di round). Serve come riferimento per convertire i round in step.
BPE = {s: D.get("local", {}).get(s, {}).get("bpe", {}) for s in SER}


def budget(lab, s):
    """(step/client mediano, step totali, round, verdetto)."""
    d = D.get(lab, {}).get(s)
    if not d:
        return None
    # ⚠️ Una cella VIVA ha per forza client senza `early stop`: non sono troncati, non hanno
    # ancora finito. Senza questo controllo il tool marca `NON RIPORTABILE` proprio le celle
    # lanciate per riparare il troncamento — che e' il modo piu' rapido di non fidarsi piu'
    # dello strumento.
    tag = next(t for t, _, l in ARMS if l == lab)
    arm = next(x for _, x, l in ARMS if l == lab)
    viva = not os.path.exists(f"artifacts/runs/{tag}/{DS}/{s}__{arm}.json")
    if d["s1"]:                                   # percorso _converged_loop
        v = list(d["s1"].values())
        tronc = d["n1"] - len(d["conv1"])
        verdetto = ("IN CORSO" if viva else
                    "patience" if tronc == 0 else f"TETTO {tronc}/{d['n1']}")
        return st.median(v), sum(v), None, verdetto, d["cap"]
    if d["rounds1"] and d["local_epochs"]:        # percorso federato
        bpe = BPE.get(s) or {}
        if not bpe:
            return None
        per = {c: d["rounds1"] * d["local_epochs"] * n for c, n in bpe.items()}
        return (st.median(per.values()), sum(per.values()), d["rounds1"], "patience round", None)
    return None


print(f"{'='*104}\nBUDGET DI STAGE 1 PER CELLA   ·   step/client (mediano) · step TOTALI · round · arresto\n{'='*104}")
LABS = [l for l in D if any(budget(l, s) for s in SER)]
for s in SER:
    print(f"\n{s}")
    base = budget("local", s)
    for l in LABS:
        b = budget(l, s)
        if not b:
            continue
        med, tot, rnd, verd, cap = b
        rap = f"{med/base[0]:5.2f}x" if base and base[0] else "    —"
        flag = "  ⛔ NON RIPORTABILE" if verd.startswith("TETTO") else ""
        print(f"   {l:<10} {med:>7.0f} step/client  ({rap} di local)   {tot:>8.0f} totali   "
              f"{('round '+str(rnd)) if rnd else ('tetto '+str(cap)):<10} {verd:<14}{flag}")

print(f"\n{'='*104}\nRIEPILOGO — chi si e' fermato per convergenza e chi per budget\n{'='*104}")
for l in LABS:
    tr = [s for s in SER if (b := budget(l, s)) and b[3].startswith("TETTO")]
    n = len([s for s in SER if budget(l, s)])
    print(f"   {l:<10} {n-len(tr)}/{n} convergite" + (f"   ⛔ al TETTO: {', '.join(tr)}" if tr else "   ✅"))
