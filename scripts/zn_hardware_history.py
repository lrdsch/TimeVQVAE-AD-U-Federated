"""Storia hardware: per OGNI cella, su che scheda e' girata e da quando a quando.

PERCHE'. I contrasti dello studio sono DENTRO la serie (`cb_only - fedavg_cb`,
`enc_* - cb_only`, ...). Se le celle di una stessa serie girassero su architetture diverse,
il confondente hardware entrerebbe nel contrasto invece di restare fra serie, dove i test
appaiati non lo vedono. La precauzione si prende nello scheduling, ma senza questo file non
e' VERIFICABILE a posteriori -- e una precauzione non verificabile non e' una precauzione.

FONTI, in ordine di attendibilita':
  1. `artifacts/runs/<tag>/placement.tsv` -- REGISTRATO da launch.sh al dispatch (host,
     indice GPU, modello e compute capability letti da nvidia-smi). Esiste solo per le celle
     dispatchate dal 2026-08-05 in poi.
  2. `logs/runs/<tag>/_orchestrator.log` -- START/DONE con timestamp e indice GPU. Il file e'
     CONDIVISO fra g2 e g4 sul filesystem montato: dal 2026-08-05 le righe portano l'host nel
     prefisso, prima no.
  3. La mappa serie->catena, INFERITA dal fatto che le catene sono state lanciate con
     `LAUNCH_ONLY_CLUSTERS` disgiunti. E' un'inferenza, non una registrazione: marcata come
     tale nel campo `fonte`.

ADDESTRAMENTO E INFERENZA sono lo STESSO processo: `federated_eval.py` allena e poi chiama
detect nello stesso interprete, senza cambiare device. Quindi per ogni cella l'hardware di
training e quello di inferenza coincidono, e l'intervallo qui sotto li copre entrambi.
"""
import csv
import datetime
import glob
import json
import os
import re
import sys

REPO = "/home/leonardo/PhD/TimeVQVAE-AD-U-Federated"
os.chdir(REPO)

# Mappa inferita per le celle anteriori a placement.tsv (catene con serie disgiunte).
CHAIN = {
    "ucr_001": ("g2", "Quadro RTX 8000", "7.5"), "ucr_011": ("g2", "Quadro RTX 8000", "7.5"),
    "ucr_014": ("g4", "GeForce RTX 3090", "8.6"), "ucr_043": ("g4", "GeForce RTX 3090", "8.6"),
    "ucr_082": ("g4", "GeForce RTX 3090", "8.6"), "ucr_083": ("g4", "GeForce RTX 3090", "8.6"),
    "ucr_086": ("g4", "GeForce RTX 3090", "8.6"),
    "ucr_170": ("g4", "RTX 4500 Ada Generation", "8.9"),
    "ucr_222": ("g4", "RTX 4500 Ada Generation", "8.9"),
    "ucr_229": ("g4", "RTX 4500 Ada Generation", "8.9"),
}
# Su g4 l'indice GPU disambigua l'architettura: 1/2/3 sono 3090, 0/4/5 sono Ada.
G4_BY_GPU = {"0": ("RTX 4500 Ada Generation", "8.9"), "4": ("RTX 4500 Ada Generation", "8.9"),
             "5": ("RTX 4500 Ada Generation", "8.9"), "1": ("GeForce RTX 3090", "8.6"),
             "2": ("GeForce RTX 3090", "8.6"), "3": ("GeForce RTX 3090", "8.6")}

# ── 1. placement.tsv: la fonte registrata ────────────────────────────────────────
placed = {}
for p in glob.glob("artifacts/runs/*/placement.tsv"):
    tag = p.split("/")[2]
    for row in csv.reader(open(p), delimiter="\t"):
        if len(row) < 7:
            continue
        ts, host, gpu, card, ds, cl, arm = row[:7]
        placed[(tag, cl, arm)] = dict(host=host, gpu=gpu, card=card.replace("/", " cc "),
                                      dispatch=ts, fonte="registrato")

# ── 2. orchestratore: START/DONE, indice GPU, host (solo dal 2026-08-05) ─────────
runs = {}
for lg in glob.glob("logs/runs/*/_orchestrator.log"):
    tag = lg.split("/")[2]
    for line in open(lg, errors="ignore"):
        m = re.match(r"\[(.*?)\](?:\[(\w+)\])? (START|DONE)\s+(\S+)", line)
        if not m:
            continue
        t, host, kind, job = m.group(1), m.group(2), m.group(3), m.group(4)
        if job.count("__") != 2:
            continue
        ds, cl, arm = job.split("__")
        k = (tag, cl, arm)
        r = runs.setdefault(k, {})
        if kind == "START":
            r["t_start"] = t
            g = re.search(r"GPU(\d+)", line)
            if g:
                r["gpu"] = g.group(1)
            if host:
                r["host"] = host
        else:
            r["t_end"] = t
            r["rc0"] = "rc=0" in line

# ── 3. unione, con la fonte dichiarata ───────────────────────────────────────────
rows = []
for tag in ("zn_main", "zn_enc", "zn_norev", "zn_a1"):
    for out in sorted(glob.glob(f"artifacts/runs/{tag}/*/*__*.json")):
        cl, arm = os.path.basename(out)[:-5].split("__")
        k = (tag, cl, arm)
        r = dict(runs.get(k, {}))
        p = placed.get(k)
        if p:
            host, card, gpu, fonte = p["host"], p["card"], p["gpu"], "registrato"
        else:
            host, card, cc = CHAIN.get(cl, ("?", "?", "?"))
            gpu = r.get("gpu", "?")
            if host == "g4" and gpu in G4_BY_GPU:          # l'indice conferma l'architettura
                card, cc = G4_BY_GPU[gpu]
                fonte = "inferito+gpu"
            else:
                fonte = "inferito"
            card = f"{card} cc {cc}"
        # amp effettivamente usata, dalla cella
        kn = glob.glob(f"artifacts/runs/{tag}/ckpt/*/{cl}/seed0/{arm}/knobs*.json")
        amp = "?"
        if kn:
            try:
                amp = json.load(open(kn[0])).get("amp_dtype", "?")
            except Exception:
                pass
        dur = ""
        if r.get("t_start") and r.get("t_end"):
            a = datetime.datetime.fromisoformat(r["t_start"])
            b = datetime.datetime.fromisoformat(r["t_end"])
            dur = f"{(b - a).total_seconds() / 3600:.2f}"
        rows.append(dict(tag=tag, serie=cl, arm=arm, host=host, gpu=gpu, scheda=card,
                         amp=amp, inizio_utc=r.get("t_start", ""), fine_utc=r.get("t_end", ""),
                         ore=dur, fonte=fonte))

os.makedirs("evidence", exist_ok=True)
OUT = "evidence/hardware_history_ucr2p_10.csv"
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["tag", "serie", "arm", "host", "gpu", "scheda", "amp",
                                      "inizio_utc", "fine_utc", "ore", "fonte"])
    w.writeheader()
    w.writerows(rows)

# ── 4. il controllo che il file esiste per fare: una serie su due architetture? ──
per_serie = {}
for r in rows:
    per_serie.setdefault(r["serie"], set()).add(r["scheda"])
mix = {s: c for s, c in per_serie.items() if len(c) > 1}
amps = {r["amp"] for r in rows}

print(f"{len(rows)} celle -> {OUT}\n")
print(f"{'serie':<10}{'celle':>6}  scheda/e")
for s in sorted(per_serie):
    n = sum(1 for r in rows if r["serie"] == s)
    flag = "  ⚠ DUE ARCHITETTURE" if s in mix else ""
    print(f"{s:<10}{n:>6}  {', '.join(sorted(per_serie[s]))}{flag}")
print(f"\nprecisione usata: {amps or '?'}")
if mix:
    print(f"\n⚠ {len(mix)} serie hanno celle su architetture diverse: i contrasti DENTRO"
          f" queste serie portano il confondente hardware. Vanno annotate nel paper.")
    for s, c in mix.items():
        print(f"   {s}: {sorted(c)}")
else:
    print("\n✅ nessuna serie mescola architetture: ogni contrasto within-series e' omogeneo.")
by = {}
for r in rows:
    by[r["fonte"]] = by.get(r["fonte"], 0) + 1
print(f"\nfonte del dato: " + ", ".join(f"{k} {v}" for k, v in sorted(by.items())))
