"""Matrice completa della campagna z-norm: ogni cella fatta, in corso o mancante.

Una riga per serie, una colonna per cella (12 = 6 zn_main + 4 zn_enc + 1 zn_norev + 1 zn_a1).

⚠ La cella si identifica con (TAG, serie, arm), MAI con (serie, arm): `zn_a1` gira lo stesso
arm di `zn_enc` (`federated_enc_fedavg`) cambiando solo `--fed-enc-bn`. Con la chiave a due
campi la cella `zn_enc` chiusa farebbe risultare chiusa anche quella `zn_a1`.

Stato:
  V  fatta      esiste l'out-json
  >  in corso   c'e' un PROCESSO VIVO (non "il log e' recente": gli arm federati possono
                tacere mezz'ora fra un round e l'altro, e i log dei job uccisi restano)
  .  mancante   ne' l'uno ne' l'altro
"""
import glob
import os
import re
import subprocess
import sys

REPO = "/home/leonardo/PhD/TimeVQVAE-AD-U-Federated"
os.chdir(REPO)

SER = ["ucr_001", "ucr_011", "ucr_014", "ucr_043", "ucr_082",
       "ucr_083", "ucr_086", "ucr_170", "ucr_222", "ucr_229"]
# (tag, arm, etichetta di colonna)
COLS = [
    ("zn_main", "centralized", "centr"),
    ("zn_main", "local", "local"),
    ("zn_main", "federated", "feder"),
    ("zn_main", "federated_cb_only", "cb_on"),
    ("zn_main", "federated_cb_only_ema", "cb_ema"),
    ("zn_main", "federated_fedavg_cb_only", "fa_cb"),
    ("zn_enc", "federated_enc_fedavg", "e_avg"),
    ("zn_enc", "federated_enc_fedprox", "e_prx"),
    ("zn_enc", "federated_enc_fedproto", "e_pro"),
    ("zn_enc", "federated_enc_commoninit", "e_ini"),
    ("zn_norev", "federated_cb_only_ema_norevive", "NOREV"),
    ("zn_a1", "federated_enc_fedavg", "A1-bn"),
]

# Ramo N: solo le 6 serie dove la rianimazione puo' scattare (+ ucr_001 come controllo
# di determinismo). Sulle altre 4 `revive=False` da' una run BIT-IDENTICA a cb_only_ema:
# non e' un risultato nullo misurato, e' un pareggio per costruzione, e non e' "mancante".
NOREV_SER = {"ucr_001", "ucr_011", "ucr_014", "ucr_043", "ucr_222", "ucr_229"}

done = {(p.split("/")[2], *os.path.basename(p)[:-5].split("__"))
        for p in glob.glob("artifacts/runs/zn_*/*/*__*.json")}

live = set()
for cmd in ("ps -eo args",
            "ssh -o BatchMode=yes -o ConnectTimeout=10 leonardo@g4.etsisi.upm.es 'ps -eo args'"):
    try:
        out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout
    except Exception:
        continue
    # il TAG si legge da --out-json: `--arms federated_enc_fedavg` da solo non dice
    # se la cella e' di zn_enc o di zn_a1.
    for m in re.finditer(r"--cluster (\S+) --arms (\S+).*?/runs/([A-Za-z0-9_]+)/", out):
        live.add((m.group(3), m.group(1), m.group(2)))

# scheda su cui e' girata ogni cella chiusa, per la colonna di destra
card = {}
for p in glob.glob("artifacts/runs/*/placement.tsv"):
    tag = p.split("/")[2]
    for line in open(p, errors="ignore"):
        f = line.rstrip("\n").split("\t")
        if len(f) >= 7:
            card[(tag, f[5], f[6])] = f[3]
CHAIN_CARD = {"ucr_001": "Quadro", "ucr_011": "Quadro"}


def state(tag, ser, arm):
    if tag == "zn_norev" and ser not in NOREV_SER:
        return "-"                       # fuori disegno, non mancante
    k = (tag, ser, arm)
    return "V" if k in done else (">" if k in live else ".")


W = 7
head = " " * 10 + "".join(f"{c:<{W}}" for _, _, c in COLS)
sep = " " * 10 + "".join(f"{'-'*(W-1):<{W}}" for _ in COLS)
print("\nMATRICE CAMPAGNA z-norm — coorte ucr2p_10, seed 0, 120 celle")
print("  V = fatta   > = in corso   . = mancante   - = fuori disegno\n")
print(head)
print(sep)

tot = {"V": 0, ">": 0, ".": 0, "-": 0}
for s in SER:
    cells = [state(t, s, a) for t, a, _ in COLS]
    for c in cells:
        tot[c] += 1
    n_done = cells.count("V")
    n_tot = 12 - cells.count("-")
    print(f"{s:<10}" + "".join(f"{c:<{W}}" for c in cells) + f"  {n_done:>2}/{n_tot}")

print(sep)
percol = []
for i, (t, a, lab) in enumerate(COLS):
    n = sum(1 for s in SER if state(t, s, a) == "V")
    tot_col = sum(1 for s in SER if state(t, s, a) != "-")
    percol.append(f"{n}/{tot_col}")
print(" " * 10 + "".join(f"{p:<{W}}" for p in percol))

print(f"\n  fatte {tot['V']}   ·   in corso {tot['>']}   ·   mancanti {tot['.']}"
      f"   ·   fuori disegno {tot[chr(45)]}   ·   totale {sum(tot.values())}")

# ── dettaglio: cosa sta girando adesso, e dove ──────────────────────────────────
if live:
    print(f"\nIN CORSO ({len(live)}):")
    print(f"  {'tag':<9}{'serie':<10}{'arm':<32}")
    for tag, cl, arm in sorted(live):
        print(f"  {tag:<9}{cl:<10}{arm:<32}")

# ── le mancanti, raggruppate per chi le deve fare ───────────────────────────────
miss = [(t, s, a) for s in SER for t, a, _ in COLS if state(t, s, a) == "."]
if miss:
    print(f"\nMANCANTI ({len(miss)}), per serie:")
    for s in SER:
        m = [f"{t.replace('zn_','')}/{a.replace('federated_','')}" for t, ss, a in miss if ss == s]
        if m:
            print(f"  {s:<10} {len(m):>2}  " + ", ".join(m))
