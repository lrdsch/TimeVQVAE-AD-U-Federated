"""Interroga e VERIFICA la partizione dichiarata in `cohorts/zn_ownership.json`.

Due usi:

  # dentro una catena, al posto di una lista scritta a mano:
  export LAUNCH_ONLY_CLUSTERS=$($PY scripts/zn_owner.py --owner "zn_a1_rush.sh" --tag zn_a1)

  # prima di lanciare qualunque cosa:
  $PY scripts/zn_owner.py --check

`--check` risponde alla sola domanda che conta: la mappa e' una PARTIZIONE? Cioe' ogni cella
del disegno ha un proprietario, e ne ha esattamente uno. Se due proprietari rivendicano la
stessa cella il conflitto e' gia' nel file, prima ancora che qualcuno lo lanci -- ed e'
esattamente cosi' che il 2026-08-05 `zn_branches.sh g4` e `zn_g4_after_chain.sh` sono finiti
a coprire le stesse dieci celle.

Esce con codice 1 se la partizione e' rotta, cosi' si puo' mettere in testa a una catena.
"""
import argparse
import json
import os
import sys

REPO = "/home/leonardo/PhD/TimeVQVAE-AD-U-Federated"
os.chdir(REPO)
M = json.load(open("cohorts/zn_ownership.json"))
SER = M["series"]


def design_cells():
    out = []
    for tag, d in M["design"].items():
        sers = SER if d["series"] == "ALL" else d["series"]
        for s in sers:
            for a in d["arms"]:
                out.append((tag, s, a))
    return out


def owned():
    """{(tag, serie, arm): [proprietari]}"""
    o = {}
    for w in M["owners"]:
        arms = M["design"][w["tag"]]["arms"]
        for s in w["series"]:
            for a in arms:
                o.setdefault((w["tag"], s, a), []).append(f'{w["name"]}@{w["host"]}')
    return o


ap = argparse.ArgumentParser()
ap.add_argument("--owner")
ap.add_argument("--tag")
ap.add_argument("--check", action="store_true")
args = ap.parse_args()

if args.owner and args.tag:
    ss = [s for w in M["owners"] if w["name"] == args.owner and w["tag"] == args.tag
          for s in w["series"]]
    if not ss:
        print(f"nessuna serie per {args.owner} / {args.tag}", file=sys.stderr)
        sys.exit(2)
    print(",".join(dict.fromkeys(ss)))       # dedup mantenendo l'ordine
    sys.exit(0)

if args.check:
    des, own = design_cells(), owned()
    doppie = {k: v for k, v in own.items() if len(v) > 1}
    scoperte = [k for k in des if k not in own]
    fuori = [k for k in own if k not in des]
    print(f"disegno: {len(des)} celle   ·   con proprietario: {len([k for k in des if k in own])}")
    if doppie:
        print(f"\n⛔ {len(doppie)} celle con DUE proprietari — la partizione e' rotta:")
        for (t, s, a), v in sorted(doppie.items()):
            print(f"   {t:<9}{s:<10}{a:<32}{', '.join(v)}")
    if scoperte:
        print(f"\n⛔ {len(scoperte)} celle SENZA proprietario — nessuno le fara':")
        for t, s, a in sorted(scoperte):
            print(f"   {t:<9}{s:<10}{a}")
    if fuori:
        print(f"\n⚠ {len(fuori)} celle rivendicate ma FUORI dal disegno (refuso nel manifesto?):")
        for t, s, a in sorted(fuori)[:10]:
            print(f"   {t:<9}{s:<10}{a}")
    if not (doppie or scoperte or fuori):
        print("\n✅ PARTIZIONE VALIDA: ogni cella ha esattamente un proprietario.")
        sys.exit(0)
    sys.exit(1)

ap.print_help()
