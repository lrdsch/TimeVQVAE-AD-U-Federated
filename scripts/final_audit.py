#!/usr/bin/env python3
"""Audit appaiato finale sulle 9 serie + tabella per il paper.

DECISIONI DI METODO CABLATE QUI, con il perche' — sono tutte state pagate.

1. **L'unita' e' la SERIE, non la cella e non il client.** I 5 client di un cluster hanno
   punteggi correlati (rho 0,66-0,999) e le due finestre vengono dalla stessa serie: contarli
   come osservazioni indipendenti gonfia n di un fattore 10.

2. **Soglie da 3 replicati veri su hardware identico**, non da una stima a due punti: la
   versione a 2 punti sottostimava VUS-PR di 3,6x e aveva resuscitato un risultato che e'
   dovuto essere ritrattato. SD per-client misurate: top-1 0,231 · top-3 0,231 · VUS-PR 0,171
   · AUROC 0,023. La soglia di un confronto appaiato su n serie e' 2 * SD/sqrt(5) * sqrt(2) / sqrt(n).

3. **Si riporta anche la CONCORDANZA DI SEGNO**, non solo la media. Un Δ medio che nasce da
   serie con segni opposti e' rumore travestito: e' cosi' che ho quasi pubblicato un effetto
   inesistente leggendo due medie quasi uguali su due serie.

4. **AUROC non va usata contro il floor.** Con anomalie allo 0,1-2% e' dominata dai negativi
   facili: il floor arriva a 0,007 dal deep su AUROC mentre perde di 0,861 su VUS-PR.

5. **VUS-PR non e' aggregabile FRA serie UCR** (AUPRC 0,994 -> VUS-PR 0,211 sugli stessi
   punteggi). Qui si usa solo appaiata dentro la serie, mai come media cross-serie assoluta.

6. Le serie **cieche** (`ucr_229` pavimento, `ucr_083` soffitto, `ucr_086` picchi concorrenti)
   si riportano ma si segnalano: contribuiscono 0 a ogni Δ e abbassano n effettivo.
"""
from __future__ import annotations

import glob
import json
import math
import os
from collections import defaultdict
from statistics import fmean

REPO = "/home/leonardo/PhD/TimeVQVAE-AD-U-Federated"
SER = ["001", "011", "014", "043", "083", "086", "170", "222", "229"]
BUILDS = {"ucr_split": "W=128", "ucr_split_w2p": "W=2P"}
PAPER = ["local", "centralized", "federated_cb_only", "federated_cb_only_ema",
         "federated_fedavg_cb_only", "federated", "federated_shared",
         "federated_fedavg_cb_sharedprior"]
# SD per-client misurate su 3 run dello stesso comando, stessa scheda (ucr_001, W=408).
SD_PC = {"paper_top1_acc_at_64": 0.231, "paper_top3_acc_at_64": 0.231,
         "vus_pr": 0.171, "auroc": 0.023}
CIECHE = {"229": "pavimento (tutti 0,00)", "083": "soffitto (local gia' 1,00)",
          # CORRETTO 2026-08-03: era «picchi concorrenti (argmax instabile)», cioe' il sintomo
          # scambiato per la causa. Dimostrato con scripts/repro_probe.py: il punteggio sul test
          # e' funzione DETERMINISTICA dello shard di train (ckpt di p0 + train di p3 ->
          # risultato di p3, argmax 48067, rho 0,999993). Non e' instabilita': e' un
          # accoppiamento train->test nella pipeline di scoring.
          "086": "punteggi dipendenti dallo shard di train (difetto di scoring, dimostrato)"}


def cella(tag: str, ds: str, cl: str, arm: str, m: str) -> float | None:
    """Media sui 5 client. None se la cella non e' completa: mai mediare su meno di 5."""
    ps = glob.glob(f"{REPO}/artifacts/runs/{tag}/ckpt/{ds}/{cl}/seed0/{arm}/*/report.json")
    if len(ps) < 5:
        return None
    vs = [json.load(open(p)).get(m) for p in ps]
    return fmean(vs) if all(v is not None for v in vs) else None


def soglia(m: str, n: int) -> float:
    """2 SD della differenza appaiata, mediata su n serie."""
    return 2 * (SD_PC[m] / math.sqrt(5)) * math.sqrt(2) / math.sqrt(n)


def per_serie(arm: str, m: str, tag_of=lambda s: f"ucr{s}_v1") -> dict[str, float]:
    """Un numero per serie = media delle due finestre. Le finestre non sono osservazioni
    indipendenti: vengono dalla stessa serie."""
    out = {}
    for s in SER:
        v = [cella(tag_of(s), ds, f"ucr_{s}", arm, m) for ds in BUILDS]
        if all(x is not None for x in v):
            out[s] = fmean(v)
    return out


def concordanza(d: list[float]) -> tuple[int, int, int]:
    """(a favore, contro, pareggi) rispetto al segno della media.

    ⚠️ CORRETTO 2026-08-03. La versione precedente era `sum(1 for x in d if (x>0)==(md>0))`,
    che con media NEGATIVA collassa a `x <= 0` e quindi **conta i pareggi come concordi**,
    mentre con media positiva li esclude. Ogni effetto negativo appariva piu' forte del vero e
    ogni positivo piu' debole: `centralized` +0,189 era dato 5/9 quando i valori sono 5 positivi,
    ZERO negativi e 4 pareggi (5 su 5 fra le informative), e `fedavg_cb_sharedprior` -0,189 era
    dato 8/9 essendo 6 negativi, 1 positivo e 2 pareggi (6 su 7). I pareggi non sono evidenza in
    nessuna direzione: vanno mostrati, non assorbiti dalla parte che conviene."""
    md = fmean(d)
    pro = sum(1 for x in d if x != 0 and (x > 0) == (md > 0))
    con = sum(1 for x in d if x != 0 and (x > 0) != (md > 0))
    return pro, con, sum(1 for x in d if x == 0)


def confronto(a: str, b: str, m: str, tag_of=lambda s: f"ucr{s}_v1") -> dict | None:
    A, B = per_serie(a, m, tag_of), per_serie(b, m, tag_of)
    ks = sorted(set(A) & set(B))
    if len(ks) < 3:
        return None
    d = [A[k] - B[k] for k in ks]
    md = fmean(d)
    pro, con, tie = concordanza(d)
    return {"delta": md, "n": len(ks), "serie": ks, "per_serie": d,
            "soglia": soglia(m, len(ks)),
            "concordi": f"{pro}/{pro+con}" + (f" (+{tie} pari)" if tie else ""),
            "sopra": abs(md) > soglia(m, len(ks))}


def main() -> None:
    righe = ["# Tabella finale — 9 serie UCR, TimeVQVAE-AD federato", ""]
    righe.append("Unita' = **serie** (media delle 2 finestre, media dei 5 client). "
                 "Soglie = 2 SD della differenza appaiata, da 3 replicati su hardware identico.")
    righe.append("")
    ok = [s for s in SER if s not in CIECHE]
    righe.append(f"⚠️ **{len(CIECHE)} serie su 9 non discriminano** e contribuiscono 0 a ogni Δ: "
                 + ", ".join(f"`ucr_{k}` ({v})" for k, v in CIECHE.items())
                 + f". Le colonne «senza cieche» usano le {len(ok)} restanti.")
    righe.append("")

    for m, lab in (("paper_top1_acc_at_64", "top-1"), ("vus_pr", "VUS-PR")):
        righe += [f"## {lab} — ogni arm contro `local`", "",
                  "| arm | Δ vs local | n | soglia | concordi | esito | per serie |",
                  "|---|---:|---:|---:|---:|---|---|"]
        for arm in PAPER:
            if arm == "local":
                continue
            r = confronto(arm, "local", m)
            if r is None:
                righe.append(f"| `{arm}` | — | | | | dati incompleti | |")
                continue
            righe.append(
                f"| `{arm}` | **{r['delta']:+.3f}** | {r['n']} | ±{r['soglia']:.3f} | "
                f"{r['concordi']} | {'🟢 sopra il rumore' if r['sopra'] else 'dentro il rumore'} | "
                + " ".join(f"{x:+.2f}" for x in r["per_serie"]) + " |")
        righe.append("")

    # K=128 contro K=64, sugli arm che hanno entrambi
    righe += ["## Codebook: K=128 contro K=64", "",
              "| arm | metrica | Δ | n | soglia | concordi | esito |", "|---|---|---:|---:|---:|---:|---|"]
    for arm in ("federated_cb_only", "federated_fedavg_cb_only", "local", "centralized"):
        tag = (lambda s: f"ucr{s}_cb128base") if arm in ("local", "centralized") \
              else (lambda s: f"ucr{s}_cb128")
        for m, lab in (("paper_top1_acc_at_64", "top-1"), ("vus_pr", "VUS-PR")):
            A, B = per_serie(arm, m, tag), per_serie(arm, m)
            ks = sorted(set(A) & set(B))
            if len(ks) < 3:
                righe.append(f"| `{arm}` | {lab} | — | {len(ks)} | | | dati incompleti |")
                continue
            d = [A[k] - B[k] for k in ks]; md = fmean(d); th = soglia(m, len(ks))
            righe.append(f"| `{arm}` | {lab} | {md:+.3f} | {len(ks)} | ±{th:.3f} | "
                         f"{concordanza(d)[0]}/{concordanza(d)[0]+concordanza(d)[1]} | "
                         f"{'🟢 sopra' if abs(md)>th else 'dentro il rumore'} |")
    righe.append("")

    # Trio encoder + commoninit. ⚠ I nomi delle cartelle portano gli iperparametri nel nome
    # (`_lam1_uniform`, `_mu0.01`): scriverli a mano come gli arm `paper` darebbe 0 celle
    # trovate e una tabella «dati incompleti» che sembra un problema di run, non di glob.
    ENC = ["federated_enc_fedavg", "federated_enc_fedprox_mu0.01",
           "federated_enc_fedproto_lam1_uniform", "federated_enc_commoninit"]
    righe += ["## Federazione dell'encoder — ogni arm contro `local`", "",
              "Il floor `movavg10` batte gia' questo blocco (mediana 0,507 a zero parametri): "
              "il confronto che conta e' contro `local`, non fra gli arm.", "",
              "| arm | metrica | Δ vs local | n | soglia | concordi | esito |",
              "|---|---|---:|---:|---:|---:|---|"]
    for arm in ENC:
        for m, lab in (("paper_top1_acc_at_64", "top-1"), ("vus_pr", "VUS-PR")):
            A = per_serie(arm, m, lambda s: f"ucr{s}_enc")
            B = per_serie("local", m)
            ks = sorted(set(A) & set(B))
            if len(ks) < 3:
                righe.append(f"| `{arm}` | {lab} | — | {len(ks)} | | | dati incompleti |")
                continue
            d = [A[k] - B[k] for k in ks]; md = fmean(d); th = soglia(m, len(ks))
            righe.append(f"| `{arm}` | {lab} | {md:+.3f} | {len(ks)} | ±{th:.3f} | "
                         f"{concordanza(d)[0]}/{concordanza(d)[0]+concordanza(d)[1]} | "
                         f"{'🟢 sopra' if abs(md)>th else 'dentro il rumore'} |")
    righe.append("")

    # FedProto: count contro uniform. Il perche' e' nel ledger: `uniform` e' cio' che fa il
    # codice ufficiale, `count` e' la lettura letterale dell'Eq. 6 e degenera nel codebook
    # merged (3.3e-8). `uniform` non era mai stato misurato a convergenza prima di questa corsa.
    righe += ["## FedProto: aggregazione `count` contro `uniform`", "",
              "| metrica | Δ (count − uniform) | n | soglia | concordi | esito |",
              "|---|---:|---:|---:|---:|---|"]
    for m, lab in (("paper_top1_acc_at_64", "top-1"), ("vus_pr", "VUS-PR")):
        A = per_serie("federated_enc_fedproto_lam1_count", m, lambda s: f"ucr{s}_proto_count")
        B = per_serie("federated_enc_fedproto_lam1_uniform", m, lambda s: f"ucr{s}_enc")
        ks = sorted(set(A) & set(B))
        if len(ks) < 3:
            righe.append(f"| {lab} | — | {len(ks)} | | | dati incompleti |")
            continue
        d = [A[k] - B[k] for k in ks]; md = fmean(d); th = soglia(m, len(ks))
        righe.append(f"| {lab} | {md:+.3f} | {len(ks)} | ±{th:.3f} | "
                     f"{concordanza(d)[0]}/{concordanza(d)[0]+concordanza(d)[1]} | "
                     f"{'🟢 sopra' if abs(md)>th else 'dentro il rumore'} |")
    righe.append("")

    # Finestra
    righe += ["## Finestra: W=2P contro W=128", "",
              "| arm | metrica | Δ | n | soglia | concordi | esito |", "|---|---|---:|---:|---:|---:|---|"]
    for arm in PAPER:
        for m, lab in (("paper_top1_acc_at_64", "top-1"), ("vus_pr", "VUS-PR")):
            d = []
            for s in SER:
                x = cella(f"ucr{s}_v1", "ucr_split", f"ucr_{s}", arm, m)
                y = cella(f"ucr{s}_v1", "ucr_split_w2p", f"ucr_{s}", arm, m)
                if x is not None and y is not None:
                    d.append(y - x)
            if len(d) < 3:
                continue
            md = fmean(d); th = soglia(m, len(d))
            righe.append(f"| `{arm}` | {lab} | {md:+.3f} | {len(d)} | ±{th:.3f} | "
                         f"{concordanza(d)[0]}/{concordanza(d)[0]+concordanza(d)[1]} | "
                         f"{'🟢 sopra' if abs(md)>th else 'dentro il rumore'} |")
    righe.append("")

    # Confronto esterno
    righe += ["## Contro il numero pubblicato (TimeVQVAE-AD: paper_top1 = 0,708)", ""]
    C = per_serie("centralized", "paper_top1_acc_at_64")
    if C:
        vals = list(C.values())
        m_ = fmean(vals)
        se = (sum((v - m_) ** 2 for v in vals) / (len(vals) - 1)) ** .5 / math.sqrt(len(vals)) if len(vals) > 1 else 0
        righe.append(f"- **tutte le serie** (n={len(vals)}): {m_:.3f}, IC95% [{m_-1.96*se:.3f}, {m_+1.96*se:.3f}]")
        v2 = [C[s] for s in C if s not in CIECHE]
        if len(v2) > 1:
            m2 = fmean(v2)
            se2 = (sum((v - m2) ** 2 for v in v2) / (len(v2) - 1)) ** .5 / math.sqrt(len(v2))
            righe.append(f"- **senza le cieche** (n={len(v2)}): {m2:.3f}, IC95% [{m2-1.96*se2:.3f}, {m2+1.96*se2:.3f}]")
        righe.append("")
        righe.append("`centralized` e' l'analogo strutturale del loro setting: il test e' condiviso "
                     "e identico fra i 5 client, quindi addestra sull'intero train della serie e "
                     "valuta sull'intero test. Periodi verificati identici ai loro su tutte e 250 "
                     "le serie, e ±64 contro ±100 non cambia nulla (0 celle su 960).")

    out = f"{REPO}/documentation/FINAL_TABLE.md"
    open(out, "w").write("\n".join(righe) + "\n")
    print("\n".join(righe))
    print(f"\n-> scritto {out}")


if __name__ == "__main__":
    main()
