"""Metriche della campagna z-norm: sei metriche, per serie e per arm, piu' i contrasti appaiati.

DECISIONI DI METODO, tutte gia' pagate una volta (vedi scripts/final_audit.py per la storia):

1. **L'unita' e' la SERIE, non il client.** I 5 client di un cluster hanno punteggi correlati
   (rho 0,66-0,999): contarli come osservazioni indipendenti gonfia n di 5. Qui la cella e' la
   media sui 5 client, e la cella e' l'osservazione.

2. **Mai mediare su meno di 5 client.** Una cella incompleta e' None, non una media parziale.

3. **VUS-PR e AUPRC non sono aggregabili FRA serie** (sugli stessi punteggi AUPRC 0,994 ->
   VUS-PR 0,211). Le medie cross-serie di queste due si stampano in grigio, come indicazione,
   e i confronti si fanno SOLO appaiati dentro la serie.

4. **AUROC non va usata per giudicare la distanza dal floor**: con anomalie allo 0,1-2% e'
   dominata dai negativi facili (il floor arriva a 0,007 dal deep su AUROC mentre perde 0,861
   su VUS-PR). Si riporta perche' e' standard, non perche' discrimini.

5. **Soglia di rumore da 3 replicati veri** sullo stesso comando e la stessa scheda, non da
   una stima a due punti (che sottostimava VUS-PR di 3,6x e aveva resuscitato un risultato poi
   ritrattato). SD per-client: top-1/top-3 0,231 · VUS-PR 0,171 · AUROC 0,023.

6. **Concordanza di segno, con i pareggi ESPLICITI.** Un Delta medio che nasce da serie di
   segno opposto e' rumore travestito. E i pareggi non sono evidenza in nessuna direzione:
   contarli con la maggioranza (bug corretto il 2026-08-03) faceva sembrare ogni effetto
   negativo piu' forte del vero.

7. **La saturazione si MISURA, non si assume.** Una serie dove tutti gli arm fanno 1,00 (o
   tutti 0,00) contribuisce 0 a ogni Delta appaiato: va segnalata perche' abbassa l'n
   effettivo, e va identificata dai dati di oggi, non da una lista scritta a mano.
"""
import glob
import json
import math
import os
import re
from collections import defaultdict
import sys
from statistics import fmean, median

# ── AGGREGATORE ─────────────────────────────────────────────────────────────────
# `--agg median` (default) o `--agg mean`. Si applica a TRE livelli: dentro la cella (sui 5
# client), fra serie, e al Delta appaiato.
#
# ⚠ SUI TOP-K LA MEDIANA FRA CLIENT E' UN VOTO DI MAGGIORANZA. `paper_topK_acc` per client
# vale 0 o 1: la media sui 5 e' la FRAZIONE di client che azzecca (sei livelli), la mediana
# e' binaria. Perde risoluzione proprio dove ne serve.
# ⚠ SULLE METRICHE CONTINUE la mediana e' invece meglio: su ucr_011/cb_only i client danno
# AUPRC [0,21 0,27 0,99 0,19 0,21] -- la media 0,374 la fa un client solo, la mediana 0,210.
# ⚠ FRA SERIE, con >50% di pareggi la mediana del Delta e' 0,000 PER COSTRUZIONE. Non e' un
# risultato nullo: e' la saturazione che vince il voto. Per questo si stampano entrambe.
#
# ⚠ LA SOGLIA CAMBIA. Quella misurata sui replicati vale per la MEDIA. La varianza asintotica
# della mediana campionaria e' pi/2 volte quella della media, e qui la mediana e' presa DUE
# volte (client e serie): la soglia va moltiplicata per (pi/2). Usare la soglia della media
# su una statistica mediana la renderebbe troppo permissiva.
AGG_NAME = "median"
for _i, _a in enumerate(sys.argv):
    if _a == "--agg" and _i + 1 < len(sys.argv):
        AGG_NAME = sys.argv[_i + 1]
AGG = median if AGG_NAME == "median" else fmean
THR_K = (math.pi / 2) if AGG_NAME == "median" else 1.0

REPO = "/home/leonardo/PhD/TimeVQVAE-AD-U-Federated"
os.chdir(REPO)
DS = "ucr_split_w2p"
SER = ["ucr_001", "ucr_011", "ucr_014", "ucr_043", "ucr_082",
       "ucr_083", "ucr_086", "ucr_170", "ucr_222", "ucr_229"]

MET = [("paper_top1_acc_at_64", "top-1"), ("paper_top3_acc_at_64", "top-3"),
       ("paper_top5_acc_at_64", "top-5"), ("auroc", "AUROC"),
       ("auprc", "AUPRC"), ("vus_pr", "VUS-PR")]
NON_AGG = {"auprc", "vus_pr"}          # non aggregabili fra serie: solo appaiate
SD_PC = {"paper_top1_acc_at_64": 0.231, "paper_top3_acc_at_64": 0.231,
         "paper_top5_acc_at_64": 0.231, "vus_pr": 0.171, "auroc": 0.023,
         "auprc": 0.171}

# (tag, arm dell'OUT-JSON, regex della cartella CHECKPOINT, etichetta).
#
# ⚠ DUE NOMI PER LO STESSO ARM, e vanno tenuti separati o la tabella mente in due modi opposti.
# L'out-json usa il nome NUDO (`ucr_011__federated_enc_fedavg.json`); la cartella dei
# checkpoint ci attacca i knob (`federated_enc_fedavg_bn-shared`, `..._prior-partial`,
# `federated_enc_fedprox_mu0.01`, `federated_enc_fedproto_lam1_uniform`) -- e' cosi' che due
# configurazioni dello stesso arm non collidono su disco.
#   · usando il nome nudo per il checkpoint si trovano 0 report.json e la colonna sparisce
#     IN SILENZIO: cosi' A1, A2, fedprox e fedproto sono rimasti invisibili qui fino al 06-08;
#   · usando un glob `nome*` si prende anche il variante con i knob e la colonna dell'arm base
#     si CONTAMINA con celle di un altro esperimento.
# Percio' la cartella e' una regex ANCORATA, mai un prefisso.
ARMS = [("zn_main", "centralized", r"centralized", "centralized"),
        ("zn_main", "local", r"local", "local"),
        ("zn_main", "federated", r"federated", "federated"),
        ("zn_main", "federated_cb_only", r"federated_cb_only", "cb_only"),
        ("zn_main", "federated_cb_only_ema", r"federated_cb_only_ema", "cb_ema"),
        ("zn_main", "federated_fedavg_cb_only", r"federated_fedavg_cb_only", "fedavg_cb"),
        ("zn_enc", "federated_enc_fedavg", r"federated_enc_fedavg", "enc_fedavg"),
        ("zn_enc", "federated_enc_fedprox", r"federated_enc_fedprox(_mu[\d.]+)?", "enc_fedprox"),
        ("zn_enc", "federated_enc_fedproto", r"federated_enc_fedproto(_lam\d+_\w+)?", "enc_fedproto"),
        ("zn_enc", "federated_enc_commoninit", r"federated_enc_commoninit", "enc_cominit"),
        ("zn_norev", "federated_cb_only_ema_norevive", r"federated_cb_only_ema_norevive", "norevive"),
        ("zn_a1", "federated_enc_fedavg", r"federated_enc_fedavg_bn-shared", "A1_bn"),
        ("zn_a2", "federated_enc_fedavg", r"federated_enc_fedavg_bn-shared_prior-partial", "A2_prior"),
        # ── LE BASELINE A BUDGET CORRETTO ────────────────────────────────────────────
        # Stesso arm, stesso seed, stessa coorte: cambia SOLO quanto a lungo si allena.
        #   `_es`  tetto alzato, early stopping ACCESO   -> il modello CONVERGITO
        #   `_ot`  tetto alzato, early stopping SPENTO   -> il modello SOVRALLENATO
        # Servono perche' i federati si fermavano a convergenza e le baseline per BUDGET:
        # il tetto di 10 000 step mordeva su 6 serie su 10, e su `ucr_011` — la serie da cui
        # dipende tutto l'effetto A2 — erano 4 client su 5. Finche' il confronto e' contro
        # `local` troncata, ogni vantaggio federato e' contaminato da quell'handicap.
        # ⚠️ `zn_es` esiste solo sulle serie che erano davvero troncate (6 local, 5 centr):
        # le altre riprodurrebbero se' stesse bit per bit e tengono i numeri di `zn_main`.
        ("zn_es", "local", r"local", "local_es"),
        ("zn_es", "centralized", r"centralized", "centr_es"),
        ("zn_ot", "local", r"local", "local_ot"),
        ("zn_ot", "centralized", r"centralized", "centr_ot"),
        # ── STAGE-2: gate, ctrl e varianti (2026-08-08) ──────────────────────────────
        # ctrl = A2 ripreso (stage-1 identico) + stage-2 FedAvg puro con ORACOLO FISSO:
        # e' la baseline di OGNI variante stage-2 (stesso oracolo da entrambi i lati del
        # contrasto). Il confronto variante−A2 mescolerebbe l'effetto oracolo.
        ("zn_a2s2_ctrl", "federated_enc_fedavg",
         r"federated_enc_fedavg_bn-shared_prior-partial", "s2_ctrl"),
        # ⛔ diagnostici federation-illegal: MAI in tabella principale, solo attribuzione.
        ("zn_fedtokcp", "fedtok_centralprior", r"fedtok_centralprior", "TETTO_s2"),
        ("zn_170_ctfp", "centraltok_fedprior", r"centraltok_fedprior", "CTOK_fedp"),
        ("zn_a2s2_tau64", "federated_enc_fedavg",
         r"federated_enc_fedavg_bn-shared_prior-partial_tau64", "tau64"),
        ("zn_a2s2_tau16", "federated_enc_fedavg",
         r"federated_enc_fedavg_bn-shared_prior-partial_tau16", "tau16"),
        ("zn_a2s2_t64adam", "federated_enc_fedavg",
         r"federated_enc_fedavg_bn-shared_prior-partial_tau64_srv-fedadam_slr0\.03", "t64adam"),
        ("zn_a2s2_pmu1", "federated_enc_fedavg",
         r"federated_enc_fedavg_bn-shared_prior-partial_pmu1_dec", "prox_mu1"),
        ("zn_a2s2_pmu3", "federated_enc_fedavg",
         r"federated_enc_fedavg_bn-shared_prior-partial_pmu3_dec", "prox_mu3"),
        # ── FIX-170 (una sola serie: righe quasi vuote per costruzione) ─────────────
        ("zn_170_ur", "federated_enc_fedavg",
         r"federated_enc_fedavg_bn-shared_cb-union_recluster_prior-partial", "170_ur"),
        ("zn_170_sched", "federated_enc_fedavg",
         r"federated_enc_fedavg_bn-shared_sched-cosine_prior-partial", "170_sched"),
        ("zn_170_k128", "federated_enc_fedavg",
         r"federated_enc_fedavg_bn-shared_prior-partial_K128", "170_k128")]

# Metriche presenti anche nei `records` dell'out-json. I top-K NO: vivono solo nei
# `report.json` per client dentro il checkpoint.
DA_OUTJSON = {"auroc", "auprc", "vus_pr"}


def cella(tag, cl, dirre, m, armjson=None):
    """Aggregato sui 5 client (--agg). None se la cella non e' completa: mai su meno di 5.

    Sorgente PRIMARIA il `report.json` per client (ha tutte e sei le metriche), FALLBACK
    l'out-json. Il fallback non e' un lusso: i checkpoint di `fedprox`/`fedproto` sono stati
    ripuliti su 8 serie su 10, quindi senza di esso quelle due colonne restano vuote anche
    dove la cella e' chiusa e il numero c'e'. Sui top-K il fallback non puo' esistere e la
    cella resta `—`: e' una perdita di dato reale, non un difetto del lettore.
    """
    ps = [p for p in glob.glob(f"artifacts/runs/{tag}/ckpt/{DS}/{cl}/seed0/*/*/report.json")
          if re.fullmatch(dirre, p.split("/seed0/")[1].split("/")[0])]
    if len(ps) >= 5:
        vs = []
        for p in ps:
            try:
                vs.append(json.load(open(p)).get(m))
            except Exception:
                vs = None
                break
        if vs and all(v is not None for v in vs):
            return AGG(vs)
    if m not in DA_OUTJSON or armjson is None:
        return None
    p = f"artifacts/runs/{tag}/{DS}/{cl}__{armjson}.json"
    if not os.path.exists(p):
        return None
    try:
        recs = json.load(open(p))["records"]
    except Exception:
        return None
    vs = [r.get(m) for r in recs]
    return AGG(vs) if len(vs) == 5 and all(isinstance(v, (int, float)) for v in vs) else None


V = {}                                   # (label, serie, metrica) -> valore
for tag, armjson, dirre, lab in ARMS:
    for s in SER:
        for m, _ in MET:
            v = cella(tag, s, dirre, m, armjson)
            if v is not None:
                V[(lab, s, m)] = v

# ── LA BASELINE A CONVERGENZA SU TUTTE E 10 LE SERIE ────────────────────────────────
# `zn_es` copre solo le serie dove il tetto di 10 000 step mordeva davvero. Sulle altre non
# e' stata rilanciata di proposito: stesso seed e un tetto che non morde danno la STESSA
# traiettoria fino all'early stop, quindi la cella riprodurrebbe se' stessa bit per bit e
# rifarla sarebbe solo compute bruciato. Incollare `zn_es` dove c'e' e `zn_main` dove non
# serviva da' quindi una colonna «local a convergenza» completa a n=10 — non una media di
# due condizioni diverse, ma la stessa condizione misurata due volte con lo stesso esito.
#
# ⚠️ NON e' la stessa cosa di `local_ot`. Quella ha l'early stopping SPENTO e corre fino a un
# tetto duro tenendo i pesi finali: e' uno stress test, non una baseline che qualcuno
# pubblicherebbe. Le due rispondono a domande diverse e possono dare segni diversi — quando
# succede, e' il dato interessante, non un errore da nascondere.
for _lab, _src in (("local_conv", "local"), ("centr_conv", "centralized")):
    for s in SER:
        for m, _ in MET:
            v = V.get((f"{'local' if _src == 'local' else 'centr'}_es", s, m))
            if v is None:
                v = V.get((_src, s, m))
            if v is not None:
                V[(_lab, s, m)] = v
ARMS = ARMS + [(None, None, None, "local_conv"), (None, None, None, "centr_conv")]

LABS = [l for *_, l in ARMS if any((l, s, "auroc") in V for s in SER)]


def hdr(t):
    print(f"\n{'='*100}\n{t}\n{'='*100}")


# ── 1. una tabella per metrica: serie x arm ─────────────────────────────────────
hdr(f"1. OGNI CELLA CHIUSA — {AGG_NAME} sui 5 client   (soglia rumore x{THR_K:.2f})")
for m, nome in MET:
    cols = [l for l in LABS if any((l, s, m) in V for s in SER)]
    if not cols:
        continue
    nota = "  ⚠ non aggregabile fra serie: leggere solo le colonne, mai la riga" if m in NON_AGG else ""
    print(f"\n{nome}{nota}")
    print(f"  {'serie':<10}" + "".join(f"{c[:11]:>13}" for c in cols))
    for s in SER:
        row = [V.get((l, s, m)) for l in cols]
        if all(v is None for v in row):
            continue
        print(f"  {s:<10}" + "".join(f"{'—':>13}" if v is None else f"{v:>13.3f}" for v in row))
    print(f"  {AGG_NAME:<10}" + "".join(
        (f"{AGG([V[(l, s, m)] for s in SER if (l, s, m) in V]):>13.3f}"
         if any((l, s, m) in V for s in SER) else f"{'—':>13}") for l in cols)
        + ("   (indicativa)" if m in NON_AGG else ""))
    print(f"  {'n serie':<10}" + "".join(
        f"{sum(1 for s in SER if (l, s, m) in V):>13d}" for l in cols))

# ── 2. serie che non discriminano — misurato, non assunto ───────────────────────
hdr("2. SERIE CHE NON DISCRIMINANO (contribuiscono 0 a ogni contrasto appaiato)")
cieche = []
for s in SER:
    vals = [V[(l, s, "paper_top1_acc_at_64")] for l in LABS
            if (l, s, "paper_top1_acc_at_64") in V]
    if len(vals) < 4:
        continue
    if max(vals) - min(vals) < 1e-9:
        cieche.append((s, f"tutti gli arm a {vals[0]:.2f}"))
if cieche:
    for s, why in cieche:
        print(f"  {s}: {why}")
else:
    print("  nessuna (su top-1, con le celle chiuse finora)")
print(f"  ⇒ n effettivo su top-1: {len([s for s in SER if any((l, s, 'paper_top1_acc_at_64') in V for l in LABS)]) - len(cieche)}")


# ── 3. contrasti appaiati ───────────────────────────────────────────────────────
def confronto(a, b, m):
    ks = [s for s in SER if (a, s, m) in V and (b, s, m) in V]
    if len(ks) < 2:
        return None
    d = [V[(a, s, m)] - V[(b, s, m)] for s in ks]
    md = AGG(d)
    pro = sum(1 for x in d if x != 0 and (x > 0) == (md > 0))
    con = sum(1 for x in d if x != 0 and (x > 0) != (md > 0))
    tie = sum(1 for x in d if x == 0)
    # 2 SD della differenza appaiata, mediata su n serie
    thr = 2 * (SD_PC[m] / math.sqrt(5)) * math.sqrt(2) / math.sqrt(len(ks)) * THR_K
    return md, len(ks), pro, con, tie, thr, abs(md) > thr, ks


COPPIE = [("centralized", "local", "la forbice dello studio"),
          ("federated", "local", "il metodo batte il puro locale?"),
          ("cb_only", "fedavg_cb", "CONTRIBUTO (A): primitiva suff-stat vs FedAvg"),
          ("cb_ema", "cb_only", "memoria del server"),
          ("norevive", "cb_ema", "RAMO N: la rianimazione e' la causa?"),
          ("cb_only", "local", "il dizionario condiviso paga?"),
          ("enc_fedavg", "cb_only", "federare anche l'encoder"),
          ("A1_bn", "enc_fedavg", "RAMO A1: BN condivisa vs chimera (isola il SOLO regime BN)"),
          ("A2_prior", "A1_bn", "RAMO A2: prior parziale condiviso, a BN gia' riparata"),
          ("A2_prior", "local", "A2 batte il puro locale?"),
          ("A2_prior", "centralized", "quanto manca ad A2 per il centralizzato"),
          # ── IL CONFRONTO EQUO ───────────────────────────────────────────────────────
          # I primi due misurano QUANTO l'handicap di budget valeva davvero: se `local_es`
          # e `local_ot` non si staccano da `local`, il troncamento non gonfiava niente e
          # tutti i contrasti sopra restano leggibili come sono. Gli altri due rifanno la
          # domanda principale contro una baseline non piu' handicappata — ed e' quella la
          # riga che va nel paper, non `A2_prior − local`.
          ("local_es", "local", "quanto valeva il troncamento (convergita vs troncata)"),
          ("local_ot", "local", "quanto valeva il troncamento (sovrallenata vs troncata)"),
          ("local_ot", "local_es", "sovrallenare oltre la convergenza aiuta o danneggia?"),
          ("A2_prior", "local_es", "⭐ A2 batte `local` A CONVERGENZA?"),
          ("A2_prior", "local_ot", "⭐ A2 batte `local` SOVRALLENATA?"),
          ("centr_ot", "local_ot", "la forbice dello studio, a budget pari"),
          # Le due righe a n=10 con la baseline convergita: sono QUESTE che vanno nel paper,
          # perche' il protocollo pre-registrato e' «early stopping acceso, tetto non
          # vincolante» — non «corri fino a un tetto duro».
          ("A2_prior", "local_conv", "⭐⭐ A2 − local a convergenza, n=10"),
          ("centr_conv", "local_conv", "⭐⭐ la forbice, entrambi a convergenza, n=10"),
          # ── STAGE-2 (2026-08-08): ogni variante SOLO contro s2_ctrl (stesso oracolo). ──
          ("s2_ctrl", "A2_prior", "null obbligatorio: resume+oracolo fisso ≠ A2? (deve ~0)"),
          ("TETTO_s2", "s2_ctrl", "⛔ il tetto: quanto stage-2 è recuperabile (diagnostico)"),
          ("TETTO_s2", "centralized", "⛔ residuo tokenizer: tetto vs centralizzato"),
          ("tau64", "s2_ctrl", "⭐ τ=64: il candidato di punta stage-2"),
          ("tau16", "s2_ctrl", "τ=16: il gradino che rompe la monotonia (64→16 cala)"),
          ("t64adam", "s2_ctrl", "FedAdam slr .03: bocciato dalla sonda (−0,18/−0,33)"),
          ("prox_mu1", "s2_ctrl", "riga di completezza FedProx μ=1 (predetto ≈0)"),
          ("prox_mu3", "s2_ctrl", "riga di completezza FedProx μ=3 (predetto ≈0)")]

hdr("3. CONTRASTI APPAIATI (Delta medio · concordanza di segno · soglia di rumore)")
for a, b, why in COPPIE:
    if not any((a, s, "auroc") in V for s in SER):
        continue
    print(f"\n{a} − {b}   [{why}]")
    for m, nome in MET:
        r = confronto(a, b, m)
        if r is None:
            continue
        md, n, pro, con, tie, thr, sig, ks = r
        conc = f"{pro}/{pro+con}" + (f" +{tie} pari" if tie else "")
        flag = "SOPRA il rumore" if sig else "dentro il rumore"
        # ── LEAVE-ONE-OUT: quanto il risultato dipende da UNA sola serie ──────────
        # Con n=9 e meta' pareggi, una singola serie patologica puo' produrre da sola un
        # Delta "significativo". Il numero da guardare non e' la media, e' quanto resta
        # togliendo la serie piu' influente -- ed e' esattamente il controllo che mancava
        # quando un +0,106 e' stato riportato e poi ritrattato.
        d = [V[(a, s, m)] - V[(b, s, m)] for s in ks]
        loo = [(AGG([x for j, x in enumerate(d) if j != i]), ks[i]) for i in range(len(d))]
        worst = min(loo, key=lambda t: abs(t[0]))
        note = ""
        if sig and abs(worst[0]) <= thr:
            note = f"   ⚠ SPARISCE togliendo {worst[1]} (resta {worst[0]:+.3f})"
        elif sig:
            note = f"   [regge: peggio {worst[0]:+.3f} senza {worst[1]}]"
        # ── TEST DEI SEGNI ──────────────────────────────────────────────────────
        # Con 7-9 pareggi su 10 la mediana del Delta e' 0,000 per costruzione e la soglia
        # sulla MAGNITUDINE non dice piu' niente. Il test dei segni invece SCARTA i pareggi
        # (che non sono evidenza in nessuna direzione) e misura solo se la direzione e'
        # coerente: 9 serie su 9 dallo stesso lato hanno p=0,004 anche se il Delta mediano
        # e' minuscolo. E' la statistica che regge quando la saturazione domina.
        k, tot = max(pro, con), pro + con
        pv = (2 * sum(math.comb(tot, i) for i in range(k, tot + 1)) / 2 ** tot) if tot else 1.0
        pv = min(pv, 1.0)
        alt = fmean(d) if AGG_NAME == "median" else median(d)
        alt_nome = "media" if AGG_NAME == "median" else "mediana"
        print(f"   {nome:<8} {md:+.3f}   ({alt_nome} {alt:+.3f})   n={n}   concordi {conc:<12} "
              f"segni p={pv:.3f}   soglia ±{thr:.3f}   {flag}{note}")
