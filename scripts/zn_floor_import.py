"""Porta il floor gia' misurato su `ucr_split_w2p` dentro `artifacts/runs/zn_floor/floor/`.

⚠️ NON e' un trucco contabile. Il floor **non applica mai** `window_normalization`: in
`floor_eval.accumulate` la `SlidingWindowDataset` serve solo per `.indices`, e i valori
arrivano da `FH._windows(x, W, blk)` sulla serie grezza. La z-norm vive in `__getitem__`,
che il floor non chiama. Quindi il cutoff z-norm del 2026-08-04 invalida le run DEEP, non
il floor: quelle righe sono esattamente cio' che produrrei rilanciando oggi.

E non e' un'asserzione, e' misurato. Il 2026-08-06 ho rilanciato `ucr_001` con il codice di
oggi e confrontato con l'archivio: **30 metriche, differenza massima 0.000e+00**.

CINQUE CONTROLLI, tutti passati prima di scrivere un byte:
  1. il codice non cambia   `floor_heads.py` 30/07, `floor_eval.py` 01/08, git pulito
  2. lo schema combacia     record `_schema=2` == `floor_eval.SCHEMA` di oggi
  3. il build non cambia    `data/raw/ucr_split_w2p/` fermo al 29/07, nulla dopo il 04/08
  4. la griglia combacia    floor stride 41/18/33/41/97/54/67/20/25 == `eval_stride_rate=0.1`
                            x window del percorso deep, serie per serie
  5. riproduzione           bit-identica su ucr_001 (sopra)

COSA NON FACCIO. Non spaccio queste righe per appena calcolate. I numeri sono identici e
ogni strumento le legge come native -- stesso percorso, stesso formato, stesso schema -- ma
`PROVENANCE.md` dice da dove vengono. In questo repo i risultati si sono gia' persi due volte
per provenienza confusa (il cutoff z-norm, la purga del 30/07): la riga di testo costa nulla
e vale la prossima sessione che si chiede «questo quando l'abbiamo girato?».

⚠️ `ucr_082` NON c'e' in archivio ed e' l'unica serie da lanciare davvero.
⚠️ `threshold_rule = quantile_fallback`: solo le metriche threshold-free (AUROC, AUPRC,
   VUS-PR, top-K) sono confrontabili con gli arm deep. Le F1 no.
"""
import json
import os
from collections import defaultdict

REPO = "/home/leonardo/PhD/TimeVQVAE-AD-U-Federated"
os.chdir(REPO)
SRC = "evidence/records_pre_znorm_20260804_floor.json"
DST = "artifacts/runs/zn_floor/floor"
DS = "ucr_split_w2p"

W2S = {408: "ucr_001", 182: "ucr_011", 330: "ucr_014", 414: "ucr_043", 892: "ucr_082",
       970: "ucr_083", 536: "ucr_086", 674: "ucr_170", 202: "ucr_222", 254: "ucr_229"}

arch = json.load(open(SRC))
rows = [r for r in arch if r["payload"]["meta"]["dataset"] == DS]
print(f"righe d'archivio su {DS}: {len(rows)}")

# ── 0. DEDUP — 63 arm su 189 compaiono DUE volte ────────────────────────────────
# `ucrNNN_floor` e `ucrNNN_floor100` su 001/011/222. NON sono due misure: stessa
# tolleranza (64), stesso stride, stesso threshold_q, e su **6615 metriche la differenza
# massima e' 0.000e+00**. L'unico scarto e' `_secs`, il tempo di esecuzione.
# `floor100` porta in piu' la colonna `paper_top1_acc_at_100` (la tolleranza del protocollo,
# calcolata offline perche' `metrics_tolerance` sta nel fingerprint della coorte e non si
# tocca). Quindi e' un SOPRAINSIEME e va preferito.
# ⚠️ Il nome del file NON contiene la tolleranza, quindi i due collidono: un "primo che
# arriva vince" tiene la variante povera meta' delle volte. Si sceglie esplicitamente.
byfile = defaultdict(list)
for r in rows:
    byfile[r["file"]].append(r)
ricchezza = lambda r: sum(1 for v in r["payload"]["records"][0].values() if v is not None)
rows = [max(v, key=ricchezza) for v in byfile.values()]
print(f"dopo il dedup: {len(rows)} arm  ({sum(1 for v in byfile.values() if len(v) > 1)} "
      f"erano doppi, tenuta la variante con la colonna @100)")

os.makedirs(DST, exist_ok=True)

# ── 1. un JSON per arm, esattamente il nome che scriverebbe floor_eval ──────────
allrecs, per_series, arms = [], defaultdict(set), set()
for r in rows:
    p = DST + "/" + r["file"]
    json.dump(r["payload"], open(p, "w"), indent=2)
    arms.add(r["payload"]["meta"]["arm"])
    for rec in r["payload"]["records"]:
        allrecs.append(rec)
        per_series[W2S.get(r["payload"]["meta"]["window"], "?")].add(rec["_arm"])

# ── 2. il records_*.jsonl che il resume di floor_eval legge ────────────────────
# Va UNITO, non sovrascritto: `ucr_082` arrivera' da una run vera nella stessa cartella e
# rifare da zero cancellerebbe l'import (o viceversa).
jl = f"{DST}/records_{DS}.jsonl"
esistenti = []
if os.path.exists(jl):
    esistenti = [json.loads(l) for l in open(jl) if l.strip()]
chiavi = {(r["_arm"], r["_entity"]) for r in esistenti}
nuovi = [r for r in allrecs if (r["_arm"], r["_entity"]) not in chiavi]
with open(jl, "a") as f:
    for r in nuovi:
        f.write(json.dumps(r) + "\n")

print(f"\nscritti {len(arms)} file d'arm  ·  {len(nuovi)} righe aggiunte a records_{DS}.jsonl "
      f"({len(esistenti)} gia' presenti)")
print(f"\n{'serie':<10}{'arm':>6}")
for s in sorted(W2S.values()):
    n = len(per_series.get(s, ()))
    print(f"{s:<10}{n:>6}" + ("   ⛔ DA LANCIARE" if n == 0 else ""))
