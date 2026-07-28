# `ucr_split` — stato, cosa lanciare, cosa manca

Aggiornato **2026-07-27 15:18**. Documento operativo: fatto / da fare / comandi.
Per il *perché* dei numeri vedi [UCR_SPLIT_OPEN_QUESTIONS.md](UCR_SPLIT_OPEN_QUESTIONS.md);
per il disegno sperimentale [UCR_SPLIT_EXPERIMENT_PLAN.md](UCR_SPLIT_EXPERIMENT_PLAN.md).

---

## 1. In una riga

Il dataset è costruito, il run baseline (cella A) è al **37 %** e sano, e le altre **tre celle
della griglia finestra×codebook sono implementate e smoke-testate ma NON lanciate**. La cosa
più importante ancora aperta non è un run: è che `window_length=128` è **sbagliata su tutte e
250 le serie UCR**, e finché non lo si testa non sappiamo se lo studio federato gira su un
modello sottodimensionato.

---

## 2. Fatto ✅

### Dataset
- `data/raw/ucr_split` — 226 cluster × 5 client = **1130 entità**, 833 MB, gitignored.
  Partizione contigua 10/10/20/20/30 % + val 10 %, test nativo intatto e condiviso.
- Verificato ri-leggendo i `.txt` grezzi: `concat(shard)+val == train` esatto su tutte e 226,
  `test == values[train_len:]` verbatim. → `scripts/check_ucr_split.py`
- Builder: `scripts/build_ucr_split.py` (opzioni `--shares`, `--val-pct`, `--limit`).

### Codice
- `pipeline/federated_eval.py`: aggiunti **`--window-length`** e **`--metrics-tolerance`**
  (nessuno dei due esisteva).
- `config.py`: `run_name()` ora porta il suffisso **`_w<N>`** — la finestra cambia la forma
  dell'input, quindi un checkpoint a 128 non si carica a 2P.
- `data.py` + `preprocessing/datasets/toy_fed.py`: dispatch del prefisso `ucr_split`.

### Runner
- `scripts/run_ucrsplit.sh` — cella A (W=128, cb=64). **In esecuzione.**
- `scripts/run_ucrsplit_w2p.sh` — celle B/C/D via `WINDOW_MODE` × `CODEBOOK`.
- `scripts/ucr_split_clusters.txt` — gli 82 rappresentanti di famiglia.

### Verifiche fatte (non rifarle)
| cosa | esito |
|---|---|
| forme a W = 128 / 226 / 1756 / 3028 | ✅ il downsample rate si adatta, ~32 posizioni latenti |
| tutti e 6 gli arm a W=350 | ✅ exit 0, 30 record |
| tutti e 6 gli arm a cb=128 (+W=350 = cella D) | ✅ exit 0, `K=128` nel path federato |
| memoria a W=1756, batch 64 | ✅ **5.91 GB / 48** — nessun rischio OOM |
| `_converged_budget` agli estremi | ✅ nessun budget degenere (peggior client = 1 batch/epoca) |
| collisione dei path fra celle | ✅ root separate `artifacts/ucrsplit_<tag>/` |

### Analisi già prodotte
- Confronto col paper (Table 1): pubblicato top-1 **0.708**, nostro `centralized` **0.415**.
- Attribuzione del divario: tolleranza **+0.09**, bias del campione (grande, si riassorbe),
  α-loop e moving-average **fedeli, nessun divario**, volume dati **inconcludente**,
  **finestra = sospetto principale, non misurato**.
- Trappola metrica: **VUS-PR non è aggregabile fra serie UCR** (griglia a 250 soglie →
  AUPRC 0.994 letta come VUS-PR 0.211). Usare `paper_top1/3/5`.
- Tabelle: [UCR_ALL_SERIES_WINDOWS.md](UCR_ALL_SERIES_WINDOWS.md) (250 serie × uso, finestre
  per client, verdetto su 128), [UCR_ALL_SERIES.md](UCR_ALL_SERIES.md), i due CSV.

---

## 3. In corso 🔄

**Cella A — `run_ucrsplit.sh`** (W=128, cb=64), lanciata il 25/07 21:51.

| | |
|---|---|
| job | **180 / 492** · **0 falliti** |
| cluster completi | **29 / 82** (+3 parziali) |
| wall trascorso | 41.4 h · ritmo 4.34 job/h |
| **ETA residua** | **~3.0 giorni** (fine ~30/07) |

Monitoraggio: `tail -f logs/ucrsplit/_orchestrator.log`
Uccidere: `pkill -f "[r]un_ucrsplit.sh"` poi `pkill -f "[f]ederated_eval.py"` (resumable —
un job il cui json esiste viene saltato).

---

## 4. Pronto, NON lanciato ⬜

La griglia. La finestra e il codebook sono **due** differenze dal paper e vanno separate:
lanciare solo D le muove insieme e non attribuisce niente.

| cella | config | script | job | stima |
|---|---|---|---:|---|
| **A** | W=128, cb=64 | `run_ucrsplit.sh` | 492 | 🔄 in corso |
| **B** | **W=2P**, cb=64 | `WINDOW_MODE=2p` | 456 | ~2.6 gg |
| **C** | W=128, **cb=128** | `WINDOW_MODE=fixed CODEBOOK=128` | 492 | ~2.8 gg |
| **D** | **W=2P**, **cb=128** | `CODEBOOK=128` | 456 | ~2.6 gg |

```bash
# PROBE — 8 cluster, 48 job, ~6 slot-ore ciascuno. Da fare PRIMA delle celle piene.
SUBSET=8 WINDOW_MODE=2p                 bash scripts/run_ucrsplit_w2p.sh   # B
SUBSET=8 WINDOW_MODE=fixed CODEBOOK=128 bash scripts/run_ucrsplit_w2p.sh   # C
SUBSET=8 WINDOW_MODE=2p    CODEBOOK=128 bash scripts/run_ucrsplit_w2p.sh   # D

# CELLA PIENA (esempio B)
setsid nohup env WINDOW_MODE=2p bash scripts/run_ucrsplit_w2p.sh \
  > logs/ucrsplit_w2p_boot.log 2>&1 < /dev/null &
```

⚠️ **I run condividono i 14 slot e il muro della CPU a 16 core.** Lanciarne uno adesso
dimezza la velocità di A. Aggiungere `DRYRUN=1` per vedere il piano senza eseguire.

**Nota sullo scope:** B e D girano su **76** cluster, non 82 — a 2P la finestra è più lunga
dello shard del client al 10 % in 6 cluster (4 con finestre *negative*). Sono stampati
all'avvio, mai in silenzio. C gira su tutti e 82, così è confrontabile con A cluster per
cluster.

---

## 5. Manca ancora ⬜

### Bloccanti per qualsiasi claim
1. **Il test della finestra.** È la domanda aperta più grande dell'intero studio: se 2P porta
   il top-1 da 0.415 verso 0.65–0.70, allora tutto il lavoro federato finora gira su un
   modello miope e le conclusioni vanno rifatte. → probe della cella B, poche ore.
2. **Regola di esclusione dichiarata prima di leggere i risultati.** `ucr_189` ha
   un'etichetta di **2 campioni** su 145k (degenere, AUROC 0.002) e `ucr_199` è un
   fallimento vero (AUROC 0.595). Serve una soglia fissata *ex ante*, tipo "escludo dove la
   skyline sta sotto X, e dichiaro quanti sono".
3. **Script di aggregazione con le metriche giuste.** `scripts/fed_aggregate.py` usa VUS-PR,
   che fra cluster non significa niente. Serve una variante su `paper_top1/3/5` + confronti
   appaiati dentro cluster. **Tutto ricostruibile offline** dai 261+ `report.json` e
   `scores.npz` già su disco — nessuna GPU.

### Importanti, non bloccanti
4. **Codebook 64 vs 128** → celle C/D.
5. **Deriva fra shard.** La claim "IID by construction" è **sovraffermata**: spread delle
   medie >0.25σ sul 41 % delle serie, >1σ sul 4 %. Stratificare l'analisi per deriva, oppure
   costruire un gemello **a strisce**. ⚠️ Le strisce richiedono di partizionare gli *indici di
   finestra*, non di concatenare array: incollare blocchi non adiacenti inietta discontinuità
   che un rilevatore di anomalie legge come anomalie, e i client da 270 campioni non sono
   stripabili affatto.
6. **Seed multipli.** Tutto è a seed singolo. L'inferenza la porta n=82 cluster, ma 3 seed
   sull'arm vincente sono il follow-up onesto prima di pubblicare.
7. **`thre` di VUS.** Alzarlo sblocca la metrica sulle serie a bassa densità di anomalie.
   Ricalcolabile offline.
8. **Stage-2 code embeddings.** Il paper li allena da zero senza re-init dallo stage 1; non
   abbiamo verificato cosa fa il nostro.

### Aperto, di analisi
9. **Confronto per singola serie col paper.** Il paper dice di aver pubblicato i CSV dei
   punteggi predetti sul loro GitHub — è la fonte esatta per "su `ucr_199` loro cosa
   facevano". Non scaricati.
10. **Confound temporale del val.** I blocchi contigui mettono il client al 30 % adiacente al
    val, favorendone l'early stopping. Mitigazione: ricostruire con `--shares` permutate.

---

## 6. Ordine consigliato

1. Lasciar finire **A** (~3 giorni), è la baseline di tutto.
2. Nel frattempo, a costo zero di GPU: lo **script di aggregazione** (punto 3) e la **regola
   di esclusione** (punto 2), entrambi dai file già su disco.
3. Appena A libera gli slot: i **tre probe** `SUBSET=8` (~mezza giornata in tutto). Dicono
   quale manopola conta.
4. Solo allora impegnare 2.6–2.8 giorni sulla cella piena che il probe indica.

Il punto 3 è quello che dà più informazione per unità di tempo dell'intero piano.
