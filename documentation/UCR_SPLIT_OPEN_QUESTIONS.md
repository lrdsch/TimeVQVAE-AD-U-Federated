# `ucr_split` — cose da esplorare, con impatto misurato

Scritto 2026-07-26, mentre il run `scripts/run_ucrsplit.sh` è al ~12 %.
Nasce da una domanda semplice — *"quanto faceva il modello originale su queste serie?"* — che ha
aperto cinque punti. Questo file li registra con l'impatto **misurato dove misurabile**, così non
vanno persi e non vanno ri-indagati da zero.

**Il numero di riferimento.** Lee, Malacarne, Aune, *"Explainable time series anomaly detection
using masked latent generative modeling"*, arXiv 2311.12550v5, **Table 1**, archivio UCR-TSA
intero (250 serie):

| metodo | top-1 | top-3 | top-5 |
|---|---|---|---|
| **TimeVQVAE-AD (pubblicato)** | **0.708** | 0.776 | 0.824 |
| Matrix Profile STUMPY | 0.512 | 0.684 | 0.744 |
| MERLIN / MERLIN++ | 0.424 | | |
| Convolutional AE | 0.352 | 0.412 | 0.448 |
| COCA | 0.236 | 0.328 | 0.408 |

**Il nostro, nella stessa metrica** (`paper_top{1,3,5}_acc_at_64`, medie su ~50 client dei primi
11 cluster completati):

| arm | top-1 | top-3 | top-5 |
|---|---|---|---|
| `centralized` (skyline) | 0.415 | 0.547 | 0.660 |
| `local` | 0.380 | 0.420 | 0.440 |
| `commoninit` (null) | 0.350 | 0.475 | 0.525 |
| `fedproto` | 0.350 | 0.525 | 0.525 |
| `fedprox` | 0.256 | 0.308 | 0.359 |
| `fedavg` | 0.222 | 0.311 | 0.378 |

Divario da spiegare sulla skyline: **0.415 → 0.708**.

---

## Classifica d'impatto

| # | punto | impatto stimato | stato |
|---|---|---|---|
| 1 | **finestra `2×periodo` vs 128 fissa** | il maggiore — tocca il 71 % delle serie | ⬜ **da testare** |
| 2 | bias del campione completato | grande, si riassorbe da solo | ✅ misurato |
| 3 | convenzione di tolleranza | **+0.09** top-1 | ✅ misurato |
| 4 | codebook 64 vs 128 | secondo ordine | ⬜ da testare |
| 5 | volume di dati per client | reale ma confuso | ✅ misurato, inconcludente |
| 6 | α-loop / moving average | **nessuno** | ✅ verificato fedele |

---

## 1. ⬜ La finestra: il paper usa `T = 2×periodo`, noi 128 fissa

**IL SOSPETTO PRINCIPALE.** L'Algorithm 1 del paper (pag. 9) comincia con *"Define a period
length P of x\*"* e poi *"x ∈ ℝ^T, T = 2P"*. **La finestra non è un iperparametro fisso: è
derivata dal periodo della serie.** Il nostro `config.py` dichiara di aver *rimosso* l'override
`period → window_length = 2*period` — decisione presa per `wsd_fed`, dove 2P = 2880 avrebbe fatto
crollare i client utilizzabili da 31 a 18 (vedi la nota `wsd-fed-period-window-trap`). Corretta
là, ma su UCR **cancella il protocollo del paper**.

Sui 82 cluster in run (periodi da `preprocessing/UCR_anomaly_dataset_periods.csv`, tutti e 250
presenti; elenco completo in [`ucr_split_series_periods.csv`](ucr_split_series_periods.csv)):

| | valore |
|---|---|
| periodo mediano | 139 |
| finestra del paper (2P), mediana | **278** |
| la nostra | **128** |
| rapporto 2P/128, mediana | **2.17** (max 13.72, min 0.36) |
| serie dove siamo **troppo stretti** (2P > 128) | **58/82 (71 %)** |
| serie dove siamo troppo larghi (2P < 128) | 24/82 |

Su tutte le 226 del dataset: 171 (76 %) hanno 2P > 128, rapporto mediano 2.59.

**Perché è più grave di quanto sembri.** `downsampled_width = 32` è fisso in entrambi, quindi la
finestra determina anche la **granularità temporale del token**: il paper comprime `2P/32`
campioni per token (fino a **55** su una serie con periodo 878), noi sempre `128/32 = 4`. Non è
una finestra più corta dello stesso modello — è un modello che guarda il segnale a una scala
completamente diversa, e che su metà delle serie non vede nemmeno un ciclo completo.

**Come testarlo (isolato, ~8 job, poche ore).** Prendere 6–8 cluster **già completati**,
rilanciare solo `local` e `centralized` con `window_length = 2*period` letto dal CSV,
**tenendo `paper_metrics_tolerance` fisso a 64**. Quest'ultimo punto è essenziale: la tolleranza
di default è `window_length//2`, quindi cambiare la finestra cambierebbe anche il metro di
giudizio e confonderebbe due variabili.

**Conseguenza se confermato.** Tutto lo studio federato gira su un modello sottodimensionato
rispetto al pubblicato, e le conclusioni sulla federazione andrebbero rifatte alla finestra
giusta. È il motivo per cui vale la pena spendere 8 job adesso invece di scoprirlo dopo.

---

## 2. ✅ Bias del campione completato — grande, ma si riassorbe

L'ordinamento LPT dello scheduler manda in esecuzione per primi i cluster più costosi, che sono
le serie più lunghe. Il campione su cui stiamo leggendo i risultati è quindi avverso per
costruzione:

| | n | `test_len` mediana | train mediana |
|---|---|---|---|
| cluster **completati** | 11 | **145 100** | 36 000 |
| **tutti** gli 82 in run | 82 | 24 950 | 7 543 |

**5.8× più lungo nel test.** Il top-1 chiede che l'argmax su *tutto* il test cada a bersaglio:
un test 6× più lungo dà ~6× più occasioni di sbagliare. **I numeri attuali sono pessimistici** e
saliranno da soli man mano che lo scheduler scende verso i cluster piccoli. Da non interpretare
finché il run non è finito, o almeno finché il campione completato non somiglia alla popolazione.

---

## 3. ✅ Convenzione di tolleranza — vale +0.09 top-1

Misurato ricalcolando il criterio dagli `scores.npz` già su disco, senza rilanciare nulla:

| arm | ±64 (nostro) | ±L, L = lunghezza anomalia (archivio) | ±100 | dentro il segmento |
|---|---|---|---|---|
| `centralized` | 0.436 | **0.527** | 0.436 | 0.436 |
| `local` | 0.380 | 0.420 | 0.380 | 0.340 |
| `commoninit` | 0.350 | 0.425 | 0.350 | 0.300 |
| `fedavg` | 0.222 | 0.267 | 0.222 | 0.200 |

±100 dà gli stessi risultati di ±64 — nessun colpo cade in quella fascia — quindi la scelta è
binaria fra "stretto fisso" e "proporzionale alla lunghezza dell'anomalia". **Da decidere e
dichiarare esplicitamente prima di pubblicare qualsiasi tabella**, perché sposta la skyline di
quasi un decimo.

---

## 4. ⬜ Codebook: paper 128, noi 64

Paper, sezione iperparametri (pag. 20): *"the same vector quantizer from TimeVQVAE except for the
use of a bigger codebook size (**128**) to better capture different patterns in time series
data"*. Noi usiamo `codebook_size = 64`, la metà. Impatto probabilmente di secondo ordine
rispetto alla finestra, ma è una scelta deliberata del paper che non abbiamo replicato, e si
testa con lo stesso set di job del punto 1 (`--codebook-size 128`).

Il paper aggiunge un dettaglio che non abbiamo verificato: *"in stage 2, we train the code
embeddings fresh without re-initialization with the learned codes in stage 1"*. Da controllare
cosa fa il nostro stage 2.

---

## 5. ✅ Volume di dati — effetto reale, ma non spiega il divario

`paper_top1_acc_at_64` in funzione della quota di training del client:

| arm | 10 % | 20 % | 30 % |
|---|---|---|---|
| `local` | 0.300 | 0.400 | **0.500** |
| `centralized` (90 %, un modello) | | | **0.436** |

Dentro `local` c'è una dose-response pulita, +0.10 per gradino. Ma **`centralized` con il 90 %
dei dati fa meno di `local` con il 30 %**, e la correlazione fra top-1 e log(campioni) dentro
`local` è −0.08, cioè nulla. La dose-response è confusa con *quale* serie: uno shard del 30 % di
una serie difficile fallisce comunque, e le celle hanno n ≈ 16.

**Conclusione: non attribuibile.** Rifare quando il run è completo, appaiando dentro cluster
invece di mediare fra cluster diversi.

---

## 6. ✅ α-loop e moving average — nessun divario, verificato

Sospetto sollevato e **ritirato**. Confronto riga per riga con l'Algorithm 1 e con la sezione
iperparametri:

| elemento | paper | nostro | |
|---|---|---|---|
| insieme di α (rate finestra latente) | `rw = 0.1, 0.3, 0.5` | `prior.score_window_size_rates = (0.1, 0.3, 0.5)` | ✅ identico |
| somma su α | sì | loop in `model/prior.py:355` | ✅ |
| media su frequenza `E_h[a_s]` | sì | `flat_T.mean(dim=1)`, `pipeline/detect.py:378` | ✅ |
| `a_final = (a_s + MA(a_s))/2` | sì, finestra T | `_moving_average_paper`, `pipeline/detect.py:232` | ✅ |
| stride rolling window | rate 0.1 | `eval_stride_rate = 0.1` | ✅ identico |

---

## Altre due cose emerse, non collegate al confronto col paper

### A. VUS-PR non è aggregabile fra serie UCR

I tassi di anomalia su `ucr_split` vanno da **0.0005 a 0.0335**, un fattore 60. Le metriche PR
hanno come baseline il tasso di positivi, quindi il VUS-PR mediano fra cluster misura la
distribuzione dei tassi, non il modello. Caso esemplare: `ucr_191` ha VUS-PR 0.211 ma **AUROC
1.000 e AUPRC 0.994** — il detector è perfetto, la metrica no.

Su `wsd_fed` i tassi stavano tutti fra 1 % e 3 %, per questo là VUS-PR era la scelta giusta e il
ledger dice che AUROC è l'asse sbagliato. **Su questo dataset quella regola si inverte.**

Regola operativa: confronti **appaiati dentro cluster** su VUS-PR (il tasso è costante, quindi è
lecito); per aggregare **fra** cluster usare `paper_top1/3/5` o il lift sul baseline casuale.

### B. Cluster da escludere con regola dichiarata prima di guardare i risultati

- `ucr_189` — l'anomalia dell'archivio è lunga **2 campioni** su 145 100 (`resperation3`,
  file `..._45000_158250_158251`). Non è un bug del builder: è l'etichetta. AUROC 0.002.
  Degenere, va escluso.
- `ucr_199` — AUROC 0.595, appena sopra il caso. Fallimento genuino del modello.

Serve una soglia dichiarata (es. "escludo i cluster dove la skyline sta sotto X, e dichiaro
quanti sono"), fissata prima di leggere i risultati finali.

### C. Il point-adjustment gonfia di 9×

`threshold_free.pa.vus_pr` = 0.135 contro `no_pa` 0.015 sullo stesso run. Lo calcoliamo, va bene
averlo, ma è sotto critica pesante in letteratura da Kim et al. 2022 in poi e **non può essere il
numero di testa**. Stessa cautela per `affiliation_f1`, che su anomalie rarissime satura
(0.855 con recall 1.000 su un detector che il top-1 boccia) e non discrimina.

---

## Cosa NON serve rilanciare

Tutto quanto sopra è stato misurato dai file già su disco: **261 `report.json`** e **261
`scores.npz`** sotto `artifacts/ucrsplit/ckpt/`. Il `report.json` contiene ~40 metriche per
client (incluse `paper_top{1,3,5}`, i gemelli point-adjusted e 3 strategie di soglia), di cui
solo 6 finiscono nel json di confronto. Il punto 1 e il punto 4 sono gli unici che richiedono
GPU.
