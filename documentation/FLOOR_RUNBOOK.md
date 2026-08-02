# FLOOR — runbook dei lanci

**Scope congelato il 2026-07-30** per un **workshop paper**. Il design completo sta in
[`FLOOR_BASELINE.md`](FLOOR_BASELINE.md); qui c'è *cosa si lancia, perché, e cosa
deliberatamente non si lancia*.

    PY=/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10

**Zero GPU.** `floor_heads.py` non importa nemmeno torch: è numpy puro. La manopola di
velocità è `--jobs` (`FLOOR_JOBS`).

---

## 0. Il gate — prima di credere a qualunque numero

    $PY scripts/floor_heads.py            # 17 invarianti standalone (zero import dal repo)
    $PY scripts/floor_eval.py --selftest  # ri-esegue quei 17 + i 2 contratti verso il repo

Sono **19 invarianti distinti**, non 36: `floor_eval --selftest` ri-esegue tutta la suite di
`floor_heads`, quindi contare le righe `[PASS]` dei due log insieme conta i 17 due volte.

Entrambi devono stampare **ALL PASS**. `run_floor.sh` li esegue da sé e **si ferma** se
falliscono — ⚠️ **tranne per `consolidate`**, che salta il gate perché non riscora niente. Il primo asserisce, fra l'altro, che la media mobile della testa è
**bit-identica** a `detect._moving_average_paper` (`max|Δ| = 0.00e+00`, questa è *esatta*),
che `ma_c(k=3) ≡ 0.25·diff1` **a precisione macchina** (`max|Δ| = 3.2e-13`, non esatta: la
cumsum), e che il controllo del gauge PCA scatta su un fixture **eterogeneo** — il
regime in cui vive il meccanismo (§8.2b: una conclusione è già stata ritrattata per
averlo testato su un fixture omogeneo).

---

## 1. 🔴 Lo scope: SOLO baseline semplici

> **Ogni testa, fittata per client (`--modes local`), su due dataset. Nessun asse di
> federazione.**

Il runner porta **solo** questo. Tutto il resto resta implementato, self-testato e
selezionabile a mano — `floor_eval` stampa `⚠ OUT OF PAPER SCOPE` — ma non entra in tabella.

### Dentro

| voce | dove | perché |
|---|---|---|
| `ma_c` k=10, impulse on, `--suite on` | batch A, wsd | **la riga del paper** (`federated_method.tex:452`). Zero parametri ⇒ arm-invariante |
| impulse **off**, `--suite on` | batch B, wsd | risultato secondario di prima classe: l'impulse term vale **+0.058 VUS-PR** di mediana **appaiata** (HL +0.068, CI [+0.025, +0.071], p=8.9e-05, positivo su 26/31). ⚠️ **Il +0.180 citato finora era la differenza di mediane marginali, sovrastimata 3,1×** — §6 difetto 11 |
| sweep di k, 6 valori | batch C, wsd | scudo anti-tuning: k=10 è l'argmax *pre-registrato*, e sopra k=20 la curva è piatta |
| `ma_causal`, `ar`, `pca` | batch D + F | insieme a `ma_c` sono le **4 teste di decisione** ⇒ l'inviluppo |
| `diff1`, `random` | batch D | estremo basso della banda e ancora inferiore. Nel testo, non in tabella |
| `gauss` | batch D, **solo wsd** | l'estremo a ~8k parametri: la banda copre **0 → 33 → 1024 → 8256** parametri fittati |
| l'**inviluppo** su 4 teste | `floor_envelope.py` | la barra superiore, oracolo su test. Va pubblicata **accanto** alla testa singola, mai al suo posto |

### Fuori: TUTTO l'asse di federazione

Tagliato il 2026-07-30 dopo un audit a 21 agenti, uno per modello. Non era una baseline: era
uno studio meccanicistico su un gemello convesso, e l'audit ha trovato che è quasi tutto
indifendibile **oggi**.

| voce | motivo |
|---|---|
| `central` | **bit-identico** a `fed_exact` su tutte e 21 le metriche per ogni (testa, client): una riga contata due volte. È parte del perché i «18 confronti appaiati» sono in realtà ~9 |
| `fed_exact` + `--witness` | resta vero e resta il più bel risultato tecnico, ma è `k-FED`/Prop.1 — §8.1 vieta di presentarlo come algoritmo nuovo, e senza il lato deep non misura nulla che venga riportato |
| `fed_scaleonly` | Δ = 0 è **forzato** dall'omogeneità positiva di impulse + max sui canali + soglia a quantile: un teorema dentro una famiglia di null empirici |
| `fed_oneclient` | il trattamento è applicato per **cluster** ma il p è calcolato su n=31 **entità** — la stessa inflazione di unità che il progetto vieta su `ucr_split`. Broadcaster `ents_order[0]`, arbitrario |
| `fed_localgd` τ=16 R=30 | 🔴 **TRONCATO**: ρ(M̄)=0.9916, servirebbero ~544 round, e a R=30 il suo obiettivo ridge è **peggiore** di FedAvg one-shot (146 vs 113). Per la vostra regola dura train-to-convergence non è riportabile |
| `fed_naive` / `_aligned` | i due arm differiscono nell'**OPERATORE** (base grezza+QR contro proiettore+eigh), non solo nel gauge ⇒ «identical aggregation operator» è **falso**. E il gauge è **una sola estrazione cablata** (`rng(1000+i)`, fuori dal tag): su 40 draw l'energia catturata va da 0.226 a 0.906, quindi il −0.33 è un effect size non identificato |
| `fed_fedavg` + eccesso di obiettivo | difendibile solo con due riferimenti mai stampati: normalizzando per `J(w*)` l'ordine fra cluster **si ribalta**, e l'eccesso del braccio `local` sullo stesso obiettivo è **4-6× più grande** ⇒ «l'obiettivo peggiora mentre la metrica non si muove» si legge **al contrario** |

Il codice c'è tutto e i fix dei difetti 11-12 restano: se un referee lo chiede, si lancia a
mano (§8) e i numeri escono già con i loro riferimenti.

### Fuori, per altri motivi

| voce | motivo |
|---|---|
| `fed_prox` (sweep di μ) | i limiti `μ=0 → local` (0.00e+00) e `μ→∞ → global` (1.88e-13) sono **già** asseriti a precisione macchina nel selftest, e §8.3 vieta l'unica inferenza che uno sweep comprerebbe |
| `central_capN` | ri-testa un claim del ledger già marcato **INVERTITO** (`centralized` 0.585 < `local` 0.617) |
| **`ucr_split` a W=128** (226 serie) | 128 è la finestra sbagliata su tutte e 250 le serie UCR |
| i **5 toy** | nessun claim vi si appoggia, e il loro lato deep è stale dal rebuild 14→47 client |
| **`ucr_ad` / `ucr_pool`** | un cluster da 248 entità: `local` vs `federated` non ha significato lì |
| `gauss` su `w2p` | O(W²) memoria / O(W³) solve con W fino a 3028, e non entra nell'inviluppo |
| `fed_sharedbasis[_avg]`, `knn`, `fed_coreset`, `fed_ensemble`, `fed_dp`, `ema_shard` | **non implementati.** `fed_ensemble` sarebbe il supporto più pulito per weight-space vs function-space ed è il miglior candidato per il seguito; `fed_dp` aprirebbe un fronte privacy che il paper declina (`tex:322-324`); `ema_shard` è marcato *PLUMBING — NESSUNA CLAIM* dal design stesso |

---

## 2. Il lancio

    bash scripts/run_floor.sh smoke        # ~8 min: 1 cluster wsd + 3 cluster w2p a
                                           #   finestre distinte, ogni cella tenuta
    bash scripts/run_floor.sh wsd          # la matrice wsd_fed          (~1 h)
    bash scripts/run_floor.sh ucr          # ucr_split_w2p, 180 serie    (~3-5 h)
    bash scripts/run_floor.sh consolidate  # inviluppo + tabella + stats (minuti)
    bash scripts/run_floor.sh all          # wsd -> ucr -> consolidate

    # il lancio non attended. Il `mkdir` serve: la redirezione la valuta la SHELL prima di
    # eseguire lo script, quindi il `mkdir -p` interno arriva troppo tardi, e `logs/` e'
    # gitignorata -> su un albero fresco il comando senza mkdir fallisce e non lancia nulla.
    mkdir -p logs/floor && FLOOR_JOBS=12 nohup bash scripts/run_floor.sh all \
        > logs/floor/all.log 2>&1 &

**L'argomento e' obbligatorio.** Senza, o con una parola non riconosciuta, lo script stampa
l'uso ed esce **2**. (Prima il default era `all`: dimenticare l'argomento lanciava il target
piu' caro, e un refuso tipo `usr` per `ucr` stampava `done ->` ed esciva 0 — indistinguibile
da un run riuscito, soprattutto sotto `nohup`.)

Tre variabili d'ambiente: `FLOOR_JOBS` (worker, default 8), **`FLOOR_OUT`** (redirige tutta la
catena, `artifacts/floor` per default) e **`FLOOR_SMOKE_OUT`** (idem per lo smoke).

**Sempre `smoke` prima.** Esercita ogni percorso di codice che le batch vere usano — le 7
teste, `--suite on`, l'impulse off, un punto dello sweep di k, la mappa di finestre per-serie
di `w2p` (3 cluster a finestre **distinte**: e' la regressione per lo split `_arm`/`_model`),
l'inviluppo, la tabella e **entrambe** le metriche delle statistiche col lookup della
tolleranza — in minuti invece di ore, scrivendo in `artifacts/floor_smoke/` per non
contaminare i record veri.

**Resumable.** Ogni batch salta le coppie `(arm, entity)` già su disco allo schema
corrente. **Nessuna batch passa `--force`**: A e F lo facevano, il che significava che
un'interruzione della batch headline o di quella UCR ripartiva da zero — le due batch più
lunghe erano le due che non sapevano riprendere.

---

## 3. Le batch, una per una

### wsd_fed — 4 cluster / 31 client, `metrics_tolerance=14`

Ogni riga qui sotto sono **flag**, non un comando: `run_floor.sh` le lancia come

    $PY scripts/floor_eval.py --jobs $FLOOR_JOBS --out-dir artifacts/floor <flag>

| batch | flag |
|---|---|
| A | `--heads ma_c --modes local --k 10 --impulse on --suite on` |
| B | `--heads ma_c --modes local --k 10 --impulse off --suite on` |
| C | `for k in 3 5 20 32 50 128; do ... --heads ma_c --modes local --k $k --impulse on --suite off; done` — **sei** run, sei arm `floor_ma_c_k<k>` |
| D | `--heads ma_causal,diff1,random,ar,pca,gauss --modes local --impulse on --suite off` |

⚠️ La riga C e' un ciclo: incollare `--k {3,5,20,32,50,128}` fa espandere le graffe da bash e
argparse muore con `unrecognized arguments: 5 20 32 50 128`.

**Non c'è una batch E.** L'asse di federazione è fuori scope (§1): ogni batch è `--modes local`.
Sono **14 arm** (1 + 1 + 6 + 6) × **31 client** su 4 cluster (5/11/9/6) = **434 righe**, di cui
**62** (i 31 client × gli arm A e B) con `--suite on`. Zero celle saltate: a `--modes local`
nessuna combinazione (testa × modo) e' provabilmente nulla.

### ucr_split_w2p — 180 serie / 900 client, W = 2×periodo per serie

    $PY scripts/floor_eval.py --dataset ucr_split_w2p \
        --heads ma_c,ma_causal,ar,pca,random --modes local --impulse on --suite off \
        --out-dir artifacts/floor

(dentro `run_floor.sh` e' la batch `F_w2p_heads`; `run` e' una funzione interna allo script,
non un comando che esista nella tua shell.)

🔴 **Non passare `--window`.** La mappa per-serie sta nel metadata e viene letta da sola:

    [floor] ucr_split_w2p carries per-series windows (180 series, W in [46, 3028])
            -> using them per cluster
    [floor] W=per-series 46..3028 (79 distinct) eval_stride=13 tolerance=64 q=0.99

L'arm prende il tag `__w<W>` per ogni **W ≠ 128** — e su questa build **nessuna** delle 180
serie ha W = 128 — quindi non collide con eventuali righe a 128.
Un `--window` esplicito vince, perché forzarlo è un'ablazione legittima.

**Tutte e 180, non un sottoinsieme.** La lista storica da 82 cluster era privilegiata solo
perché la usavano i runner deep; **nessun run deep esiste sotto un cohort fingerprint**,
quindi quella lista non conferisce nulla e sottoinsiemarla sarebbe una restrizione
inspiegata. Per la cronaca: di quelle 82 ne sopravvivono **61** in questa build, non le 76
che il vecchio runbook implicava (quel numero riguardava l'universo a 226 serie di
`ucr_split`, un insieme diverso).

Distribuzione di W: min 46, mediana 250, p90 950, max 3028. Sopra 1024 ci sono 10 cluster,
sopra 2048 tre. Il *solve* PCA su tutti e 180 costa **~4 min** in totale — il collo è lo
scoring e il calcolo VUS/PATE, non l'algebra.

---

## 4. Consolidamento

    bash scripts/run_floor.sh consolidate

fa, per `wsd_fed` e `ucr_split_w2p`:

    $PY scripts/floor_envelope.py --dataset <ds> --records-dir artifacts/floor \
        --out-dir artifacts/floor/_derived
    $PY scripts/floor_table.py   --dataset <ds> \
        --records-dir "artifacts/floor,artifacts/floor/_derived" --out-dir artifacts/floor
    $PY scripts/floor_stats.py   --dataset <ds> --metric vus_pr --csv-dir artifacts/floor
    $PY scripts/floor_stats.py   --dataset wsd_fed       --metric paper_top1_acc_at_14 --csv-dir artifacts/floor
    $PY scripts/floor_stats.py   --dataset ucr_split_w2p --metric paper_top1_acc_at_64 --csv-dir artifacts/floor

La tolleranza nel nome della metrica viene da `data/raw/<ds>/metadata.json:metrics_tolerance`
(**14** su `wsd_fed`, **64** su `ucr_split_w2p`).

### `floor_envelope.py` — la barra superiore, finalmente riproducibile

`floor_heads.DECISION_HEADS` era definito alla riga 602 e **letto da nessuno**: la coppia
`0.4993 / 0.5585` nel `.tex` era calcolata fuori dal repo, quindi irriproducibile (viola
§8.9). Ora esiste lo script.

Cosa fa, con precisione: per ogni `(cluster, entity)` sceglie la **singola testa** col
miglior `--select-metric` e riporta **tutta** la riga di quella testa. È **selezione di
testa**, non massimizzazione per-metrica — quest'ultima prenderebbe `vus_pr` da una testa e
`auprc` da un'altra, descrivendo un detector che nessuno può costruire.

È un **oracolo su test**: la selezione usa la metrica di test, perché §6 registra che in
questo repo non esiste un criterio label-free capace di ordinare queste teste (AR predice un
passo avanti, PCA ricostruisce una finestra che *contiene* il punto: i loro MSE su val non
sono sulla stessa scala). Ogni riga porta quindi `_oracle: true` e l'arm si chiama
`floor_env4`. **Va pubblicato accanto alla testa singola pre-registrata, mai al suo posto.**

Si **rifiuta** di produrre un numero se: manca una delle 4 teste; gli insiemi di entità
differiscono fra teste (un max su un insieme irregolare confronterebbe l'inviluppo sulle
entità facili contro la testa singola su tutte); due arm diversi corrispondono alla stessa
`(testa, entità)`; o le teste non concordano su `_window` / `_tolerance` / `_eval_stride`.
Stampa anche le vittorie per testa — è il conteggio "`ma_c` 13, `pca` 9, `ma_causal` 6,
`ar` 3" che mostra che nessuna testa domina cliente per cliente.

### `floor_table.py`

    --records-dir  lista separata da virgole, glob ammessi.  "artifacts/runs/*/floor"
                   unisce ogni tag di coorte, applicando la stessa deduplica per
                   (arm, entity) che usa floor_eval. È ANCHE lo step di merge.
    --out-dir      dove finiscono CSV e matrix json. Default: il primo records-dir se e' un
                   path letterale; se e' un GLOB, `artifacts/floor` (non la prima directory
                   che il glob risolve). Un path fuori dal repo va bene.

Stampa in coda i `cohort_fingerprint` di ogni `artifacts/runs/<tag>/` e un avviso se
compaiono alberi pre-coorte: una riga floor-vs-deep costruita su quelli **non** è
fingerprint-matched e va etichettata, non cancellata.

### `floor_stats.py`

    --csv-dir   dove sta <ds>_all_models.csv
    --unit      auto: `cluster` su ucr_split*, `entity` su wsd (⚠ mai per-client su UCR:
                le 5 shard condividono il test set, ICC 0.82, i p si gonfiano di √5)
    --min-n     sotto questo n appaiato l'arm è riportato come SKIPPED, mai omesso
    --tost-margin  se il margine è più stretto dell'`m80` dell'arm, il verdetto TOST è
                   marcato UNDERPOWERED

La colonna **`m80`** è il margine necessario per l'80% di potenza **all'n appaiato di
quell'arm**. Prima veniva calcolata a `n = len(base)`, il conteggio di chiavi dell'arm di
riferimento, che è ≥ l'n appaiato di ogni arm e quindi riportava un margine **più piccolo**
del necessario: esattamente la direzione che fa sembrare adeguatamente potente un TOST che
non lo è.

---

## 5. Le regole che non si negoziano

| # | regola | perché |
|---|---|---|
| 1 | **Un processo per dataset.** Mai due sullo stesso `--dataset` | ⚠️ **non** e' la config: `build_cfg` costruisce un `Config()` nuovo a ogni chiamata (verificato, nessun leak fra dataset nello stesso processo). Il motivo vero e' la regola 2 — due processi appendono allo stesso `records_<ds>.jsonl` |
| 2 | **Shard ⇒ `--out-dir` distinto**, poi merge via `--records-dir` | le righe sono appese con `O_APPEND` e i record JSON superano i 4 KB di `PIPE_BUF`: due processi sullo stesso file possono **strappare una riga a metà** |
| 3 | `--window` cambia l'identità dell'**arm** (`__w<W>`), **non** quella del **modello** | `W` fissa *sia* la geometria di accumulo *sia* la lunghezza della MA dell'impulse term ⇒ l'arm deve portarla, o due finestre si fondono in silenzio. Ma su `ucr_split_w2p` la finestra è per-serie (79 valori), quindi **un modello indossa 79 nomi di arm**: `_arm` è l'identità di **storage**, `_model` (= `_arm` senza `__w<W>`) è quella di **analisi**. Tabelle e test appaiati vanno **sempre** per modello, e il range di finestre va in didascalia |
| 4 | `--fit-stride` cambia l'identità dell'arm (`__fs<N>`) | cambia lo **stimatore**: su `ucr_split` a stride 1 solo 60/1130 client hanno `n < 2W` e nessuno `n < W`; a stride 13 diventano 734/1130 e 546/1130 con covarianza strettamente singolare. Pre-registrato a **1** (= ciò che fa il training deep) |
| 5 | Confrontare col deep solo le metriche **threshold-free** | il floor usa il fallback a quantile, il deep la regola per-τ del paper |
| 6 | Non chiamare mai un arm `local` / `centralized` | collide con `summarize_converged.py:64` — match letterale su `local`+`centralized`, **dimostrato**: assorbe le righe floor senza avvisare. ⚠️ La parte «`fa_*` collide con `aggregate_all.py:30`» era **falsa**: quella stringa non c'e' in quel file |
| 7 | Mai >4 worker mentre `federated_eval.py` è vivo | il muro è la CPU a 16 core. `floor_eval` si auto-limita e va a `nice 19` |
| 8 | Non riportare un singolo numero come "il floor" | è una banda per testa, e va con l'inviluppo accanto |
| 9 | `paper_top1/3/5` su wsd è **sanità, non ranking** | mediana 1.00 per **ogni testa di decisione** — l'unico arm che si distingue e' `floor_random` (0.00), che e' in scope (batch D). Su UCR invece discrimina (0.283 mean / 0.000 median) ed è la metrica headline della letteratura UCR |

---

## 6. Cosa è cambiato il 2026-07-30 — dodici bug, tutti verificati su disco

I primi sei vengono da un audit del codice; i numeri 7-8 li ha trovati lo **smoke run**; i 9-12 una
storm di 21 agenti che ha auditato un modello di baseline ciascuno leggendo il codice vero. Nessuno
dei tre livelli avrebbe trovato da solo quello che hanno trovato gli altri due.

| # | difetto | effetto | fix |
|---|---|---|---|
| 1 | `floor_table.py` scriveva in `artifacts/floor/` **senza `mkdir`** | la directory era stata archiviata il 29-07 ⇒ `FileNotFoundError` al primo run, che si legge come input mancante | `out_dir.mkdir(parents=True)` + `--out-dir` |
| 2 | `deep_reports()` costruiva il glob da `f"artifacts/{tree}/ckpt/..."` usando la **chiave** del dict invece del path | cercava `artifacts/converge60/ckpt/…` (archiviato) e `artifacts/runs/ckpt/…` (il layout è `runs/<tag>/ckpt/`): **ogni** pattern mancava, e le colonne estese (top-K, `event_*`, `best_f1`, `vus_roc`, delay) uscivano vuote — indistinguibili da "l'arm non le ha mai scritte" | `REPORT_GLOBS` separato, profondità globbata, dataset filtrato su `report.json["dataset_name"]` (due dei tre alberi archiviati non hanno affatto il livello dataset nel path) |
| 3 | `floor_table.py` leggeva **un solo** path cablato | `launch.sh --engine floor` scrive in `artifacts/runs/<tag>/floor/`, uno per tag: la tabella non li vedeva | `--records-dir` con lista + glob, e la deduplica di `floor_eval` applicata fra i file. È lo step di merge |
| 4 | `floor_stats.py` calcolava il margine di potenza a `n = len(base)`, non all'n **appaiato** | margine riportato **più piccolo** del necessario ⇒ un TOST sotto-potenziato sembrava adeguato. E gli arm con <5 chiavi comuni sparivano **in silenzio**, indistinguibili da arm non girati | `m80` per arm al suo n appaiato, blocco `SKIPPED` esplicito, flag `UNDERPOWERED` sul TOST |
| 5 | l'**inviluppo non esisteva**: `DECISION_HEADS` definito e letto da nessuno | `0.4993 / 0.5585` nel `.tex` era calcolato fuori dal repo ⇒ irriproducibile (§8.9) | `scripts/floor_envelope.py`, con i quattro rifiuti di §4 |
| 6 | `witness_verdict()` non asseriva `_pooled_path` | il campo era registrato ma non controllato: sopra `floor_heads.CHUNK` righe `pooled_solve` ripiega su `chunked_pooled`, che **somma le stesse statistiche** che somma `fed_exact` ⇒ la circolarità che §0.4 esiste per vietare, travestita da 1e-13 invece di uno zero esatto | il path ora è un gate: `FAIL(circular: pooled path is 'chunked_pooled', not independent)` |
| **7** | 🔴 **`__w<W>` rendeva `ucr_split_w2p` non aggregabile.** La finestra è nel nome dell'arm per impedire la collisione fra finestre (§8.10) — giusto. Ma su `w2p` la finestra è una proprietà **della serie**: **79 valori distinti** su 180 cluster ⇒ **395 nomi di arm per 5 modelli concettuali**, ognuno con ~5 entità | la tabella avrebbe avuto 395 righe invece di 5, e **nessun arm** con un n usabile per un test appaiato: tutto il lato UCR del paper era non aggregabile. Nessuna review lo aveva visto | separazione **`_arm` (identità di storage, porta la finestra) / `_model` (identità di analisi, non la porta)**. `floor_table` aggrega per modello, `floor_stats` appaia per modello, e il **range di finestre viene dichiarato** — su `w2p` il trattamento non è di taglia costante fra cluster, e va detto in didascalia |
| **9** | 🔴 **`floor_env4` non arrivava a valle.** `floor_envelope` faceva `row = dict(src)` e non sovrascriveva `_model`, quindi ogni riga di inviluppo portava il `_model` della **testa vincente** | le righe venivano assorbite dentro le teste vincenti come finti seed in più: **nessuna riga `floor_env4`** nel CSV né nelle stats. `0.4993/0.5585` restava irriproducibile — esattamente la violazione di §8.9 che lo script esisteva per chiudere. E falliva **in silenzio**: il JSONL veniva scritto e niente dava errore | `_model: ARM` nel blocco di update |
| **10** | 🔴 **`floor_stats` includeva gli alberi pre-coorte per DEFAULT.** Il filtro era `trust == "stale"`, ma `floor_table.TRUST` emette `ok` / `archived` / `archived-stale` — mai la stringa nuda `"stale"` | predicato insoddisfacibile ⇒ `--keep-stale` era un no-op e **434 righe deep senza fingerprint** entravano nelle statistiche appaiate senza che nulla lo dicesse | il test è su `!= "ok"`, e le righe escluse vengono **contate e stampate** |
| **11** | 🔴 **La colonna `Δmed` non era appaiata.** Stampava `median(a) − median(b)` (differenza di mediane **marginali**) sulla stessa riga di un `p` di Wilcoxon appaiato e di un CI bootstrap di `median(a−b)`. Le mediane non sono additive sotto appaiamento | l'effetto dell'impulse era sovrastimato **3,1×** (+0.181 contro +0.058 appaiato) e la stima puntuale cadeva **fuori** dal CI stampato accanto. **Ogni** effect size di §0.6 va ri-derivato | `Δmed = median(a−b)`, più una colonna **`HL`** (Hodges-Lehmann), che è lo stimatore consistente col signed-rank |
| **12** | 🔴 **`_excess_objective` era un quadratico grezzo senza riferimento.** 113/46/35/14 su c0..c3 | normalizzando per `J(w*)` (1813/4868/5811/405) diventa 6,25%/0,94%/0,60%/3,52% e **l'ordinamento fra cluster si ribalta**. Peggio: l'eccesso del braccio **`local`** sullo stesso obiettivo è 490,6/…/89,8, cioè **4-6× PIÙ GRANDE** di quello di FedAvg ⇒ la frase «l'obiettivo peggiora in modo misurabile mentre la metrica non si muove» si legge come *FedAvg fa danno*, quando su quell'obiettivo FedAvg è molto **più vicino** all'ottimo del braccio `local` da cui è dichiarato indistinguibile | aggiunti `objective_star()` e i campi `_objective_star`, `_excess_objective_rel`, `_excess_objective_local`, `_excess_objective_local_rel`: la frase non può più essere scritta senza i suoi riferimenti |
| **8** | 🔴 **`floor_stats` appaiava contro la testa SBAGLIATA su `w2p`.** Il riferimento era cercato con un match esatto su `floor_ma_c_k10`; lì l'arm si chiama `floor_ma_c_k10__w408`, quindi il match falliva e il codice ripiegava su `floor_keys[0]` = il primo in ordine alfabetico | **`floor_ar_p32_lam0.0001__w408`**: ogni Δ e ogni p su `ucr_split_w2p` erano calcolati contro la testa ridge-AR invece della media mobile pre-registrata, annunciato da **una sola riga di header** | l'appaiamento per modello toglie il suffisso, e il fallback arbitrario ora è **fatale**: `floor_stats` si rifiuta di girare senza il riferimento pre-registrato, o con un `--floor-arm` esplicito |

Due difetti minori chiusi insieme: `--fit-stride` e `--suite` non erano nell'identità
dell'arm (il primo ora tagga `__fs<N>`; il secondo **non** deve taggare, perché aggiunge
metriche senza cambiarne nessuna — ora è `dedupe()` a proteggere il blocco `_suite` da una
passata `--suite off` successiva); e `--force` è stato tolto dalle batch A e F.

⚠️ **I nomi degli arm pre-registrati sono invariati** — verificato: `floor_ma_c_k10`,
`floor_ar_p32_lam0.0001__central`, `floor_pca_K8_gamma0.05__fed_naive`,
`floor_ma_c_k10__noimpulse`. Solo uno stride di fit ≠ 1 aggiunge un suffisso.

---

## 7. Costi misurati

| cosa | throughput | note |
|---|---|---|
| `ma_c`, suite off | ~17 righe/min (J misto) | 1130 righe ucr in 67 min |
| teste fittate (`ar`/`pca`/`gauss`), suite off | **15,4 righe/min a J=12** | 150 righe in 585 s |
| `--suite on` | **~30 s/client** (misurato 2026-07-30 su c0 sotto contesa di CPU; il vecchio ~110 s/client non e' stato riprodotto) | il collo è VUS/PATE |
| PCA solve, W=3028 | 1,78 s stats + 4,61 s solve | Σ = 73 MB, buffer chunk 484 MB **per worker** |

🔴 **`--jobs` parallelizza fra CLUSTER, non fra client.** Un job è `(cluster, testa, modo)` e
dentro il job i client sono scorati in sequenza, perché il parametro va fittato una volta per
coorte. Quindi il wall-clock di una batch è dettato dal **cluster più grande**, non da
`totale/J`: su `wsd_fed` c1 ha 11 client, quindi ogni batch `--suite on` costa ~11×30 s ≈
**5-6 min** qualunque sia `FLOOR_JOBS`, e alzare J oltre il numero di job (4 cluster su wsd)
non serve a niente. Le batch a `--suite off` con molte teste hanno invece molti job e scalano
davvero con J.

Righe di un run: `#cluster × #client_per_cluster × #arm`, con `#arm = #teste × #modi` meno
le celle saltate (le teste a zero parametri sono arm-invarianti, quindi ogni modo federato
viene saltato **stampando il motivo**).

**Il totale dello scope del paper: ~4-6 h di CPU, zero GPU, resumable.**
wsd_fed 434 righe (di cui 62 con la suite) + ucr_split_w2p 4500 righe.

---

## 8. Lanciare una voce fuori scope, comunque

È supportato e a volte giusto (un referee che chiede FedProx su dati). Basta chiamare
`floor_eval` a mano; il banner ricorda che la riga non è nel set riportato:

    $PY scripts/floor_eval.py --dataset wsd_fed --heads ar --modes fed_prox \
        --mu 0.1 --rounds 30 --suite off --out-dir artifacts/floor_ablation
    # [floor] ⚠ OUT OF PAPER SCOPE: 'fed_prox' — ...

Usa **sempre un `--out-dir` separato** per le ablazioni: così `floor_table` con
`--records-dir artifacts/floor` non le tira dentro per sbaglio, e puoi includerle
esplicitamente quando vuoi.

---

## 9. Output

    artifacts/floor/records_<ds>.jsonl     # 1 riga per (arm, entity), 21 metriche + 30 campi (31 con --suite on)
    artifacts/floor/<ds>__<arm>.json       # {meta, summary, records}
    artifacts/floor/_derived/records_<ds>.jsonl   # SOLO l'arm sintetico floor_env4
    artifacts/floor/<ds>_all_models.csv    # tidy: floor + inviluppo + ogni albero deep
    artifacts/floor/<ds>_matrix.json       # client × modello × metrica
    logs/floor/<batch>.log

Nel CSV le colonne di identità sono **cinque**, e la distinzione conta:

| colonna | cos'è |
|---|---|
| `model` | l'identità di **analisi**. Raggruppa e appaia **sempre** su questa |
| `arm` | l'identità di **storage**: porta `__w<W>`, quindi su `w2p` un modello ha molti arm |
| `window` | la finestra di quella riga; `floor_table` stampa il range per modello |
| `tree` | `floor` (baseline + inviluppo) oppure l'albero deep |
| `trust` | `ok` solo per `floor` e `runs`. `archived` = **nessun fingerprint** ⇒ non appaiato |
