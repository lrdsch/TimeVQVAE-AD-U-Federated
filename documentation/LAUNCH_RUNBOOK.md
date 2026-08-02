# LAUNCH RUNBOOK — l'unico file per lanciare

Dal **2026-07-29** ogni lancio passa da qui. Tutto ciò che c'era prima è in
`artifacts/_archive_20260729/` (vedi il README lì dentro) e **non è confrontabile con i run
nuovi**: albero, finestra, tolleranza e insieme di cluster non erano pinnati da nessuna parte.

    PY=/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10

---

## 1. L'idea in una riga

> **Una coorte pinna QUALI dati copre un run. Un lancio nomina una coorte. Due run con la
> stessa coorte sono appaiati per costruzione, e un'ablazione è la stessa coorte con manopole
> diverse.**

Ogni confronto che questo progetto ha dovuto ritirare è caduto su un asse non pinnato: albero
diverso, finestra diversa, tolleranza diversa, insieme di cluster diverso. La coorte li pinna
tutti insieme, in un file, con un fingerprint.

    cohorts/<nome>.json     <- QUALI dati (dataset, cluster, finestra, tolleranza, seed)
    artifacts/runs/<tag>/   <- UN run su quella coorte
    artifacts/runs/<tag>/RUN.json   <- coorte + fingerprint + tutte le manopole

---

## 2. I dataset (10, di cui 8 federati)

    $PY scripts/cohort.py datasets

| dataset | cluster | entità | finestra | tol | cos'è |
|---|---|---|---|---|---|
| `wsd_fed` | 4 | 31 | 128 | **14** | **reale**, KPI WSD |
| `ucr_split` | 226 | 1130 | 128 | 64 | UCR quantity-skew, finestra costante |
| `ucr_split_w2p` | 180 | 900 | **per-serie** (48…3028) | 64 | idem, finestra `2×periodo` del paper |
| `toy_fed_uni` | 6 | 47 | 128 | 64 | sintetico principale |
| `toy_fed_uni_scarce` | 6 | 47 | 128 | 64 | asse scarsità |
| `toy_fed_uni_scarcer` | 6 | 47 | 128 | 64 | asse scarsità, estremo |
| `toy_fed_uni_ucrlike` | 6 | 47 | 128 | 64 | lunghezze campionate da UCR |
| `toy_fed_uni_wsdlike` | 4 | 31 | 128 | **14** | gemello sintetico di wsd |
| `ucr_ad` | 1 | 248 | 128 | 64 | ⚠️ pool di pretraining, **non federato** |
| `ucr_pool` | 1 | 248 | 128 | 64 | ⚠️ idem (stesse 248 serie di `ucr_ad`, build diverso) |

`--datasets all` prende **gli 8 federati** ed esclude i due pool: hanno un cluster solo, quindi
`local` vs `federated` non ha significato lì.

⚠️ `ucr_split` e `ucr_split_w2p` hanno **177 cluster in comune, byte-identici**; 49 esistono
solo a 128 e 3 solo a 2P. Un test appaiato fra i due gira **solo** sull'intersezione —
[`UCR_SPLIT_PAIRING.md`](UCR_SPLIT_PAIRING.md).

---

## 3. Creare una coorte

    # tutto: 8 dataset, 438 cluster
    $PY scripts/cohort.py new full --datasets all

    # i primi N cluster di ogni dataset (probe economico)
    $PY scripts/cohort.py new probe --datasets wsd_fed,ucr_split_w2p --clusters 4

    # serie/cluster SCELTI a mano
    $PY scripts/cohort.py new ucr10 --datasets ucr_split_w2p \
        --clusters ucr_001,ucr_002,ucr_003,ucr_005,ucr_006

    # da file (una riga per cluster) — es. le 82 storiche
    $PY scripts/cohort.py new paper82 --datasets ucr_split \
        --clusters-file scripts/ucr_split_clusters.txt

    $PY scripts/cohort.py show full
    $PY scripts/cohort.py verify full      # ogni cluster risolve ancora?

⚠️ **Su `ucr_split*` un cluster È una serie UCR.** `--clusters` è quindi il selettore di serie.
Gli ID hanno buchi (226 serie su 250, manca `ucr_004` fra le prime dieci): un ID inesistente fa
morire subito il comando con la lista dei validi, non viene saltato in silenzio.

---

## 4. Lanciare

### Deep (GPU)

    bash scripts/launch.sh --cohort full --arms local,centralized --tag main_v1

    # prima sempre, costa zero:
    bash scripts/launch.sh --cohort full --arms local,centralized --tag main_v1 --dry

Manopole via ambiente: `SLOTS_PER_GPU` (7), `PROTOCOL` (converged), `S1_ROUNDS` (300),
**`S2_ROUNDS` (= `S1_ROUNDS`)**, `LOCAL_EPOCHS` (10), `PATIENCE` (6), `BATCH` (64).

`S1_ROUNDS`/`S2_ROUNDS` sono **tetti, non budget**: entrambi gli stage onorano
`--fed-patience-rounds`, quindi si sovra-provvisionano e ogni stage si ferma dove appiattisce.
Sovra-provvisionare è la scelta giusta — il ginocchio varia da r37 a oltre r89 fra cluster, e
un budget fisso taglia i cluster lenti (regola dura 2026-07-24).

**Arm disponibili** (22): `local`, `centralized`, `centralized_cap`, `federated`,
`federated_cb_only`, `federated_cb_only_ema`, `federated_cb_only_ema_norevive`,
`federated_enc`, `federated_enc_partial`, `federated_enc_neck`, `federated_fedavg_cb`,
`federated_fedavg_cb_sharedprior`, `federated_fedavgm`, `federated_fedsgd`, `federated_fedsgd_align`,
`federated_fedsgd_fedenc`, `federated_fedsgd_pooltok`, `federated_pooltok_centralprior`,
`federated_protoprior`, `federated_shared`, `federated_align`, `federated_anchor`.

La lista sopra è **storica e incompleta**: mancano 10 dei 32 nomi — `federated_fedavg_cb_only`,
il deprecato `federated_fedavg_whole` e gli **otto** arm del trio encoder (`federated_enc_fedavg`,
`_fedprox`, `_fedproto`, `_commoninit`, `_commoninit_cbshared`, `_commoninit_cblocal`,
`_fedavg_cblocal`, `_fedprox_cblocal`). Erano dati per «sette»: ricontati in `OTHER_ARMS` il
2026-07-30, sono otto. (Gli arm `federated_enc*` in totale sono **11**: i tre FedPer —
`federated_enc`, `_partial`, `_neck` — erano già nella lista storica.) La fonte unica sono
`PAPER_ARMS` + `OTHER_ARMS` in `pipeline/federated_eval.py`, **32 nomi**, e il launcher li
valida prima di lanciare:

    bash scripts/launch.sh --cohort probe --arms sbagliato --tag x --dry   # muore con la lista

`--arms paper` si espande da `PAPER_ARMS`, quindi il comando non può divergere dalla tabella
del paper. **Le otto righe da riportare, e i nove arm da NON riportare, sono la §5.**

✅ **Dal 2026-07-29 ogni arm federato onora `--protocol converged`** (tutte e 19 le chiamate a
`train_federated` passano `protocol=`, pinnato da `scripts/fed_regression_unittest.py`). Prima
14 non lo facevano: `select_on_val` restava OFF, `val_loss` era NaN, `--fed-patience-rounds`
era morto e il run bruciava 300 round × 10 epoche senza fermarsi. Se questa riga ricompare nel
log, quel difetto è tornato e **il job non è riportabile**:

    [fed] patience_rounds set but val is not finite — Convergence stop DISABLED for this run.

⚠️ La riga «il trio encoder si ottiene da `federated_enc` + i flag `--fedprox-*` /
`--fedproto-*`» che stava qui **era falsa**: `federated_enc` cabla `enc_fed_algo="fedavg"`
e non legge nessuno di quei flag. Vedi §5.

### Floor (CPU, zero GPU)

🔴 **Dal 2026-07-30 il floor si lancia da `scripts/run_floor.sh`, non da qui.** Quel runner
porta lo scope congelato del paper (dentro/fuori con il motivo voce per voce), fa il gate
dei selftest, calcola l'inviluppo e consolida. Vedi [`FLOOR_RUNBOOK.md`](FLOOR_RUNBOOK.md).

    bash scripts/run_floor.sh smoke     # sempre prima: ~10 min, ogni cella tenuta
    FLOOR_JOBS=12 bash scripts/run_floor.sh all

`launch.sh --engine floor` resta valido quando ti serve un **fingerprint di coorte** sulle
righe floor (l'unico criterio di confrontabilità, §6):

    FLOOR_JOBS=12 bash scripts/launch.sh --cohort full --engine floor \
        --heads ma_c,ma_causal,ar,pca --modes local --tag floor_v1 --extra "--suite off"

🔴 **`launch.sh` non passa mai `--suite`**, quindi eredita il default **`on`** di
`floor_eval`: la nested suite costa **~10×** il blocco flat (~110 s/client, il collo è
VUS/PATE) e non serve a nessun confronto — §3 di `FLOOR_BASELINE.md` restringe il paragone
col deep al solo blocco threshold-free. **Metti `--extra "--suite off"`** su ogni lancio a
coorte tranne i due arm che finiscono in tabella.

Il floor legge da sé la finestra per-serie quando il dataset la porta (`ucr_split_w2p`), e
i record di più tag si uniscono con

    $PY scripts/floor_table.py --dataset <ds> --records-dir "artifacts/runs/*/floor"

---

## 5. FedAvg su TimeVQVAE-AD — la tabella del paper

### 5.0 Lancia tutto (il comando corto)

    $PY scripts/fed_regression_unittest.py                               # 0. secondi — il codice è sano?
    $PY scripts/fed_enc_algo_unittest.py                                 #    (CPU, nessun dato)
    bash scripts/smoke_arms.sh                                           # 1. minuti  — tutto gira?
    bash scripts/launch.sh --cohort probe --arms paper --tag probe_v1     # 2. ore     — la pipeline regge?
    bash scripts/launch.sh --cohort paper --arms paper --tag paper_v1     # 3. giorni  — i numeri
    bash scripts/launch.sh --cohort encdrift --tag encdrift_v1 \
         --arms federated_enc_fedavg,federated_enc_fedprox \
         --extra "--fedprox-mu 0.1"                                      # 4. ~ore    — le frasi di §5.2
    bash scripts/launch.sh --cohort encdrift --tag fedproto_uniform_v1 \
         --arms federated_enc_fedproto,federated_enc_commoninit \
         --extra "--fedproto-weight 0.1 --fedproto-agg uniform"          # 5. ~ore    — il debito di §5.2

Il passo 4 **non produce righe di tabella**: produce le due frasi che rispondono a «avete
provato FedAvg sulla rete?» e «avete provato il fix standard del client drift?». Gira sulla
coorte `encdrift` (solo `wsd_fed`, 4 cluster, fingerprint `4170f026e3bf6dd7`) perché il
confronto è **contro il floor**, e il floor esiste lì — non servono i 428 cluster. Costo: 8 job.

🔴 **Il passo 5 non c'era, e FedProto non aveva NESSUN comando di lancio in questo runbook**
(`grep -n fedproto` dava due sole righe di tabella). Aggiunto il 2026-07-30, verificato in
`--dry`: 8 job sulla stessa coorte `encdrift`, stesso fingerprint del passo 4, quindi appaiato
con esso riga per riga.

**`--fedproto-agg uniform` è la scelta obbligata, e il motivo è l'opposto di quello che questo
runbook diceva prima.** `uniform` è ciò che calcola il codice ufficiale degli autori
(`proto_aggregation`, media uniforme sui client che possiedono la classe — §5.2 punto 2): è
la lettura **canonica** *e* l'unica **non degenere** qui. Con `--fedproto-agg count` il target
prototipo coincide con il codebook merged (rel err ~3e-8) e `federated.py` stampa da sé un
WARNING a setup: quell'arm non porta nessuna informazione cross-client oltre a
`federated_cb_only` + common init. Tutti i run FedProto convergiti che esistono su disco sono
`count`; **`uniform` non è mai stato misurato**, ed è questo passo a saldare il debito.

`--fedproto-weight 0.1` è il λ dei run esistenti, così l'unica cosa che cambia rispetto a loro
è l'aggregazione. ⚠️ Ma il confronto `uniform` vs `count` **non si fa contro l'archivio**: quei
run non hanno `cohort_fingerprint` e per la regola di §6 non sono confrontabili con niente.
Per il delta appaiato servono **due tag sulla stessa coorte** — questo passo, più il suo gemello
con `--fedproto-agg count` (§6, stessa coorte tag diverso). `federated_enc_commoninit` va nella stessa
lanciata perché è il **null del trio** — senza di esso «FedProto aiuta» non si distingue da
«un init condiviso aiuta» — e non legge nessuno dei flag `--fedproto-*` (λ=0 pinnato nel
dispatch), quindi il rifiuto dei flag non-posseduti di `federated_eval` non scatta: basta che
**almeno un** arm richiesto li onori.

🔴 **`--fedprox-mu 0.1` non è opzionale.** Il default della CLI è **0.01**, e a quel valore
`federated.py` stampa da sé un WARNING: in forma `loss` sotto μ=0,1 il termine prossimale è
probabilmente sotto la soglia di kill 1e-2. Tutte le misure citate in §5.2 (`prox_grad_ratio`
mediana 0.0550) sono a **μ=0,1, forma `loss`** — la forma del paper. Lanciarlo al default
riprodurrebbe esattamente l'artefatto «no-op del solver» la cui refutazione è il punto.
🟡 Quella mediana **si riproduce esatta, ma solo dall'archivio** — che il suo stesso README
vieta di citare: è uno dei motivi per cui il passo 4 va rilanciato sotto coorte prima di
stampare il numero (§5.2b).

⚠️ **Controllo dopo il lancio: `grep g_ratio`, ma IGNORA il round 0.** Il denominatore del
rapporto è il gradiente del task, che collassa ~1000× dentro il primo round: la soglia va letta
al round in cui l'arm viene riportato. Misurato in questo repo il 2026-07-30 su un run reale
(smoke, μ=0,1, `wsd_fed`): **round 0 → 4.2e-3 / 9.5e-3 / 1.15e-2** (cioè *sotto* la soglia),
**round 1 → 1.28e-1** (30× sopra). Un giudizio dato al round 0 dice il contrario del vero —
è esattamente l'errore che ha prodotto il verdetto «no-op» poi refutato.

⚠️ Al passo 3 la coorte è **`paper`**, non `full`: 428 cluster invece di 438, perché
`--max-window 1024` toglie i 10 `ucr_split_w2p` che non stanno in GPU ai default (§5.7).

💾 **Disco: `paper` ci sta.** 3424 job, **~110 GB** stimati contro 631 GB liberi (margine 5,7×).
Dal 2026-07-30 `launch.sh` stima il footprint prima di dispatchare, lo stampa accanto al
conteggio dei job, avvisa sotto 1,5× di margine e **rifiuta con exit 2** se non ci sta (un disco
pieno a metà di un run da giorni lascia un albero indistinguibile da uno i cui job sono falliti).
Un rifiuto non lascia tracce: la dir del tag viene rimossa se l'ha creata lui, e il `RUN.json` di
un tag preesistente viene ripristinato. `ALLOW_LOW_DISK=1` scavalca.

⚠️ **La stima è MISURATA, non derivata dai parametri.** Il conto ovvio — scalare col numero di
parametri dell'encoder, da 0,1 M a W=128 fino a 85,7 M a W=3028 (§5.7) — **sbaglia di ~30×**,
perché l'encoder non è ciò che riempie il disco. In una directory cluster-arm vera il costo per
entità è `stage2.ckpt` (il prior, 3,7 MB a W=128) contro `stage1.ckpt` (0,5 MB), più un solo
`_fed_resume.pt` (4,4 MB) per l'intera coorte — e il prior non cresce con la finestra come
l'encoder. Calibrazione su **277 directory reali** in `artifacts/_archive_20260729/`:

| albero | finestra | n dir | MB per cluster-arm | MB per entità |
|---|---|---|---|---|
| `ucrsplit` | 128 | 266 | 5,6 | 1,1 (senza stage2) |
| `converge60` wsd | 128 | — | 27,0 | 5,4 (con stage2 + resume) |
| `ucrsplit_w2p` | >1024 | 11 | 36,4 (max 55,3) | 7,3 |

Fra W=128 e finestra grande il salto misurato è **6×, non 59×**. I gradini nel codice prendono
il regime superiore a ogni finestra, quindi la stima sovra-, non sotto-approssima.

`--arms paper` si espande da `federated_eval.PAPER_ARMS`, quindi **il lancio non può
divergere dalla tabella** che il paper stampa. Aggiungi un arm lì e il comando lo prende.

Il launcher ora, da sé: alza il demone MPS se manca, **valida i nomi degli arm prima di
lanciare** (un refuso costava un caricamento dati per ognuno dei 438 job), legge i **seed
dalla coorte** invece di cablare `--seeds 0`, passa `--s2-rounds` (prima restava al default 2
mentre lo stage 1 ne faceva 300), e a fine run fa l'audit di convergenza **su entrambi gli
stage**.

Non salti il passo 0: sono secondi, girano su CPU senza dati, e coprono i quattro difetti che
in questo repo sono già costati un numero sbagliato o uno sweep buttato (§5.8).

### 5.1 Le otto righe, e perché

| # | arm | codebook | prior | ruolo |
|---|---|---|---|---|
| 1 | `local` | — | — | pavimento |
| 2 | `centralized` | *pooled* | *pooled* | skyline |
| 3 | `federated_cb_only_ema` | suff-stat **+ EMA server** | locale | **(A)** — l'unica variante federata che batte il floor |
| 4 | `federated_cb_only` | suff-stat, γ=0 | locale | ablazione di γ |
| 5 | `federated_fedavg_cb_only` | **FedAvg** | locale | **−(A)**, controllo appaiato |
| 6 | `federated_shared` | suff-stat | condiviso | **−(B)** |
| 7 | `federated_fedavg_cb_sharedprior` | **FedAvg** | condiviso | entrambe alla maniera ovvia |
| 8 | `federated` | suff-stat | **parziale** | il metodo |

Le righe 3–7 sono un **fattoriale (primitiva del dizionario × condivisione del prior)**, non
un elenco di ablazioni — i due effetti principali si leggono come effetti principali:

|  | codebook **suff-stat** | codebook **FedAvg** |
|---|---|---|
| **prior locale** | `federated_cb_only(_ema)` | `federated_fedavg_cb_only` |
| **prior condiviso** | `federated_shared` | `federated_fedavg_cb_sharedprior` |

🔴 **La riga (A) è `_ema`, non `cb_only`.** Contro `movavg10` (mediana 0.5054, zero
parametri): `cb_only_ema` **+0.137, p=3e-5**; `cb_only` **+0.016, p=0.027 ma non
distinguibile sul secondo estremo (p=0.209)**. [`FLOOR_BASELINE.md` §0.3](FLOOR_BASELINE.md)
dice esplicitamente che un arm che pareggia il floor «va riportato come tale, non come
promettente». `cb_only` resta in tabella come **ablazione di γ**, non come headline.

⚠️ E `cb_ema` vs `local` è un **null ritrattato** (Wilcoxon appaiato +0.0000, p=0.581). Il
claim onesto è *«federare il dizionario pareggia il training locale superando una baseline a
zero parametri che la media dei pesi non supera»* — non «la federazione vince».

⚠️ Non spiegare l'EMA con `floor_ema_shard`: è un errore di categoria ([§2.2](FLOOR_BASELINE.md)).

### 5.2 Algoritmi × superfici — dove FedProx / FedProto / FA possono davvero stare

FedProx, FedProto e la Federated Analytics **non sono righe accanto a FedAvg**: sono
*operatori* che si applicano a una superficie. La domanda giusta non è «lo mettiamo?» ma «su
quale delle quattro superfici è definito?». Matrice di applicabilità:

| | encoder (44,9 %) | codebook | decoder (55,1 %) | stage 2 (prior) |
|---|---|---|---|---|
| **FedAvg** | ✅ `federated_enc_fedavg` | ✅ `merge=fedavg` (strawman) | ⚪ implementabile, fuori scope | ✅ `federated_shared` |
| **FedProx** | ✅ `federated_enc_fedprox` | ⛔ **non definito** — il dizionario è aggiornato da EMA k-means sotto `no_grad` ([`vector_quantizer.py:168`](../model/vector_quantizer.py#L168)): non c'è solver locale da penalizzare | ⚪ come sopra | ⛔ **non implementato, e volutamente** — vedi sotto |
| **FedProto** | ✅ `federated_enc_fedproto` | 🔵 **degenere: È già la riga (A)** — Eq. 6 count-weighted ≡ merge Prop. 1, **esatto** (4.6e-6 col Laplace) | ⛔ nessuna struttura di classe | ⚪ `federated_protoprior` |
| **FA** | ⛔ per definizione (è apprendimento a gradiente) | 🔴 **etichetta refutata** — il merge è un M-step di Lloyd, non una query | ⛔ | ⛔ **non ben posta** — vedi sotto |

**Le tre celle che valgono una frase ciascuna, a costo zero di run:**

1. **FedProto Eq. 6 == il nostro merge del codebook.** Con i codeword come classi,
   l'aggregazione count-weighted p̄ = Σm/Σn *è* `e_j` della Prop. 1. Riprodotto il 2026-07-30 con
   `scripts/fed_enc_algo_unittest.py` test [2]: uguaglianza **esatta** contro il centroide pooled,
   **4.60e-06** contro il codebook con smoothing di Laplace (la differenza *è* lo smoothing).
   ⚠️ la cifra «2e-8» che girava nelle note non è riproducibile con questi test: non citarla.
   Tre derivazioni indipendenti — Lloyd distribuito
   (Dhillon & Modha 1999), l'update EMA del VQ-VAE (van den Oord 2017 App. A.1) e FedProto Eq. 6
   — convergono sullo **stesso operatore**. È una prova di canonicità, non un'invenzione: si
   cita, non si misura. ⚠️ Il rovescio da anticipare: «avete solo rinominato FedProto». Difesa
   precisa — in FedProto i prototipi restano un *target di loss* e non sostituiscono mai i
   parametri; qui l'aggregato **è** il parametro (viene ribroadcastato e usato come quantizzatore).
   Stessa aritmetica, ruolo diverso.
2. **La lettura count-weighted dell'Eq. 6 sull'encoder == il commitment loss.** Prendendo il
   codice degli autori per la *forma* (`update_weights_het`, per-campione) e l'Eq. 6 stampata
   per l'*aggregazione* (count-weighted), il termine si riduce a `mean_i‖z_i − e_{k(i)}‖²`,
   cioè **esattamente** `commitment_loss` (rel. err **6.71e-08**, riprodotto il 2026-07-30,
   test [8]). Aggiungerlo con λ alza solo `commitment_weight`. Una frase; l'arm non serve.

   🔴 **Correzione 2026-07-30 — qui c'era scritto il contrario, ed è la correzione più
   importante di questo giro.** La frase diceva «le nostre due deviazioni (forma
   per-classe-media, **aggregazione uniforme**) sono forzate»: dava cioè `uniform` per
   deviazione e `count` per paper-faithful. **È falso.** Il codice ufficiale
   (`yuetan031/FedProto`, `lib/utils.py`, `proto_aggregation` righe 151-170, e `agg_func`
   141-149) fa `agg_protos_label[label] = [proto / len(proto_list)]`: media **uniforme** sui
   client che possiedono la classe, senza pesi `|D_ij|`. Verificato alla fonte il 2026-07-30, e
   verificato numericamente che `_aggregate_prototypes(..., "uniform")` di `pipeline/federated.py`
   riproduce quella formula a rel err **0.0 esatto**. Conseguenze:
   - `--fedproto-agg uniform` (il default) **non è una deviazione**: è l'aggregazione degli
     autori, e la forma per-classe è l'Eq. 8 del testo. L'arm come parte è canonico su
     entrambi gli assi, ciascuno rispetto a una fonte primaria diversa. **Non siamo in
     trappola logica** — sono le due fonti a non concordare fra loro.
   - la combinazione che questo repo chiamava «null canonico di FedProto» (per-campione **+**
     count) non corrisponde **né** al paper **né** al codice: è un **ibrido costruito qui**,
     prendendo un asse da ciascuna fonte. Va ridescritta come *«la lettura count-weighted
     dell'Eq. 6 come stampata, che QUI degenera nel codebook merged (rel err ~3e-8 misurato)»*.
   - causa a monte: il repo aveva letto `update.py::update_weights_het` (citato in 3 file) e
     **mai** `lib/utils.py`. Prima del 2026-07-30
     `grep -rn "proto_aggregation" .` dava **zero** occorrenze; oggi ne dà una sola, il WARNING
     aggiunto quel giorno a `pipeline/federated.py:1469`. Tabella corretta in
     [`FED_ENCODER_ALGOS.md`](FED_ENCODER_ALGOS.md), sezione «Fidelity to the original papers».

   🔴 **DEBITO DI MISURA — la lettura canonica non è mai stata misurata.** L'unico numero
   FedProto mai portato a convergenza è `count`, cioè **la configurazione degenere**: su disco
   esiste solo `federated_enc_fedproto_lam0.1_count`
   (`artifacts/_archive_20260729/converge60/`, 10 cluster), e tutti e tre i runner di
   produzione lo cablano — `run_converge60.sh:55`, `run_ucrsplit.sh:68`,
   `run_ucrsplit_w2p.sh:89`, tutti `FEDPROTO_AGG="${FEDPROTO_AGG:-count}"`. Di `uniform` non
   esiste **nessun** run convergito. Finché non gira (comando al passo 5 della §5.0), ogni
   frase su «FedProto qui» è una frase sulla cella degenere, e la cella canonica è **non
   misurata**, non «misurata e uguale».
3. **L'unica query genuinamente analitica disponibile non è ben posta.** L'istogramma dei conteggi
   di token *è* FA canonica (Elkordy et al. 2023 §3.1.4), ma richiede un tokenizer comune a tutti
   i client — e qui il token agreement cross-client misura **0.0000**, con occupancy JS 0.686–0.693
   contro un soffitto ln2 = 0.6931 (supporto disgiunto). Non è un buco: è il risultato.
   🔴 **SENZA ARTIFACT (2026-07-30).** Lo 0.0000 è citato in 6 file (`mixture_eval.py:60,188`,
   `launch_all_fa.sh:8`, `DISPOSITION.md`, `RESEARCH_LEDGER.md` righe 22 e 115, e qui) ma
   **nessun record lo supporta**: `grep -rl "agreement" artifacts/` non trova niente in tutto
   il repo. Il ledger stesso avverte che «queste grandezze sono in attesa di ri-misura».
   Lo strumento c'è ed è CPU: `scripts/fed_codebook_autopsy.py` calcola `token_agreement` e
   `usage_js` (`:242`). **Rigirarlo prima di stampare il numero**; il difetto *strutturale*
   (encoder locali ⇒ alfabeti non allineati) resta vero comunque.

### 5.2b Cosa NON va in tabella, e perché

| arm | famiglia | perché fuori | dove finisce |
|---|---|---|---|
| `federated_enc_fedavg` | FedAvg | **p=0.992** contro il floor — indistinguibile da una media mobile. 🔴 **NON RIPRODUCIBILE (2026-07-30)**: `floor_table.py --dataset wsd_fed` trova **434 righe deep e ZERO righe floor** (`artifacts/floor/` non contiene nessun `records_wsd_fed.jsonl`) ed esce **RC=1**. Il p non è ricalcolabile finché il floor non viene rigenerato dentro una coorte (`scripts/run_floor.sh`, §4) | **una frase** (più forte di una riga) — ma il numero va rimisurato prima di stamparlo |
| `federated_enc_fedprox` | FedProx | ⚠️ **motivo corretto il 2026-07-30.** Non è un no-op: su `converge60`, μ=0,1 forma `loss`, `prox_grad_ratio` mediana **0.0550** (min 0.0412, max 0.1712) = 4× sopra la soglia di kill 1e-2. Il vecchio verdetto «no-op del solver» veniva da uno smoke a 3 round ed è **REFUTATO**. Fuori perché **non supera il floor lo stesso**, non perché sia inerte. 🟡 **RIPRODUCIBILE MA NON CITABILE**: il ricalcolo dà esattamente 0.0550 sui 4 cluster wsd, ma **solo** da `artifacts/_archive_20260729/`, e il README dell'archivio vieta di citarlo («va **rigenerato** dentro una coorte, non citato da questo archivio»). Estendendo agli stessi run toy la mediana su 10 cluster è 0.0429 | **stessa frase** di `enc_fedavg` — è la risposta a «avete provato il fix standard del client drift?» |
| `federated_enc_fedproto` | FedProto | il fatto pubblicabile è il **teorema** (la lettura count-weighted degenera nel commitment loss), non il numero. ⚠️ **NON** «serve un paragrafo per difendere le deviazioni dal canonico»: dal 2026-07-30 il default `uniform` risulta essere l'aggregazione del codice ufficiale, quindi non c'è deviazione da difendere — c'è da spiegare **quale** delle due fonti primarie si segue su quale asse (§5.2 punto 2) | frase 2 sopra |
| `federated_enc_commoninit` | null | è il controllo del trio: serve solo se il trio è in tabella | — |
| `federated_enc_partial` / `_neck` | FedPer | `partial` (42,7 %) ≈ `full` (44,9 %): non è uno sweep | — |
| `federated_enc_*_cblocal` | — | encoder federato + dizionario locale: cella diagnostica | — |
| `federated_protoprior` | FedProto su stage 2 | **TOY-ONLY**: toy +0.100 vs `cb_only`, ma su wsd **pareggia `local`** (n=60, 0.352, <MDE). Un risultato solo-toy in un workshop paper invita il rifiuto | — |
| **stage-2 FedProx** *(non implementato)* | FedProx | ⛔ **volutamente vuoto**: μ→0 è `federated_shared` (riga 6), μ→∞ è il prior locale (riga 3). È un'**interpolazione fra due righe che già abbiamo**, e lo slot dell'interpolatore è già occupato da `federated` (riga 8, personalizzazione parziale FedRep/FedPer). Implementarlo aggiunge codice per un punto limitato da max(riga 3, riga 6) | una frase |
| **decoder federato** *(non implementato)* | FedAvg | fuori per **scope dichiarato**; la riga 7 è già «entrambe le superfici alla maniera ovvia» | — |
| **suite `fa_*`** (**14** script — `ls scripts/fa_*.py \| wc -l`, ricontati il 2026-07-30; «13» era sbagliato) | FA | 🔴 **CONGELATA**: `mixture_eval` carica **un solo** stage-1, quello di `have[0]`, e tokenizza con esso i dati di ogni client (`:174` storico, oggi **`:204`** — `shared_stage1 = load_stage1(s1_ckpts[have[0]], …)`; `:174` oggi è una docstring). Nessun risultato `fa_*` è mai stato prodotto dopo il purge | — |
| `federated_fedsgd*` | FedSGD | τ swept 16×, inerte | una **clausola** nella frase su FedNova |
| `federated_fedavgm` | FedAvgM | momentum server sopra un prior che collassa | — |
| `federated_anchor` | — | degenere: `anchor_weight=λ` ≡ `commitment_weight+λ` | — |
| `federated_align` | FedMD/FedDF | il test del gauge — bello, ma è un secondo paper | — |
| `federated_fedavg_cb` | FedAvg | il prior federato in mezzo confonde il contrasto (A) | superato dalla riga 5 |
| **prior federato + dizionario locale** | — | ⛔ **impossibile**, non tagliato: `token_embedding` è indicizzato per code id (guard `SystemExit`) | vincolo strutturale, frase 5 |

Le frasi che valgono ciascuna una riga di tabella.

🔴 **Stato di riproducibilità dei tre numeri che compaiono qui sotto — verificato il
2026-07-30, e sono tre stati diversi.** Nessuno dei tre va stampato senza il suo stato:

| numero | dove | stato |
|---|---|---|
| `prox_grad_ratio` mediana **0.0550** | frase 1 | 🟡 **si riproduce esatto, ma solo dall'archivio.** `artifacts/_archive_20260729/converge60/**/federated_enc_fedprox_mu0.1/fed_history.json`, 4 cluster wsd, round ≥1 → min 0.0412 / mediana 0.0550 / max 0.1712. Il README dell'archivio **vieta** di citarne i numeri: «va rigenerato dentro una coorte». Rilanciare il passo 4 di §5.0 |
| **p=0.992** enc_fedavg vs floor | frase 1 | 🔴 **non si riproduce affatto.** `floor_table.py --dataset wsd_fed` trova 434 righe deep e **zero** righe floor, ed esce RC=1: in `artifacts/floor/` ci sono solo due file aggregati (`wsd_fed_all_models.csv`, `wsd_fed_matrix.json`), nessun `records_*.jsonl`. Rigenerare il floor con `scripts/run_floor.sh` (§4) prima di riusare il p |
| token agreement **0.0000** | frase 4 | 🔴 **nessun artifact in tutto il repo.** `grep -rl "agreement" artifacts/` non trova niente. Il numero circola in 6 file di testo e in zero record. Misurarlo con `scripts/fed_codebook_autopsy.py` (CPU, mai girato) |

> Sharing the encoder by FedAvg in addition to the codebook did not improve over the
> codebook-only variant; against a zero-parameter moving-average baseline it is
> indistinguishable (p=0.99). Adding a FedProx proximal term (μ=0.1) — verified active, median
> proximal-to-task gradient-norm ratio 0.055 over training — improves the Stage-1 objective but
> still does not clear that floor. The failure of weight-space federation here is therefore not
> client drift, and we keep the encoder client-local.

> The codebook merge is the count-weighted prototype aggregation of FedProto (Eq. 6) applied to
> codewords, which coincides exactly with the distributed Lloyd M-step of
> Dhillon & Modha (1999) and with the VQ-VAE EMA update of van den Oord et al. (2017) with the
> accumulation boundary moved from the minibatch to the round. Unlike FedProto, the aggregate
> replaces the parameter rather than serving as a loss target.

> Reading FedProto's Eq. 6 literally — count-weighted aggregation — the prototype penalty
> reduces exactly to the VQ commitment loss (rel. err 6.7e-8), so applying it to the encoder
> only rescales `commitment_weight`; we therefore do not report it as a distinct mechanism. We
> use the uniform aggregation of the authors' released implementation, under which the target
> is distinct from the merged codebook.

⚠️ La frase sopra diceva «*the canonical* FedProto penalty». Corretto il 2026-07-30: la
combinazione count + per-campione non è il canone di nessuna delle due fonti (§5.2 punto 2).
Scrivere «Eq. 6 read literally», mai «canonical».

> The one query on this system that is analytic in the sense of Elkordy et al. (2023) — the
> pooled token-count histogram — is not well posed: cross-client token agreement is 0.000 and
> codeword-occupancy Jensen-Shannon divergence is 0.686-0.693 against a ln2 = 0.6931 ceiling,
> i.e. the clients' induced alphabets have disjoint support. We therefore describe the codebook
> aggregation as non-gradient and exactly additive, and avoid the term "federated analytics".

> Federating the prior necessarily implies federating the codebook: the prior's token
> embedding is indexed by code id, so averaging it across clients with private dictionaries
> mixes unrelated symbols. Prior-only federation is not an available ablation.

> Objective inconsistency (Wang et al., 2020) does not affect Stage 1: the merged codebook
> is the exact pooled centroid regardless of per-client step counts. The prior is exposed;
> we control for it by holding local steps constant (τ-step regime) and the collapse
> persists.

🔴 **Vietato in titolo, abstract e proposizioni: «federated analytics».** La demarcazione
(Elkordy et al., APSIPA TSIP 2023 §2.1) è *«query che non richiederebbero ottimizzazione se
risolte centralmente»*; qui il task centrale è SGD su un VQ-VAE e il merge da solo è Lloyd, un
ottimizzatore con loop di convergenza (37–93 round). La difesa «passano solo statistiche
additive» è refutata **per nome** nella stessa sezione. Dire invece: *non-gradient, exactly
additive, sufficient statistic*. Vedi `documentation/RESEARCH_LEDGER.md` riga 27.

### 5.3 Cosa vuol dire «FedAvg» qui — tre oggetti diversi

| dove | funzione | arm | cos'è |
|---|---|---|---|
| encoder di Stage 1 | `_fedavg_encoder` | `federated_enc_fedavg` | FedAvg da manuale (McMahan 2017) |
| corpo del prior | `_fedavg_shared` | `federated`, `federated_shared` | FedAvg sul transformer, teste locali (split FedPer) |
| codebook | `merge="fedavg"` | righe 5 e 7 | **strawman deliberato**: il controllo contro il merge di Prop. 1 |

Perimetro: l'encoder è il **44,9 %** dei parametri di Stage 1 a W=128. `decoder_2d`,
`RefinementHead`, le running stats di BN, lo stato AdamW e la soglia per-entità **non
attraversano mai la rete**. Non esiste un arm che faccia FedAvg sul modello *intero*
(`_encoder_shared_keys` filtra su `k.startswith("encoder.")`) — ed è una **scelta di scope
da dichiarare**, non un buco: il claim è «basta federare il dizionario», e la riga 7 è già
«entrambe le superfici che il metodo federa, fatte alla maniera ovvia».

⚠️ **`federated_fedavg_whole` → `federated_fedavg_cb_sharedprior`** (2026-07-29). Il vecchio
nome si leggeva come «FedAvg su tutto il modello» e non lo era: lascia `fed_encoder="off"`,
quindi il 100 % della rete di Stage 1 resta locale. Funziona ancora ma stampa un banner di
deprecazione; **non metterlo in tabella**.

### 5.4 Cosa succede in un round, e cosa viaggia

Setup, prima del round 0: un `Stage1VQVAE` per client con `AdamW` costruito **una volta** e
tenuto vivo; il codebook di client 0 clonato e broadcastato (`collect_stats_only=True`,
`initialized=True` ⇒ **congelato** durante il training locale, k-means di seeding spento);
se l'encoder è federato, `_broadcast_encoder` dà a tutti lo stesso init — obbligatorio,
mediare conv net partite da init diversi è il problema di permutazione di FL.

Per round, client per client in sequenza:

1. `torch.manual_seed(_round_seed(seed, round, client))` — ordine dei batch deterministico.
2. `local_epochs` passate. Forward in fp16 (Turing), **matematica del VQ in fp32**. Il VQ
   accumula le statistiche grezze `n_j` (conteggio per codice) e `m_j` (somma dei latenti
   assegnati al codice j) contro il codebook congelato.
3. Assert: il codebook è bit-identico al broadcast.

Poi il server:

4. **Codebook**: `N = Σ_k n_k`, `M = Σ_k m_k`, `e_j = M_j / smoothed(N_j)` — M-step di
   k-means sul pool, esatto (Prop. 1). Con `merge="fedavg"` invece il dizionario si muove
   localmente via EMA e il server ne fa la media pesata: **è la riga di controllo**.
5. `enc_drift` misurato **prima** dell'aggregazione (dopo è 0 per costruzione).
6. **FedAvg**: `w̄ = Σ_k (n_k/Σn)·w_k`, `n_k` = finestre di training del client. Somma fp32
   su CPU, copia nel dtype di destinazione. Le running stats di BN sono **escluse** dalla
   media lineare (con `--fed-enc-bn shared` passano da `_pool_encoder_bn`, varianza totale).
7. Val in **fp32**, media **uniforme sui client** — non pesata: il cluster spedisce UN
   codebook e il silo più grosso non deve decidere quando la coorte ha convergito.
8. Best-on-val, pazienza, stop di convergenza.

Il messaggio, per round e per client:

    client -> server :  n_j (K=64 float), m_j (64×D float) [, 52 tensori encoder se federato]
    server -> client :  codebook e (K×D), (N, M) aggregate [, i tensori mediati]

Mai fuori dal client: finestre grezze, gradienti, decoder, running stats di BN, momenti
AdamW, loss scale, scaler e soglia per-entità.

### 5.5 Cosa guardare nel log

**Stage 1** (ogni arm federato):

    [fed] convergence stop ARMED: patience=6 rounds ... (needs select_on_val ...: ON)
    [fed:suffstat] round 2: loss=0.808 val=0.262 perplexity=10.3/64 dead=4.7% revived=3 drift=177.3 enc_drift=0.170

1. **`val=` deve esserci.** Se manca, `select_on_val` è OFF, la pazienza è morta e il run va a
   budget pieno. Lo smoke test lo controlla esplicitamente.
2. **`enc_drift` deve scendere** (solo arm con encoder federato). Misurato: toy
   1,116→0,803→0,200; wsd c0 3,554→0,849→0,170.
3. **`perplexity`** — ⚠️ **NON è «quanti codici usa un client».** Riscritto il 2026-07-30: qui
   c'era «a K=64 i client ne usano 10,3–33,5, cioè il 16–52 %», e quei due numeri **non sono
   tracciabili** — 10,3 veniva da un log `ucr_split` (round 13) e 33,5 da un log toy (round 11),
   due dataset diversi presentati come un range. Sui **15.618** valori di `perplexity` nei log
   il range reale è **2,1–54,4**.

   Cosa misura davvero: `_perplexity_from_counts(N)` ([`federated.py:259`](../pipeline/federated.py#L259),
   chiamata a [`:1803`](../pipeline/federated.py#L1803)) riceve `N = Σ_k n_k`, i conteggi
   **aggregati dal server**. È quindi una statistica **di coorte** — la diversità d'uso del
   dizionario sul pool dei client — non l'occupancy di un singolo client. In `merge='local'`
   è ancora peggio come lettura per-client: `N` è la somma delle usage di K dizionari
   *diversi*, e il codice lo dice già in un commento.

   🔴 **La lettura per-client non è recuperabile dai checkpoint.** Il broadcast a
   [`federated.py:1682`](../pipeline/federated.py#L1682)
   (`v.set_codebook(weight_s, ema_cluster_size=N_s, ema_embed_sum=M_s)`) sovrascrive lo stato
   EMA di **ogni** client con l'aggregato del server, quindi dopo il round nessun client porta
   più i propri `n_j`. Serve una misura diretta: `scripts/fed_codebook_autopsy.py` (esiste, è
   **CPU**, e **non è mai stato girato**) ricalcola l'occupancy per client su una probe
   condivisa. Finché non gira, l'argomento anti-k-FED nel paper non ha il numero che dichiara
   di avere: riportare il range di coorte 2,1–54,4 come tale, oppure girare l'autopsia.
4. **`[fed:<merge>]`** dice quale primitiva sta girando: `suffstat` (Prop. 1) o `fedavg` (il
   controllo). Se una riga della tabella mostra il merge sbagliato, l'arm non è quello che credi.

**Stage 2** — solo i tre arm a prior condiviso (`federated`, `federated_shared`,
`federated_fedavg_cb_sharedprior`); gli altri hanno il prior locale e non stampano round:

    [fed-s2] prior shared keys=55 (local heads kept: ('channel_embedding', 'output_bias'))
    [fed-s2] convergence stop ARMED: patience=6 rounds ... (needs select_on_val: ON)
    [fed-s2] round 1: prior_loss=4.0891 val=6.9292

5. **`prior shared keys=`** deve essere > 0 su questi tre e **0** su tutti gli altri. È il
   controllo che il prior sia davvero federato (o davvero locale) come l'arm dichiara.
6. **`[fed-s2] convergence stop ARMED`** deve comparire. Fino al 2026-07-29 lo stage 2 **non
   aveva pazienza**: qualunque `--s2-rounds` era il budget, speso per intero.

**In coda al run**, l'audit del launcher legge **entrambi** gli stage:

    TRUNCATED, NOT CONVERGED (stage2): M1_rotary/seed0/federated/fed_history.json
    1 truncated / 1 runs

Ogni riga così **non è riportabile** (regola dura 2026-07-24) — alza il tetto corrispondente.
Leggeva solo lo stage 1 fino al 2026-07-30, quindi diceva `0 truncated` su prior troncati.

### 5.6 Ablazioni: stessa coorte, tag diverso

`--codebook-size` **non è un arm**, è una manopola. Il modo supportato è §6:

    bash scripts/launch.sh --cohort probe --arms paper --tag paper_v1
    bash scripts/launch.sh --cohort probe --arms paper --tag abl_cb128 --extra "--codebook-size 128"

Per un paper corto K=64 vs 128 è **una frase con la perplexity misurata**, non una tabella —
ma la frase serve, perché a K più grande la frazione di dizionario che un client usa scende e
l'argomento anti-k-FED si indebolisce. ⚠️ «la frazione che un client usa» **non** si legge
dalla `perplexity` del log, che è una statistica di coorte sui conteggi aggregati dal server
(§5.5 punto 3): per quella frase serve `scripts/fed_codebook_autopsy.py`.

**L'ablazione sulla finestra è invece una coorte a sé**, non un `--extra`: W è un asse pinnato
dalla coorte, e i due valori vivono in due *build* diversi (`ucr_split` a W=128,
`ucr_split_w2p` a W=2·periodo). Il set di 10 serie scelto per quel confronto — quali, perché,
quanto costa e cosa non dimostra — è in **`documentation/UCR_WINDOW_ABLATION_SET.md`**.

### 5.7 Costo e capienza — usa la coorte `paper`, non `full`

    $PY scripts/cohort.py new paper --datasets all --max-window 1024 --overwrite
    bash scripts/launch.sh --cohort paper --arms paper --tag paper_v1

**428 cluster invece di 438**: `--max-window 1024` toglie i 10 cluster `ucr_split_w2p` che non
stanno in GPU ai default. Picco misurato su GPU, 5 client a batch 64:

| W | param/client | picco | max slot/GPU (48 GB) | cluster con W ≤ |
|---|---|---|---|---|
| 256 | 0,4 M | 0,41 GB | 118 | 91/180 |
| 512 | 1,5 M | 1,11 GB | 43 | 134/180 |
| **1024** | 5,9 M | **3,00 GB** | **16** | **170/180** |
| 1782 | 22,5 M | 6,87 GB | 6 | 177/180 |
| 3028 | 85,7 M | **16,79 GB** | **2** | 180/180 |

È una **scala, non una curva**: fra W=1400 e W=1782 i parametri fanno 6,9 M → 22,5 M perché
si aggiunge un blocco di downsampling. E le serie grandi sono **contigue nell'ordine di
dispatch** (`ucr_215`…`ucr_221` tutte ≥1702, poi `ucr_239/240/241` ≥2992), quindi con 14 slot
partono insieme: tre da 16,79 GB su una scheda da 48 sono un OOM, non un rischio.

A W ≤ 1024 il default `SLOTS_PER_GPU=7` passa con **2,3× di margine** e non serve nessuno
scheduler. La ragione buona però non è la GPU: a W=3028 il modello è **85,7 M parametri contro
0,1 M a W=128**, e aggregare metriche di detection su modelli distanti tre ordini di grandezza
in capacità è difficile da interpretare comunque. L'esclusione è **registrata nella coorte con
un fingerprint**, non lasciata decidere all'hardware — una frase nel paper e hai chiuso:

> On `ucr_split_w2p` we cap the per-series window at 1024, retaining 170 of 180 series; the
> excluded ones reach 85.7M parameters against 0.1M at W=128, a capacity range over which
> aggregate detection metrics are not comparable.

Se poi li vuoi, girano come tag separato sulla stessa ricetta: `SLOTS_PER_GPU=2` e una coorte
`w2pbig`. Ma per il workshop non servono.

⚠️ La frazione federata **non è costante** su w2p: 15 tensori / 59,8 % a W=46, 52 / 44,9 % a
W=128, 126 / 40,1 % a W=512. Il cap non lo risolve (il 59,8 % è all'estremo piccolo, che
resta): va riportato, e l'unità appaiata resta il cluster.

### 5.8 Smoke test

    bash scripts/smoke_arms.sh              # 13 casi su un cluster toy: gli 8 arm del paper
                                            # + il trio encoder (fedavg, fedprox, fedproto in
                                            # ENTRAMBE le letture) + il null commoninit
    bash scripts/smoke_arms.sh --full       # + wsd_fed (reale) e ucr_split_w2p (W per serie)
    ARMS=federated_fedavg_cb_only bash scripts/smoke_arms.sh    # un arm solo

Tre controlli per arm, uno per ciascun modo in cui questo repo si è già rotto: **exit 0**
(un arm che solleva), **json presente** (un arm che allena e non salva), **`val=` finito nel
log di stage 1** (`--protocol` non plumbato ⇒ pazienza morta ⇒ 300 round senza stop).

Gira sotto `TVQ_SMOKE=1`, che collassa i budget del loop converged a 40 step: il percorso
esercitato è quello vero, i numeri prodotti **non lo sono** e ogni run lo stampa da sé.

Test unitari, **CPU, secondi, nessun dato** — lanciali prima di ogni sweep:

    $PY scripts/fed_regression_unittest.py   # 40 check in 7 gruppi: i difetti del 2026-07-29/30, pinnati
    $PY scripts/fed_enc_algo_unittest.py     # 54 check in 8 gruppi: chiavi BN, prox, prototipi

⚠️ I due conteggi erano **25** e **37**: sbagliati, e ora ricontati eseguendo gli script il
2026-07-30 (`grep -cE '^\s+PASS\s'`). Se li ritocchi, ricontali — non stimarli.

`fed_regression_unittest.py` è la rete di sicurezza sui difetti che sono già costati un numero
sbagliato o uno sweep buttato, e che sono facili da ri-rompere:

| gruppo | cosa impedisce |
|---|---|
| matematica FedAvg | media non-`n_k`-pesata, client non identici dopo l'aggregazione, decoder toccato, `running_var` nella media lineare |
| schema dei seed | il ritorno di `(seed 7, r 10) ≡ (seed 8, r 0)` — 0 collisioni su 640k combinazioni |
| nomi liberi | il `NameError` di `federated_stage2` (AST su tutte e 3 le funzioni di orchestrazione) |
| plumbing `--protocol` | una chiamata a `train_federated` che dimentica `protocol=` — 19/19 |
| registro arm | registro e dispatch che divergono, e **il fattoriale 2×2 incompleto** (fallisce se togli una cella) |

Verifica minima del launcher stesso, se ne hai toccato il codice — esercita MPS, validazione
arm, slot, reap e audit in pochi minuti su un cluster solo:

    $PY scripts/cohort.py new _lt --datasets toy_fed_uni --clusters M1_rotary --overwrite
    TVQ_SMOKE=1 S1_ROUNDS=2 S2_ROUNDS=2 LOCAL_EPOCHS=1 PATIENCE=2 SLOTS_PER_GPU=1 \
      bash scripts/launch.sh --cohort _lt --arms federated --tag _launchtest
    rm -rf artifacts/runs/_launchtest logs/runs/_launchtest cohorts/_lt.json

---

## 6. Ablazioni — **stessa coorte, tag diverso**

È l'unico modo supportato, ed è quello che rende il delta appaiato:

    bash scripts/launch.sh --cohort full --arms local --tag main_v1
    bash scripts/launch.sh --cohort full --arms local --tag abl_cb128 --extra "--codebook-size 128"
    bash scripts/launch.sh --cohort full --arms local --tag abl_w2p   --extra "--width-base 32"

    # floor
    bash scripts/launch.sh --cohort full --engine floor --heads ma_c --tag floor_k10
    bash scripts/launch.sh --cohort full --engine floor --heads ma_c --tag floor_k20 --extra "--k 20"

Verificato end-to-end: due tag sulla stessa coorte danno **le stesse identiche entità**, quindi
il delta è appaiato riga per riga.

    $PY -c "
    import json
    for t in ('main_v1','abl_k20'):
        r=json.load(open(f'artifacts/runs/{t}/RUN.json'))
        print(t, r['cohort'], r['cohort_fingerprint'], r['extra_flags'])"

**Regola:** se i `cohort_fingerprint` coincidono, i run sono confrontabili. Se non coincidono,
**non lo sono** — e nessuna quantità di post-processing lo sistema.

---

## 7. Leggere i risultati

    artifacts/runs/<tag>/RUN.json            # coorte, fingerprint, arm, protocollo, extra
    artifacts/runs/<tag>/<dataset>/<cluster>__<arm>.json
    artifacts/runs/<tag>/ckpt/               # checkpoint deep
    artifacts/runs/<tag>/floor/              # record floor
    logs/runs/<tag>/_orchestrator.log        # START/DONE/FAIL per job

    $PY scripts/floor_table.py --dataset wsd_fed      # legge artifacts/runs/ + l'archivio
    $PY scripts/floor_stats.py --dataset wsd_fed --metric vus_pr

Il launcher fa in automatico l'**audit di convergenza** a fine run, su **entrambi** gli stage:
ogni riga marcata `TRUNCATED, NOT CONVERGED (stage1|stage2)` **non è riportabile** (regola dura
2026-07-24). Il suffisso dice quale tetto alzare — `S1_ROUNDS` o `S2_ROUNDS`.

`RUN.json` registra ora anche `s2_rounds` e `seeds`, così un tag è ricostruibile per intero
senza rileggere il comando dalla shell history.

---

## 8. Stato — cosa è stato sistemato, cosa resta

### Sistemato il 2026-07-29/30 (verificato)

| # | era | fix |
|---|---|---|
| 1 | `pipeline/federated.py` — l'allarme di troncamento di `federated_stage2` stampava `rounds_done`, che **non esiste** in quella funzione: `NameError` a fine training, proprio nel caso che l'allarme deve segnalare | usa `best_round` (stage 2 non ha resume, l'indice è già assoluto) |
| 2 | **14 arm federati ignoravano `--protocol`** ⇒ `select_on_val` OFF ⇒ `--fed-patience-rounds` morto e nessun best-on-val: 300×10 epoche senza fermarsi | `protocol=args.protocol` su **tutte e 19** le chiamate a `train_federated`; verificato una per una |
| 3 | Il contrasto (A) non era mai stato misurato pulito: ogni via a un codebook mediato federava anche il prior | nuovo arm **`federated_fedavg_cb_only`** — gemello appaiato di `cb_only`, cambia **solo** la primitiva di merge |
| 4 | Repliche di seed non indipendenti: `seed*1000+round*100+client` faceva coincidere (seed 7, r 10) con (seed 8, r 0) | `_round_seed()` con hash blake2b — **0 collisioni su 640k** combinazioni testate, deterministico e portabile |
| 5 | `federated_fedavg_whole` si leggeva come «FedAvg su tutto il modello» e non lo era | rinominato **`federated_fedavg_cb_sharedprior`**; vecchio nome = alias con banner di deprecazione |
| 6 | Nessuna validazione dei nomi arm: un refuso costava un caricamento dati per ognuno dei 438 job | registro `PAPER_ARMS`/`OTHER_ARMS`, validato in `launch.sh` **prima** di lanciare e in `federated_eval` prima dei loader |
| 7 | `launch.sh` cablava `--seeds 0` ignorando i seed della coorte | li legge dalla coorte e li passa (`federated_eval` li mette in pool in un json solo) |
| 8 | `launch.sh` esportava la pipe dir MPS ma non alzava né verificava il demone | alza il demone se manca e lo scrive nell'orchestrator log |
| 9 | Nessun modo di smoke-testare gli arm senza pagare ore di loop converged | `scripts/smoke_arms.sh` + `TVQ_SMOKE=1` (budget a 40 step, banner «non riportabile») |
| 10 | **`launch.sh` non passava `--s2-rounds`**: restava al default argparse **2**, quindi i tre arm con prior condiviso allenavano quel corpo per 2 round mentre lo stage 1 ne faceva 300 — e `RUN.json` non lo registrava | knob `S2_ROUNDS` (default = `S1_ROUNDS`), passato e scritto in `RUN.json` |
| 11 | **Lo stage 2 non aveva pazienza**: qualunque `--s2-rounds` *era* il budget, nessun early stop, contro la regola dura «train to convergence» | `federated_stage2(patience_rounds=...)`, stesso contratto dello stage 1 ⇒ `--s2-rounds` diventa un **tetto** sovra-provvisionabile |
| 12 | L'audit di convergenza leggeva **solo lo stage 1**: un run col prior troncato veniva riportato «0 truncated», e i tre arm a prior condiviso sono proprio quelli il cui stage 2 può troncare | legge entrambi gli stage; verificato su un run reale che diceva `0/1` e ora dice `1/1 (stage2)` |

**Aggiunti il 2026-07-30, verificando davvero FedProx/FedProto/FA** (§5.2) — i primi tre sono
tutti della stessa specie: *un artefatto che sembra riportabile e non lo è*.

| # | era | fix |
|---|---|---|
| 13 | **`RUN.json` non registrava `TVQ_SMOKE`.** Un run smoke scriveva un manifest con `cohort_fingerprint` valido e la nota «Comparable to any other tag with the same cohort_fingerprint» — mentre ogni suo numero è privo di senso (budget a 40 step). Il manifest mentiva proprio sulla proprietà che il sistema a coorte esiste per garantire | campo `"smoke": true/false`, e la nota diventa «comparable to NOTHING». Pinnato dal test `[G]` |
| 14 | **`--dry` coniava un `RUN.json`** per un run mai avvenuto (trovato: `artifacts/runs/encdrift_v1/RUN.json`, `s1_rounds: 300`, zero job). Un lettore non può distinguere un manifest orfano da un run i cui job sono tutti falliti | il manifest si scrive solo se `DRY -eq 0`; il dry stampa le joblines come prima |
| 15 | **`scripts/launch_all_fa.sh` non aveva nessun guard.** La suite FA è congelata dal 27/07, ma solo nel ledger: il `SystemExit` sta in `mixture_eval.py`, che è **un altro entry point**. Erano lanciabili 14 esperimenti da GPU-giorni che producono numeri già ritirati | rifiuta con `exit 2` stampando il motivo per intero; override esplicito `FA_I_KNOW_ITS_SHELVED=1` |
| 16 | **`smoke_arms.sh` non copriva `federated_enc_fedprox`** (promosso ad arm da lanciare lo stesso giorno), e lo avrebbe comunque girato a `--fedprox-mu 0.01`, cioè una configurazione che nessuno lancia | aggiunto, con extra per-arm che lo esegue a **μ=0.1**. In più `ARMS=<un arm>` ora è onorato davvero: gli arm da testo si appendono solo se il chiamante non ha scelto |

**Trovati dall'audit multi-agente del 2026-07-30, corretti e verificati** — questi **cambiano
i comandi**, non solo il testo: sono collisioni di percorso, flag che venivano ignorati in
silenzio e provenienze mentite. Stessa specie di prima: *un run che sembra riportabile e non lo è*.

| # | era | fix |
|---|---|---|
| 17 | **`launch.sh` non metteva il DATASET in `--out-dir`**, e i nomi dei cluster non sono unici fra dataset: sulla coorte `paper` (428 cluster) **177 nomi erano condivisi da 2–4 dataset**, quindi i checkpoint collidevano. Il caso peggiore è silenzioso e sbagliato — `ucr_split` a W=128 e `ucr_split_w2p` a W=408 scrivevano nello stesso `ucr_001`, cioè due modelli con capacità diverse di ordini di grandezza nello stesso file | `--out-dir` porta ora un segmento **dataset**: `artifacts/runs/<tag>/ckpt/<dataset>/…` |
| 18 | **`--extra` poteva sovrascrivere i flag che la coorte pinna.** `$EXTRA` è appeso **per ultimo** e argparse tiene l'ultima occorrenza, quindi `--extra "--metrics-tolerance 64"` sostituiva in silenzio la tolleranza della coorte mentre `RUN.json` continuava a dichiarare il `cohort_fingerprint` di quella coorte: il manifest mentiva sull'unica proprietà che rende due run confrontabili | `launch.sh` **rifiuta** con `exit 2` gli 8 flag pinnati — `--window-length --metrics-tolerance --seeds --dataset --cluster --clusters --out-dir --cohort` — nominando il flag e stampando come si crea la coorte nuova che quell'ablazione in realtà è (`--clusters` c'è perché il motore floor scrive lo stesso asse al plurale) |
| 19 | `launch.sh` non validava `--engine`, e `RUN.json` era costruito per **interpolazione shell** (un valore con una virgoletta produceva json invalido, o peggio json valido e sbagliato) | `--engine` validato; `RUN.json` serializzato con `json.dumps`; niente `RUN.json` in `--dry` |
| 20 | **Un flag del trio impostato su un arm che non lo legge veniva ignorato in silenzio** e finito comunque in `RUN.json`: `--arms federated_cb_only --extra "--fedprox-mu 0.1"` era accettato e registrato verbatim. **30 arm su 32** ignorano `--fedprox-mu`, e la provenienza dichiarava un trattamento mai applicato | `federated_eval` esce con errore se **nessun** arm richiesto legge il flag, e stampa quali arm lo onorerebbero (`FLAG_OWNER_ARMS`) |
| 21 | **`_resume_tmp` non portava né l'arm né il PID**: due arm dello stesso cluster lanciati a 3 s di distanza si scambiavano il bundle di resume. Riprodotto su disco | il nome del tmp contiene ora tag dell'arm e PID |
| 22 | Con `--fedproto-agg count` il target prototipo **è** il codebook merged (rel err ~3e-8), cioè l'arm non porta informazione cross-client oltre a `federated_cb_only` + common init — e nulla lo diceva a chi lanciava | `federated.py` stampa un WARNING a setup che nomina la degenerazione, la kill rule del 5 % e il fatto che la lettura non degenere è `agg='uniform'` |
| 23 | **`floor_table.py` non teneva il TAG nella chiave dei report deep**: con più tag ne sopravviveva **UNO**, scelto dall'ordine del filesystem, senza nessun avviso. Un'ablazione a due tag mostrava quindi una sola delle due colonne, e sembrava completa | il TAG entra nella chiave dei report deep |

### Resta aperto

| # | stato | dettaglio |
|---|---|---|
| 1 | 🟡 | ~~Il motore deep non è mai stato eseguito attraverso `launch.sh`.~~ **Girato per la prima volta il 2026-07-30** su una coorte a 1 cluster sotto `TVQ_SMOKE=1`, arm `federated`, rc=0: MPS, validazione arm, slot, reap e audit tutti esercitati, e l'allarme di troncamento dello stage 2 è scattato senza crashare. Resta 🟡 perché **non è mai girato a budget vero**: `--cohort probe --arms paper` è ancora il primo passo. |
| 2 | 🟢 | ~~Lo sweep vecchio su `artifacts/ucrsplit_w2p_cb64/` girava ancora all'archiviazione.~~ **Chiuso il 2026-07-30.** L'albero era già stato spostato il 29, ma **i log erano rimasti orfani in `logs/`** — separati dai record che descrivono. Ora sono in `artifacts/_archive_20260729/logs/`, insieme ad altri due orfani della stessa specie (`ucrsplit_w2p`, `ucrsplit_w2p_cb128`, del 27/07). `logs/` contiene ora solo `runs/`, `floor/` e `vram.csv`. Rimpiazzato da: `launch.sh --cohort paper --arms paper`. |
| 3 | 🟢 | ~~Non esiste un aggregatore che confronti **due tag** direttamente.~~ **Chiuso il 2026-07-30 per il floor**: `floor_table.py --records-dir "artifacts/runs/*/floor"` unisce ogni tag applicando la deduplica per `(arm, entity)` di `floor_eval`, e stampa in coda il `cohort_fingerprint` di ciascuno più un avviso se compaiono alberi pre-coorte. Il confronto fra due tag **deep** si fa ancora a mano. |
| 4 | 🟢 | ~~`merge="fedavg"` tocca solo lo stage 0 del Residual-VQ.~~ **Chiuso il 2026-07-30**: `federated_stage1` ora **rifiuta** `merge='fedavg'` con più di uno stage RVQ, prima del round loop. Correzione: `merge="local"` non era un bug — non aggrega niente, usa `c.vq` solo per la diagnostica. |
| 5 | 🟢 | ~~Nessuno scheduling consapevole della taglia.~~ **Chiuso il 2026-07-30 senza scheduler**: `cohort.py --max-window` e la coorte `paper` (428 cluster). A W ≤ 1024 il default `SLOTS_PER_GPU=7` ha 2,3× di margine — §5.7. |
| 6 | 🟡 | Lo stato AdamW **sopravvive all'aggregazione** (variante cross-silo, non McMahan). Effetto plausibilmente non trascurabile — uno step di Adam (≈lr=1e-3/coordinata) è dell'ordine dell'intera deriva del round (7,7e-4). Dichiararlo in Method; misurarlo solo se serve. |
| 7 | 🟡 | Encoder e decoder **non sono federabili insieme** (`_encoder_shared_keys` filtra su `encoder.`). È una scelta di scope, non un buco — ma va dichiarata (§5.3). |

---

## 9. Costo di un lancio completo

Coorte `full` = **438 cluster**. A 6 arm ⇒ 2628 job deep. Con il costo osservato
(~35 min/job, 14 slot) sono **~4,5 giorni**. Il floor sulla stessa coorte è ore, su CPU.

**Non lanciare `full` come prima cosa.** L'ordine sensato:

    1. probe   (2 cluster/dataset, 1 arm)   -> minuti,  valida la pipeline
    2. floor   (full, teste economiche)     -> ore,     dà la scala a tutto
    3. deep    (full o paper82, 6 arm)      -> giorni
    4. ablazioni sulla stessa coorte
