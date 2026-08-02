# FLOOR — baseline closed-form: sanity floor + federazione esatta

**Stato (2026-07-28): IMPLEMENTATO.** Il design di §3–§8 è ora codice in-repo:

| file | ruolo |
|---|---|
| `scripts/floor_heads.py` | le teste (`ma_c`, `ma_causal`, `diff1`, `random`, `ar`, `pca`, `gauss`) + le iterate `fedprox`/`localgd`. Pure numpy, **zero import dal repo**, `python scripts/floor_heads.py` = 17 invarianti asseriti |
| `scripts/floor_eval.py` | il framework: 11 modi di federazione, witness di esattezza, percorso `detect` fedele, **suite metrica completa**, pool `--jobs`, resume |
| `scripts/run_floor.sh` | la misura completa in ordine di dipendenza, resumable |
| `scripts/floor_table.py` | matrice client × modello × metrica (CSV + `<ds>_matrix.json`) contro ogni albero deep |
| `scripts/floor_demo.py` | walkthrough numerico su un client + demo di esattezza AR (indipendente per costruzione) |

**Cosa NON è implementato, per scelta dichiarata:** `knn`/`fed_coreset`/`fed_ensemble` (non additivi — la
linea di demarcazione va testata, ma è un lavoro a sé), `fed_dp`, `ema_shard` (§2.2 lo vieta come
spiegazione, quindi non serve), `--emit-fed-tree` (mai: regola di soglia diversa, §8.4),
**`fed_sharedbasis` / `fed_sharedbasis_avg`**.

🔴 **Correzione 2026-07-30.** L'ultima coppia era la lacuna più insidiosa del documento: §2.2 *descrive*
`floor_fed_sharedbasis` e `floor_fed_sharedbasis_avg` come le due celle mancanti di un 2×2
«(rappresentazione vs testa) × (esatto vs mediato)», e quel 2×2 **non esiste in codice** — `floor_eval.ALL_MODES`
non li contiene. Un lettore del design li dava per misurati. Realizzarli è un work package (nuovo percorso di
merge + prova di esattezza + invariante nel selftest), e il loro movente era coprire un arm deep `\pending` che
non arriva comunque sotto fingerprint: **tagliati** dallo scope del paper (`FLOOR_RUNBOOK.md` §1).

**Lo scope effettivamente riportato è un sottoinsieme di questo documento.** Il triage del 2026-07-30 vive in
[`FLOOR_RUNBOOK.md`](FLOOR_RUNBOOK.md) §1 (dentro/fuori, con il motivo voce per voce) e §6 (i sei bug chiusi in
`floor_table` / `floor_stats` / `floor_eval` e l'inviluppo, che prima non esisteva come codice).

> ⚠️ Il prototipo originale in `/tmp` **è andato perso** come previsto da §8.9: le righe `ar32_*`, `pca8_*`,
> `gauss` di §0.1, lo sweep di §0.1b e i residui di §0.4 nascevano lì. Tutte quelle celle sono state
> **riscritte da zero** in `floor_heads.py` e rimisurate; dove il numero nuovo differisce dal vecchio è
> segnalato, e il vecchio non va più citato.

**Perché esiste.** L'audit del 2026-07-15 elenca la baseline "moving average" come mancante ed
**esistenziale**: senza di essa nessun numero assoluto del progetto ha una scala. `federated_method.tex:281`
porta ancora una riga `\pending`. Questo documento la chiude e — più importante — aggiunge un arm che nessuna
baseline classica offre di norma: una **federazione esatta per costruzione**, che separa il beneficio-dati
dalla perdita-di-aggregazione senza confondimenti.

---

## 0. Evidenza misurata (verificata su disco)

Tutto su `wsd_fed`, **n=31 client matched**, `window=128`, `eval stride=13`, `metrics_tolerance=14` (da
`data/raw/wsd_fed/metadata.json`), impulse term **ON** (`config.py:298`), soglia = fallback quantile 0.99,
metrica headline `vus_pr`. Percorso: `data.load_scaled_records` → `detect._init_entity` →
`detect._assemble_rolling` → `detect._finalize_entities` → `detect._fit_threshold_paper` →
`detect._detection_metrics`. Fit **solo su train**, nessun leak di label (audit del codice: pulito).

### 0.1 Le teste FLOOR

**Tutte le teste, rimisurate 2026-07-28** (batch D di `run_floor.sh`, n=31, fit a stride 1, scoring a
stride 13). `Δ` e `p` sono Wilcoxon appaiato contro `ma_c` k=10:

| head | param. fittati | median `vus_pr` | mean | `top1` | Δ med vs `ma_c` | p |
|---|---|---|---|---|---|---|
| **`ma_c` k=10** — box centrato | **0** | **0.5054** | **0.4421** | 0.87 | — | — |
| `pca` K=8 — sottospazio di finestra | 128·8 | 0.4277 | 0.4228 | 0.87 | −0.078 | 0.75 |
| `ma_causal` k=10 | 0 | 0.4394 | 0.4170 | 0.90 | −0.066 | 0.37 |
| `ar` p=32, λ=1e-4 | 33 | 0.3961 | 0.3895 | 0.90 | −0.109 | **0.046** |
| `gauss` γ=0.05 | 128·129/2 | 0.3363 | 0.3848 | 0.87 | −0.169 | 0.37 |
| `diff1` (≡ `ma_c` k=3 riscalata) | 0 | 0.2547 | 0.2775 | 0.87 | −0.251 | 9.3e-09 |
| `random` | 0 | 0.0082 | 0.0111 | 0.06 | −0.497 | 9.3e-10 |

**Riproduzione del prototipo perduto:** `pca8` esce **identico** (0.4277 / 0.4228, quattro decimali) e
`random` pure (0.0082 / 0.0111 vs 0.008 / 0.011). `ar32` **no** (0.3961 vs 0.3597): il prototipo usava una
λ relativa non documentata e uno stride di fit diverso — la riga vecchia non va citata, questa sì.

### 0.1a Il floor NON è `ma_c`: è l'inviluppo su più teste

`ma_c` è la testa singola migliore, ma **tre delle altre non sono statisticamente distinguibili da lei**
(`pca` p=0.75, `ma_causal` p=0.37, `gauss` p=0.37) e **nessuna testa domina cliente per cliente**: su 31
client vince `ma_c` 13 volte, `pca` 9, `ma_causal` 6, `ar` 3.

| | median | mean |
|---|---|---|
| `ma_c` k=10 (testa **pre-registrata**) | 0.5054 | 0.4421 |
| **inviluppo = max su {`ma_c`, `ma_causal`, `ar`, `pca`}** | **0.5585** | **0.4993** |

⚠️ L'inviluppo è un **oracolo su test** (+0.055 in mediana), quindi è un *limite superiore*, non l'headline.
Ma il limite inferiore (la testa singola pre-registrata) e quello superiore **danno verdetti diversi**, e
solo ciò che sopravvive a entrambi è solido:

| arm deep | vs `ma_c` (p) | vs inviluppo (p) | sopravvive a entrambi? |
|---|---|---|---|
| `federated_cb_only_ema` | batte, 1e-05 | batte, 0.0020 | ✅ |
| `local` | batte, 6e-05 | batte, 0.0051 | ✅ |
| `centralized` | batte, 0.0029 | **non distinguibile, 0.058** | ❌ |
| `federated_cb_only` | batte, 0.027 | **non distinguibile, 0.209** | ❌ |
| `enc_fedproto` / `commoninit` / `fedprox` / `fedavg` | non distinguibili | non distinguibili | — (mai distinguibili) |

**Lettura onesta.** Contro il floor pre-registrato tre arm battono la baseline; contro il massimo di quattro
detector da 0–1024 parametri **ne restano due**, e la skyline `centralized` cade a p=0.058. Il paper deve
riportare **entrambe le barre**, non scegliere quella comoda.

⚠️ **Difetto del design, §6.** La regola pre-registrata "headline = testa selezionata su *val* con criterio
label-free (MSE one-step / di ricostruzione)" **non è implementabile come scritta**: le teste risolvono
compiti predittivi diversi (AR predice un passo avanti, PCA ricostruisce una finestra che *contiene* il
punto), quindi i loro MSE su val non sono sulla stessa scala e il confronto è senza senso. Non esiste in
questo repo un criterio label-free che ordini le teste. Finché non ne esiste uno, la regola operativa è:
**headline = testa singola pre-registrata, inviluppo pubblicato accanto ed etichettato come oracolo.**

> **Geometria canonica.** La riga `movavg10` (0.4421 mean / 0.5054 median) è rigenerata con la regola di
> accumulo di §3 su tutti i 31 client — **questo è il numero nel paper** (`federated_method.tex`, riga
> "Moving-average (10 min)", riempita il 2026-07-28). Riprodotta bit-identica dopo la riscrittura del
> runner: `Δ = 0.00e+00` su tutte le metriche di c0.
>
> ⚠️ Le righe `pca8_*`, `ar32_*` e `gauss` **erano** numeri del prototipo perduto, con geometria non
> canonica e λ/γ non documentati: **non citarle**. Le teste sono state riscritte
> (`scripts/floor_heads.py`, AR `p=32 λ=1e-4`, PCA `K=8 γ=0.05`, fit a stride 1) e rimisurate dal
> batch D/E di `run_floor.sh`; i valori nuovi vivono in `artifacts/floor/records_wsd_fed.jsonl` e si
> leggono con `python scripts/floor_table.py --dataset wsd_fed`.

### 0.1b Sensibilità a k (sweep a posteriori, n=31) — ✅ RIPRODOTTA 2026-07-28

Rimisurata da zero col runner riscritto (batch C di `run_floor.sh`, un run per valore, arm
`floor_ma_c_k<k>`): **ogni cella coincide con la tabella storica** — mean, median e p-value, a tutte e sette
le righe. La tabella qui sotto era quindi già in geometria canonica (a differenza delle righe `ar32_*` /
`pca8_*` di §0.1, quelle sì da buttare). Il numero di controllo più stringente è k=3, dove la testa degenera
**esattamente** in `0.25·diff1`: 0.2775 / 0.2547 allora, 0.2775 / 0.2547 adesso.

| config | mean | median | Wilcoxon vs k=10 |
|---|---|---|---|
| `diff1` (differenza prima) | 0.2775 | 0.2547 | p < 0.001 (peggiore) |
| `ma3` — **≡ 0.25·diff1**, esattamente | 0.2775 | 0.2547 | p < 0.001 (peggiore) |
| `ma5` | 0.3306 | 0.3458 | p < 0.001 (peggiore) |
| **`ma10` (pre-registrato)** | **0.4421** | **0.5054** | — |
| `ma20` | 0.4289 | 0.4344 | p = 0.915 |
| `ma32` | 0.3966 | 0.3750 | p = 0.622 |
| `ma50` | 0.3599 | 0.2807 | p = 0.195 |
| `ma128` | 0.3462 | 0.2880 | p = 0.120 |

**k=10 è l'argmax** per media e mediana — cioè il valore pre-registrato *prima* di qualunque misura è
anche il migliore. Da k=10 in su la curva è **piatta** (nessun k ≥ 20 differisce significativamente):
la baseline è insensibile alla manopola nell'intervallo utile. Solo k ≤ 5 degrada davvero, e a k=3 la
testa **degenera esattamente** nella differenza prima riscalata (il kernel edge-clipped di
`_moving_average_paper` a window=3 media 2 punti ⇒ residuo = 0.25·(x_t−x_{t−1})², e ogni metrica è
invariante a riscalamenti positivi).

Meccanismo, non fortuna: i segmenti anomali di wsd_fed hanno lunghezza **mediana 14** campioni
(= `metrics_tolerance`). Una media mobile lunga quanto l'evento è quella che l'evento disturba di più;
più corta lo insegue, più lunga lo diluisce. **L'ottimo sta sulla scala temporale dell'anomalia.**

⚠️ Lo sweep è una **verifica di robustezza, non una selezione**: si riporta il k pre-registrato. Non
esiste comunque un criterio label-free per tarare k (minimizzare il residuo su val premia k→2 senza
ottimo interno), quindi fissarlo a priori è l'unica strada difendibile.

**Il floor non è un numero: è 0.35–0.51 a seconda della testa.** Va riportato come *massimo su un set di teste
pre-registrato*, mai come singola riga scelta a posteriori (§6, regola anti-oracolo).

### 0.2 Le barre deep, sugli stessi 31 client

| arm | albero | median | mean |
|---|---|---|---|
| `federated_cb_only_ema` | `artifacts/converge60` | 0.6442 | 0.6302 |
| `local` | `artifacts/converged_all` | 0.6190 | 0.6169 |
| `centralized` | `artifacts/converged_all` | 0.5857 | 0.5847 |
| `federated_enc_fedproto` | `converge60` | 0.5357 | 0.5541 |
| `federated_enc_commoninit` | `converge60` | 0.5327 | 0.5143 |
| `federated_enc_fedprox` | `converge60` | 0.5238 | 0.4688 |
| `federated_cb_only` | `converge60` | 0.5231 | 0.5548 |
| `federated_enc_fedavg` | `converge60` | 0.4439 | 0.4512 |
| `local`/`centralized`/`federated` (NON usare) | `artifacts/fed_eval/wsd_fed`, pre-purge | 0.3030 / 0.3930 / 0.0161 | — |

⚠️ **L'albero va nominato nella pre-registrazione.** `fed_eval` e `converge60`/`converged_all` danno, per lo
stesso arm e gli stessi 31 client, numeri che differiscono di più dell'MDE. Solo i secondi sono post-purge e
convergiti.

### 0.2b `ucr_split` — misurato 2026-07-29 (1130 client / 226 cluster, W=128, tol 64)

`floor_ma_c_k10`, unità **cluster**: median **0.0094**, mean 0.0478 (identico al run precedente ⇒ percorso
di scoring invariato). Verificato che le 5 shard di ogni cluster danno metriche threshold-free identiche
entro **8.5e-06** su 226/226 cluster: l'unità onesta è il cluster, n=226 e non 1130.

**Su questo dataset la top-K si comporta all'opposto di wsd** — ed è la metrica headline della letteratura
UCR, quindi conta:

| metrica | wsd_fed | ucr_split |
|---|---|---|
| `paper_top1` | 0.871 mean, **1.000 median** (satura) | **0.283 mean, 0.000 median** (discrimina) |
| `paper_top3` / `top5` | 0.935 / 0.968 | 0.332 / 0.345 |
| `vus_pr` | 0.5054 median | 0.0094 median |
| `auroc` | 0.9883 (satura) | 0.5809 |

Cioè: **una media mobile a 10 campioni azzecca la localizzazione top-1 sul 28% delle serie UCR**. È il numero
da mettere accanto al `paper_top1` degli arm deep — che è il modo in cui l'archivio UCR viene valutato in
letteratura — non solo accanto a VUS-PR.

**Matched contro gli arm deep** (`floor_stats.py`, unità cluster). ⚠️ n=45–49 e non 226: lo sweep deep si è
fermato al 19,5% ed è stato interrotto il 2026-07-29. Su quella frazione **ogni** arm deep batte il floor:

| arm | median | Δ med | p | verdetto |
|---|---|---|---|---|
| `centralized` | 0.2318 | +0.224 | <1e-5 | batte |
| `local` | 0.1195 | +0.111 | <1e-5 | batte |
| `enc_commoninit` | 0.1034 | +0.095 | <1e-5 | batte |
| `enc_fedproto` | 0.0677 | +0.060 | <1e-5 | batte |
| `enc_fedavg` | 0.0624 | +0.054 | <1e-5 | batte |
| `enc_fedprox` | 0.0607 | +0.052 | <1e-5 | batte |

**Contrasto con wsd, che è il risultato interessante:** là il trio `enc_*` non è distinguibile da una media
mobile; qui lo è, con p<1e-5. Non è il metodo a cambiare — è il *task*. Su `ucr_split` la baseline sta a
0.0094 e c'è spazio sopra; su `wsd_fed` sta a 0.5054 e metà degli arm ci sbatte contro. Prima di concludere
"la federazione non funziona" bisogna dire su quale dei due dataset.

⚠️ Il floor è a `W=128` perché a `W=128` erano gli arm deep su disco. Con la finestra sbagliata su tutte e
250 le serie UCR (`documentation/UCR_SPLIT_*`), **entrambi i lati** del confronto sono da rifare quando il
deep passa a `2×periodo` — e il floor va rifatto con `--window` corrispondente.

### 0.3 Il risultato che decide: FLOOR (`movavg10`) vs ogni arm deep

Wilcoxon signed-rank appaiato su `(cluster, entity)`, n=31, α=0.05. **Rigenerabile:**

    python scripts/floor_table.py --dataset wsd_fed          # costruisce il CSV matched
    python scripts/floor_stats.py --dataset wsd_fed --metric vus_pr

(`floor_stats.py` aggiunge CI bootstrap della differenza mediana, TOST opzionale e il **margine minimo per
l'80% di potenza** a questo n — 0.102–0.210 secondo l'arm: qualunque margine più largo *fabbrica*
l'equivalenza.) I p-value qui sotto sono del prototipo; `floor_stats.py` dà gli stessi verdetti con p
leggermente diversi (scipy usa il test esatto sotto n=50): `cb_ema` 1e-5, `local` 6e-5, `centralized`
0.0029, `cb_only` 0.027, `fedproto` 0.087, `commoninit` 0.224, `fedprox` 0.456, `enc_fedavg` 0.992.

| arm deep | median | Δ median | mean Δ | SD(Δ) | SE | p | verdetto |
|---|---|---|---|---|---|---|---|
| `federated_cb_only_ema` | 0.6442 | +0.137 | +0.187 | 0.201 | 0.036 | **0.00003** | batte il floor |
| `local` | 0.6190 | +0.112 | +0.174 | 0.219 | 0.039 | **0.0001** | batte il floor |
| `centralized` | 0.5857 | +0.078 | +0.141 | 0.239 | 0.043 | **0.0051** | batte il floor |
| `federated_enc_fedproto` | 0.5357 | +0.028 | +0.111 | 0.372 | 0.067 | 0.094 | **non** distinguibile |
| `federated_enc_commoninit` | 0.5327 | +0.025 | +0.071 | 0.376 | 0.068 | 0.217 | **non** distinguibile |
| `federated_enc_fedprox` | 0.5238 | +0.016 | +0.026 | 0.369 | 0.066 | 0.444 | **non** distinguibile |
| `federated_cb_only` | 0.5231 | +0.016 | +0.112 | 0.317 | 0.057 | 0.027 | p<0.05 ma Δmedian ≈ 0 |
| `federated_enc_fedavg` | 0.4439 | −0.063 | +0.008 | 0.417 | 0.075 | 0.992 | il floor **non** è peggiore |

**Lettura.** Le tre barre buone (`cb_ema`, `local`, `centralized`) battono davvero una media mobile a 10
campioni con **zero parametri** — il modello deep non è una frode. Ma **l'intero trio di federazione
dell'encoder più `cb_only` non è statisticamente distinguibile da una media mobile**, e `enc_fedavg` sta
sotto. Nota anche `SD(Δ)` 0.32–0.42 per gli `enc_*` contro 0.20–0.24 per i buoni: quegli arm sono instabili
tra client, non solo mediocri.

### 0.4 Esattezza della federazione via statistiche sufficienti

**Rimisurato 2026-07-28** su tutti e 4 i cluster (`floor_eval.py --witness`, gate 1e-9,
`artifacts/floor/witness_wsd_fed.json`). `param relerr` = `‖θ_fed − θ_central‖/‖θ_central‖`,
`score relerr` = `max|Δs|/max|s|` sui punteggi di test:

| head | c0 | c1 | c2 | c3 |
|---|---|---|---|---|
| AR(32) | 8.3e-14 / 3.8e-14 | 4.0e-14 / 1.5e-14 | 2.0e-14 / 9.1e-15 | 1.4e-13 / 1.7e-13 |
| PCA(8) | 6.0e-14 / 5.3e-15 | 1.9e-13 / 1.4e-14 | 1.4e-13 / 1.8e-15 | 9.5e-14 / 5.2e-15 |
| Gauss | 5.1e-14 / 3.5e-14 | 3.3e-14 / 1.7e-14 | 2.3e-14 / 8.2e-15 | 5.6e-14 / 3.9e-14 |

**12/12 PASS**, tutti fra 1e-15 e 2e-13 — arrotondamento float64, la firma corretta.

⚠️ Nota sulla vecchia tabella: riportava `Gauss diag → param relerr 0.00e+00`. Per la regola di §0.4 quello
**è un fallimento**, non un successo: uno zero esatto significa che i due percorsi condividono un oggetto.
La riga vecchia falliva il proprio gate e nessuno se n'era accorto; il gate è ora implementato in
`witness_verdict()` e restituisce `FAIL(circular: exact zero)`.

A livello di *metrica* invece `fed_exact` e `central` danno `max|Δvus_pr| = 0.00e+00` su tutti i client — ed
è corretto così: i parametri differiscono di 1e-14, molto sotto la risoluzione di qualunque metrica. Il gate
va sui parametri e sui punteggi, mai sulla metrica.

Confronto **non circolare**: statistiche calcolate sulle finestre **concatenate** (`stats(pooled)`) contro la
**somma** delle statistiche per-client (`Σ_k stats_k`), poi risolte e scorate indipendentemente. Le
magnitudini 1e-12…1e-14 sono arrotondamento float64 — la firma corretta di una verifica reale.

> ⚠️ **Trappola da non ripetere.** Un prototipo precedente riportava `max|Δvus_pr| = 0.00e+00` tra
> `fed_exact` e `centralized`. Era **circolare**: entrambi gli arm ricevevano lo stesso oggetto `w_central`,
> già calcolato sommando statistiche. Quel run dimostra determinismo della pipeline, non esattezza della
> federazione. Uno zero *esatto* invece di ~1e-13 è il campanello d'allarme. Il gate di §7 richiede un
> **percorso pooled indipendente**.

### 0.5 Il controllo positivo SCATTA — e isola il meccanismo

**2026-07-28.** L'arm `floor_fed_naive` era pre-registrato come **controllo positivo**: rompere il gauge
(segno + permutazione) di una base *deve* far collassare anche il modello convesso. **Collassa**, su tutti e
quattro i cluster, con un margine enorme:

| | c0 (5) | c1 (11) | c2 (9) | c3 (6) | tutti (31) |
|---|---|---|---|---|---|
| `local` | 0.3374 | 0.3542 | 0.5354 | 0.4911 | 0.4277 |
| `fed_fedavg` (media dei **proiettori**) | 0.3767 | 0.3765 | 0.5386 | 0.4727 | 0.4471 |
| `fed_naive_aligned` (basi grezze LAPACK) | 0.3061 | 0.3350 | 0.5700 | 0.3659 | 0.3810 (p=0.95) |
| **`fed_naive` (gauge randomizzato)** | **0.0206** | **0.2851** | **0.0781** | **0.0843** | **0.1154 (p<1e-4)** |

Stesso identico input, stessa aggregazione, unica differenza: **in che base sono espressi i parametri
mediati**. Media dei proiettori → nessun danno (0.4471, indistinguibile da local). Media delle stesse basi
in un gauge arbitrario → **−0.33 di mediana**, p<1e-4 contro `fed_fedavg`.

> 🔴 **Ritratta una mia conclusione di poche ore prima.** Avevo scritto che il controllo "non collassa e non
> può collassare", sulla base dell'argomento *una base permutata-e-segnata spanna lo stesso sottospazio,
> quindi il proiettore si recupera*. L'argomento è vero **solo se i client condividono il sottospazio**, ed
> era esattamente il caso del fixture sintetico che avevo usato (tre fette dello stesso segnale, K=4).
> Appena i sottospazi differiscono — cioè sempre, su dati veri — mediare la colonna *j* di A con la colonna
> *σ(j)* di B produce qualcosa che non sta in nessuno dei due. Il selftest ora asserisce **entrambi i
> regimi**: `pca(a)` gauge innocuo a sottospazio condiviso, `pca(b)` collasso a coorte realistica (K=8, 6
> client: energia catturata 0.525 → 0.236). La lezione: un fixture omogeneo non falsifica un meccanismo che
> vive nell'eterogeneità.

**Cosa compra il controllo.** Il collasso deep (`NaiveAvg` 0.048 vs `Ensemble` 0.326) aveva due spiegazioni
concorrenti: (i) mediare i pesi è intrinsecamente sbagliato, (ii) mediare pesi espressi in basi non
allineate è sbagliato. Il modello convesso le separa: `fed_fedavg` **non fa danno** (media in coordinate
canoniche), `fed_naive` **distrugge** (stessa media, gauge arbitrario). ⇒ **è la simmetria di
riparametrizzazione, non la media, a rompere il weight-space** — e ora è un'affermazione con un controllo
positivo dietro, non un'inferenza da un null.

### 0.5b Il contrasto meccanicistico (il contributo positivo) — misurato su n=31

**Sull'intero asse della federazione, su un modello convesso, non succede niente.** Wilcoxon appaiato
contro `local`, stessa testa, stessi 31 client (batch E):

| modo | AR(32) median (p) | PCA(8) median (p) | Gauss median (p) |
|---|---|---|---|
| `local` | 0.3961 | 0.4277 | 0.3363 |
| `central` ≡ `fed_exact` | 0.3752 (0.98) | 0.4249 (0.82) | 0.3334 (0.19) |
| `fed_fedavg` | 0.3844 (0.99) | 0.4471 (0.39) | 0.3115 (0.86) |
| `fed_fedavg_uniform` | 0.4202 (0.72) | 0.4704 (0.27) | 0.3146 (0.94) |
| `fed_oneclient` | 0.2918 (0.13) | 0.3700 (0.85) | 0.3638 (0.21) |
| `fed_scaleonly` (AR) | **0.3961, Δ = 0.00e+00 esatto** | — | — |

**18 confronti, zero significativi.** Federare esattamente, mediare i parametri, mediarli senza pesi, o
ritrasmettere il modello di un singolo client: sul modello convesso sono tutti indistinguibili dal fit
locale. Nel caso deep gli stessi modi coprono 0.44–0.64 e `NaiveAvg` crolla a **0.048** contro `Ensemble`
0.326 su chiavi identiche (`documentation/MATCHED_FUSION_FINDINGS.md`).

⇒ **il collasso weight-space non è causato dalla media**, ed è il controllo positivo di §0.5 a dire da cosa
è causato: dalla **riparametrizzazione**. Stessa media, stesso input, gauge arbitrario invece che canonico →
0.1154 contro 0.4471. Il null convesso da solo sarebbe stato un'inferenza debole; con il controllo che scatta
è un'attribuzione.

**La perdita di aggregazione ha comunque un numero**, anche dove il Δ metrico è nullo: `J(ŵ)−J(w*)` in forma
chiusa (§1.4) vale **113.3 / 45.8 / 35.1 / 14.3** su c0/c1/c2/c3 per `fed_fedavg` su AR (e 106.1 / 27.8 /
17.3 / 48.7 per `fed_fedavg_uniform`) — stampigliato
in `_excess_objective` su ogni riga. È il modo di pubblicare il risultato senza appoggiarsi a un
non-risultato: *la media dei pesi peggiora l'obiettivo in modo misurabile e non lo si vede in metrica*.

`fed_scaleonly` chiude il regression test: Δ **esattamente** 0.00e+00 contro `local` su tutte e 31 le
entità ⇒ nessun leak sull'asse metrico (normalizzazione, impulse, soglia sono tutti positivamente omogenei
come previsto).

### 0.6 L'impulse term vale più della baseline — misurato su n=31 (2026-07-28)

⚠️ **Il numero "+0.41" era il MASSIMO, non il tipico.** Veniva da `kpi_015`, che è il client dove l'impulse
term rende di più. Ora l'ablazione è appaiata su tutti e 31 i client
(`floor_ma_c_k10` vs `floor_ma_c_k10__noimpulse`, batch B di `run_floor.sh`, Wilcoxon appaiato):

| metrica | ON (median / mean) | OFF (median / mean) | Δ median | Δ mean | p |
|---|---|---|---|---|---|
| **`vus_pr`** | **0.5054 / 0.4421** | **0.3249 / 0.3543** | **+0.180** | +0.088 | **8.9e-05** |
| `auprc` | 0.5335 / 0.5086 | 0.4232 / 0.4228 | +0.110 | +0.086 | 1.0e-07 |
| `pate_f1` | 0.5808 / 0.5687 | 0.4852 / 0.4804 | +0.096 | +0.088 | 7.0e-06 |
| `auroc` | 0.9883 / 0.9522 | 0.8391 / 0.8235 | +0.149 | +0.129 | 1.7e-06 |
| `best_f1` | 0.5652 / 0.5413 | 0.5000 / 0.4905 | +0.065 | +0.051 | 2.9e-04 |
| `affiliation_f1` | 0.8693 / 0.8261 | 0.7740 / 0.7676 | +0.095 | +0.058 | 6.4e-04 |
| `f1` (soglia fissa) | 0.1564 / 0.1914 | 0.2188 / 0.2138 | **−0.062** | −0.022 | 0.24 (ns) |
| `paper_top1` | 1.0000 / 0.8710 | 1.0000 / 0.8710 | 0.000 | 0.000 | 1.00 |

**Tre letture che il singolo client nascondeva:**

1. Il guadagno è **reale ma non uniforme**: positivo su **26/31** client, range **−0.164 … +0.409**. Chi
   perde (`kpi_131` 0.329→0.165) ha anomalie brevi che il filtro a 128 campioni diluisce.
2. Aiuta il **ranking**, non la **soglia**: `f1` a quantile fisso va nella direzione opposta (ns). L'impulse
   term sposta la forma della distribuzione dei punteggi, quindi una soglia calibrata sul train non ne
   beneficia automaticamente.
3. `paper_top1/3/5` è **identico** con e senza: la posizione del massimo assoluto non si muove. Conferma che
   su wsd la top-K è satura e non discrimina (§3).

Il post-processing di `detect` (`0.5·(raw + _moving_average_paper(raw,128))`, `config.py:298`) vale quindi
**+0.18 VUS-PR in mediana su un filtro da zero parametri**, fino a +0.41 nel caso migliore. Nessuno nel
progetto l'aveva mai misurato: riformula il significato di ogni numero assoluto riportato finora e va
pubblicato come risultato secondario di prima classe. I valori `0.3742 → 0.7875` del prototipo sono
**superati** (geometria non canonica): non citarli.

### 0.7 Costo (misurato, CPU, zero GPU)

- statistiche sufficienti, tutti i 31 client: **0.6 s**
- per (arm, entity) end-to-end score+metriche: n=90, tot 488.8 s → **mean 5.43 s**, median 3.21, max 19.0.
  Percorso con shim: n=186, mean 8.98 s.
- Il collo di bottiglia è **il calcolo VUS**, non il modello.
- 1 arm × 1 head × 31 client ≈ **4.6 min** su singolo core. Matrice completa (~992 eval) ≈ **2.5 h** singolo
  core, minuti con `--jobs`.

---

## 1. Il modello

### 1.1 Setting

Client `k`: serie train z-scorata per-entità (`PerEntityScaler` fittato su train, `data.py:98,463-468`) —
**mai ri-normalizzare**. `C=1`. `W = cfg.dataset.window_length = 128`,
`S = detect._resolve_eval_stride(cfg) = 13`.

Ogni head è una tripla `(stats, solve, score)`:
- `stats(x) → Σ` statistica sufficiente **additiva** (o `None` se non-parametrica),
- `solve(Σ, knobs) → θ` forma chiusa, nessun ottimizzatore,
- `score` **series-causal** (`R^T → R^T`, calcolato una volta e affettato per finestra) oppure
  **window-native** (`R^W → R^W`, ricalcolato per finestra). Le due granularità devono essere esplicite:
  PCA/Gauss non sono esprimibili con la sola granularità a fetta.

### 1.2 Il set di teste pre-registrato

**Teste di decisione** (il floor è il **massimo su queste quattro**, selezionato su *val*, §6):

**H1 `ma_c`** — box centrato, riusando `detect._moving_average_paper(x, k)` **verbatim**
(`pipeline/detect.py:232-250`):

    ŝ_t = ( x_t − MA_k(x)_t )²        k = 10        # 10 min a dt_sec=60

Zero parametri fittati ⇒ **arm-invariante per costruzione**. È la riga "Moving-average (10 min)"
pre-registrata in `federated_method.tex:281`. Il detector deep è una **ricostruzione di finestra
non-causale**: il filtro centrato è quindi il comparatore *equo*; quello causale è una variante handicappata,
riportata a parte.

**H2 `ma_causal`** — `x̂_t = (1/k)Σ_{j=t−k}^{t−1} x_j`, `k=10`. Variante a informazione causale.

**H3 `ar`** — ridge AR(p), series-causal. Con `φ_t = [x_{t−1},…,x_{t−p}]`, `t = p…T−1`:

    n_k = T_k − p ,  G_k = Σ_t φ_t φ_tᵀ ,  b_k = Σ_t φ_t x_t ,  q_k = Σ_t x_t²
    w(S)  = ( G + λ·N·I )⁻¹ b            N = Σ_k n_k ,  G = Σ_k G_k ,  b = Σ_k b_k
    σ̂²(S) = max( ( q − 2 wᵀb + wᵀG w ) / N , 1e-8 )      # forma chiusa, nessun secondo passaggio
    ŝ_t   = ( x_t − φ_tᵀ w )² / σ̂²   per t ≥ p ;   ŝ_t = 0  per t < p

`λ` scalato per `N` rende la penalità invariante a lunghezza e dimensione coorte. Uplink = `1+p²+p+1` float64
= **8.7 kB a p=32**.

`σ̂²` è **metrica-neutra**: costante positiva per-entità, e impulse term + aggregazione canali + soglia a
quantile su train sono tutti positivamente omogenei ⇒ ogni metrica è invariante. Sfruttato come rilevatore di
vacuità (`floor_fed_scaleonly`, §2).

**H4 `pca`** — window-native su finestre a stride `S`:

    n_k = #finestre ,  s_k = Σ_i w_i ,  S_k = Σ_i w_i w_iᵀ
    μ = s/n ,  Σ = S/n − μμᵀ ,  Σ_γ = (1−γ)Σ + γ(tr Σ/W)I ,  γ = 0.05
    Σ_γ = U diag(λ) Uᵀ  →  U_K = U[:,:K] ,  P = U_K U_Kᵀ            # K = 8
    r_i = (I − P)(w_i − μ)  →  contributo per-timestep dentro la finestra i = r_i ⊙ r_i

`γ>0` ⇒ sempre PD, quindi definita anche sui client con poche finestre. Uplink `W²+W+1` float64 = **132 kB**.
**Il parametro canonico è `P`, non `U_K`** — invariante a segno e degenerazioni, cosa che conta sia per la
media sia per il witness.

**Teste secondarie** (mai nel massimo): `gauss` (whitener `Σ_γ^{−1/2}`, = NLL gaussiana a meno di costanti);
`knn` (**deliberatamente non-additiva**: banca di finestre train, cap 20 000 righe *identico* per local e
pooled, score = media dei κ=5 quadrati minimi — nessuna statistica additiva finita esiste, quindi è il test
falsificabile della linea di demarcazione); `random` (ancora inferiore: dà la scala e mostra la saturazione di
AUROC).

### 1.3 La proposizione di esattezza

La design matrix pooled su un cluster è la **concatenazione verticale** delle righe dei membri — esattamente
ciò che il `ConcatDataset` di `_pooled_loader` produce (`pipeline/federated_eval.py:381-400`). Quindi
`G_pool = Σ_k G_k`, `b_pool = Σ_k b_k`, `N = Σ_k n_k` è un'**identità**, e in aritmetica esatta
`solve(Σ_k stat_k) ≡ solve(pooled)`. Vale identicamente per momenti primi/secondi di finestra (PCA, Gauss).

Non è novità algoritmica: è `k-FED`/Prop.1, già nel ledger come "non novel". La novità d'uso è che dà **lo
zero dell'asse perdita-di-aggregazione**, che nessun arm deep può fornire.

### 1.4 Il quantificatore di perdita in forma chiusa

Per l'head AR l'eccesso di objective di qualunque `ŵ` sull'ottimo pooled `w*` è calcolabile senza dati:

    J(ŵ) − J(w*) = (ŵ − w*)ᵀ (G + λN I) (ŵ − w*)

⇒ la perdita di aggregazione ha un numero anche quando il Δ metrico è **null** (che è il caso, §0.5). È il
modo di pubblicare quel risultato senza appoggiarsi a un non-risultato.

---

## 2. Tabella di mapping degli arm

Naming: `floor_<arm>__<head>[__<knob>]`. **Mai** prefisso `fa_` (`scripts/aggregate_all.py:30` non li
globba); **mai** chiamare un arm letteralmente `local`/`centralized` (`scripts/summarize_converged.py:64-65`
appaia esattamente quei due nomi e mislabellerebbe la baseline).

### 2.1 Arm critici per la decisione

| Arm | Rispecchia | Cosa condivide | Aggregazione | fed == central? | Cosa misura comunque |
|---|---|---|---|---|---|
| `floor_local` | `local` (`federated_eval.py:423`) | niente | — | n/a | Riferimento per-client. Stesso information set del `local` deep. Deterministico ⇒ varianza di seed **esattamente 0**, e l'unità onesta può solo essere n_entità (31 su wsd; **226, non 1130**, su `ucr_split`). |
| `floor_central` | `centralized` (`:453-478`) | finestre grezze (skyline illegale) | **percorso lento indipendente**: materializza la design matrix pooled e risolve su quella | definizione di riferimento | Beneficio del pooling **con zero confondimento di ottimizzazione**. |
| `floor_fed_exact` | `federated_cb_only`, `protoprior`, ogni merge Prop.1 | 8.7 kB (AR) / 132 kB (finestra), **un round**, nessun gradiente | il server somma elemento per elemento, risolve una volta, ritrasmette | **ESATTO** (§1.3), testimoniato contro il percorso indipendente | Il termine quantità/diversità-dati con perdita di aggregazione **provabilmente nulla**. *Onestà:* qui "federazione" è semanticamente sottile — un round, nessuna iterazione, nessun drift. È un'affermazione di **legalità** (solo statistiche sufficienti lasciano il client), non un algoritmo di ottimizzazione. La sottigliezza è il punto: è lo zero dell'asse. |
| `floor_fed_fedavg` | `federated_enc_fedavg`, `naiveavg` | il **parametro risolto** (`w_k`, o `P_k`) | media pesata `n_k` (come `pipeline/federated.py:1899`); sottospazi mediati **come proiettori** `P̄ = Σα_k P_k → top-K` | **NO**, con motivo in forma chiusa: `(ΣG_k)⁻¹Σb_k ≠ Σα_k G_k⁻¹b_k` a meno che tutti i `G_k` siano proporzionali | La cella portante. Misurato: null metrico (0.348 vs 0.356) con gap parametrico 15–33%. Da riportare con `J(ŵ)−J(w*)` accanto. |
| `floor_fed_fedavg_uniform` | `--fedproto-agg uniform` | `w_k` | media non pesata | NO | Isola il *pesaggio* dalla *media*. Su wsd `n_k` va da 2862 a 16669 finestre ⇒ il contrasto ha range. |
| `floor_fed_naive` / `floor_fed_naive_aligned` | la patologia `NaiveAvg` deep | matrici `U_K` grezze (solo PCA) | media **senza** allineamento (gauge segno+permutazione randomizzato per client / base grezza LAPACK), poi ri-ortonormalizza | NO — **collassa**: 0.1154 vs 0.4471 di `fed_fedavg`, p<1e-4 | ✅ **Controllo positivo, scatta** (§0.5). Isola la causa: stessa media, unica differenza il gauge. `_aligned` è il caso di confronto (LAPACK dà segni deterministici e coerenti ⇒ nessun danno, 0.3810 p=0.95). |
| `floor_random` | — | niente | — | arm-invariante | Ancora inferiore. |

### 2.2 Arm di mapping / meccanismo

| Arm | Rispecchia | Aggregazione | fed == central? | Misura |
|---|---|---|---|---|
| `floor_fed_sharedbasis` | `federated_cb_only` = **contributo (A)** | il server somma `S_k`, autodecompone, ritrasmette `U_K`; `μ_k` e scale restano locali | **ESATTO** nella componente condivisa | Contributo (A) con zero perdita di ottimizzazione — irraggiungibile per gli arm deep. Con `floor_fed_fedavg` forma un 2×2 pulito: (rappresentazione vs testa) × (esatto vs mediato). |
| `floor_fed_sharedbasis_avg` | il foil naive di (A) | `P̄ = Σα_k P_k → top-K` | NO | L'altra cella del 2×2. |
| `floor_fed_scaleonly` | **contributo (B)** al suo minimo | scala pooled `σ̂²`, `w` tutto locale | **PARI BIT-IDENTICO a `floor_local`, pre-registrato** | Etichetta onesta: questa "federazione" è **semanticamente vuota** (σ̂² è metrica-neutra, §1.2). Serve come **regression test sull'asse metrico**: qualunque Δ diverso da zero significa che normalizzazione o scaling di soglia stanno perdendo. Non misura nulla sulla federazione e non va mai presentata come se lo facesse. |
| `floor_fed_prox[μ]` | `federated_enc_fedprox` | punto prossimale esatto `w_k = (G_k + λn_kI + μn_kI)⁻¹(b_k + μn_k w^(r))`, poi media pesata; μ ∈ {0,1e-2,1e-1,1,10,∞} | NO (μ=0 ⇒ local, μ→∞ ⇒ global) | Il null deep di FedProx è un **artefatto del solver** (AdamW persistente divide il gradiente prox per √v̂ ⇒ μ≤0.1 è no-op, `documentation/FED_ENCODER_ALGOS.md`). Qui non c'è ottimizzatore. **Dizione legale:** "μ è ben comportato nel caso convesso, quindi il no-op deep non è una proprietà dell'obiettivo". **Non** stabilisce se μ possa aiutare un corpo deep disallineato. |
| `floor_fed_localgd[η,τ,R]` | `federated_fedsgd*`, `federated_fedavgm`, ogni arm a R round × τ passi | iterata quadratica in forma chiusa `w^(r+1) = Σ_k α_k (I − ηA_k)^τ w^(r) + bias`, `A_k = G_k/n_k + λI` | NO; τ→1 tende a pooled, τ→∞ a local | **Senza questo nessun arm FLOOR rispecchia ciò che gli arm deep fanno davvero** (tutti sono R round × τ passi locali). Chiuso per round, costo trascurabile. |
| `floor_central_capN` | `centralized_cap` (audit-SOLID, capN +0.410 su c3) | statistiche da sottocampione limitato | non esatto vs central pieno | **Due manopole**: (a) fissa #finestre pooled, varia #client; (b) fissa #client, varia #finestre. L'unica curva analitica diversità-vs-quantità senza confondimento. Ritesta uno dei due risultati audit-SOLID. |
| `floor_oneclient` | il controllo "corpo da un client" (deep 0.019) | un `w_{k0}` ritrasmesso | NO | Peggior caso legale. Il pilota misura 0.253/0.186 con collasso su c3 ⇒ il collasso one-client deep si riproduce in un modello a 8 parametri: è un fatto di **eterogeneità dei dati**, non dei corpi deep. |
| `floor_fed_coreset` (knn) | FedProto per non-parametrici | M=256 centroidi k-means + conteggi | **impossibile** — nessuna statistica additiva finita | Quantifica la perdita da sommarizzazione lossy; con la riga sotto testa se **l'additività** (non la profondità) è la linea di demarcazione della federazione lossless. |
| `floor_fed_ensemble` (knn) | `Ensemble` nella matrice di fusione | array di score per-timestep, `s(t) = min_k s_k(t)` (anche `mean`) | NO, e deliberatamente **function-space** | La dicotomia weight-vs-function-space (NaiveAvg 0.048 vs Ensemble 0.326) su un detector **senza pesi da mediare** — confondimento rimosso per costruzione. |
| `floor_fed_dp[ε]` | — | `Σ_k stat_k` + rumore gaussiano calibrato sulla sensibilità L2 clippata di un client, δ=1e-5, ε ∈ {∞,10,1,0.1} | ESATTO a ε=∞, degrada monotono | Un positivo genuino che la pipeline deep multi-round **strutturalmente non può** eguagliare (brucia budget per round). Precedente: `scripts/fa_dp.py`. |
| `floor_ema_shard[γ]` | `federated_cb_only_ema` | statistiche da shard cronologici disgiunti, `S^(r) = γS^(r−1) + (1−γ)Σ_k S_k^(r)`, γ=0.8 | NO; su dati statici è un **no-op provabile** | **SOLO PLUMBING — NESSUNA CLAIM.** Tutti e tre i giudici: come spiegazione di `cb_ema` è un errore di categoria. `cb_ema` smorza il drift di un codebook SGD *non stazionario* tra round (37.7→12.1); una soluzione in forma chiusa non ha ottimizzatore e non ha drift, quindi un null qui non distingue "il guadagno di cb_ema non è smoothing" da "non c'era nulla da smorzare". Tenere l'arm, vietare l'inferenza. |

### 2.3 Fatti di arm-invarianza da dichiarare, non da vendere

`ma_c`, `ma_causal`, `diff1`, `random` hanno **zero parametri fittati** ⇒ local == central == ogni modo federato,
bit-identico. Riportati **una volta**, con la nota "a questa testa non si può porre alcuna domanda di
federazione". Idem `floor_fed_scaleonly`. Il runner **salta** le combinazioni (head × mode) provabilmente
nulle stampando il motivo, e non le emette mai come risultati di federazione.

---

## 3. Percorso score → metrica

**Step 0 — config (obbligatorio).** Come `scripts/mixture_eval.py:126-134`:

    cfg = Config(); cfg.dataset.name = ds
    apply_dataset_overrides(cfg)   # config.py:436-467 → wsd_fed tolerance 14 (altrimenti silenziosamente 64)
    apply_env_overrides(cfg)       # config.py:398
    cfg.evaluation.save_plots = False

Banner da verificare a runtime:
`[config] wsd_fed: metadata.json declares metrics_tolerance=14 -> overriding the window-derived default (64)`.

**Step 1 — record.** `tr, va, te = data.load_scaled_records(c)` con `c.dataset.entity_id = e`
(`data.py:440`). Nessun DataLoader, nessun modello torch, nessuna GPU (0.004 s/client).

**Step 2 — accumulatore.** `ents = detect._init_entity(records)` (`detect.py:193`) → `channel_sum` `(T,C)`
float64 + `coverage` `(T,)`.

**Step 3 — geometria finestre.** Enumerare con
`data.SlidingWindowDataset(records, cfg.dataset.window_length, detect._resolve_eval_stride(cfg)).indices` —
lo stesso oggetto e lo stesso stride (13) che `detect` costruisce a `detect.py:306-311`. Eredita gratis il
drop `T < W` (`data.py:369-370`).

    v = head_score_for_window(...)          # vettore lungo W, ≥ 0
    e["channel_sum"][start:stop, 0] += v
    e["coverage"][start:stop]       += 1.0

**Nessun offset `skip`.** Mascherare i primi `p` timestep *della finestra* cambia `coverage(t)`, e poiché
`rolling_aggregation="sum"` **non divide mai** per coverage (`detect.py:264-267`) de-allinea il profilo dagli
arm deep. Contribuire la finestra intera (i primi `p` timestep *della serie* prendono 0), tenere coverage
uniforme, e **dichiarare** che le teste series-causal usano tutto il passato causale.

**Step 4 — assemblaggio.** `detect._assemble_rolling(channel_sum, coverage, cfg.scoring.rolling_aggregation)`
— modo `"sum"`: nessuna divisione, testa/coda riempite dal punto coperto più vicino
(`detect.py:215-229, 253-270`).

**Step 5 — finalize, TRAIN PRIMA.** `detect._finalize_entities(train_ents, cfg)` **poi**
`scores, labels = detect._finalize_entities(test_ents, cfg)` (`detect.py:430-468`) — impulse term
`0.5·(raw + _moving_average_paper(raw,128))`, poi `_aggregate_channels_np` (`max`, no-op a C=1). Nessun
troncamento: `metrics_core.align_score_and_label` è codice morto, non chiamarlo.

**Step 6 — soglia.** `thr = detect._fit_threshold_paper(train_ents, cfg)` (`detect.py:517`). Senza
`per_rate_CF_sum` (caso `stage2=None`) prende il fallback documentato
`np.quantile(concat(train overall_scores), 0.99)` (`:539-543`). Lo Step 5 sui train ents **deve** precedere:
il fallback legge `overall_scores`. Ogni record stampigliato `_threshold_rule: "quantile_fallback"`.

**Step 7 — metriche: TUTTE quelle che `detect()` scrive in `report.json`.** `preds = (scores > thr)`, poi
esattamente la sequenza di `detect.detect()` (`detect.py:1347-1422`), implementata in
`floor_eval.full_metrics`:

    rep.update(detect._detection_metrics(labels, preds, scores, cfg))  # 15 chiavi
    rep.update(detect._event_metrics(labels, preds))                   # 3 chiavi
    # blocco macro *_macro, gated su >1 entità (come detect.py:1361)
    # per-entità detect._paper_metrics -> paper_top1/top3/top5_acc_at_<tol>, mediati
    rep["_suite"] = metrics_core.evaluate_scores(...)   # threshold_free.{no_pa,pa}
                                                       # + by_threshold.<strategy>.{no_pa,pa}

Elenco completo delle chiavi flat prodotte (buffer VUS/PATE = `cfg.evaluation.paper_metrics_tolerance`):

| blocco | chiavi | soglia? |
|---|---|---|
| ranking | `vus_pr`, `vus_roc`, `auprc`, `auroc`, `best_f1`, `pate_f1` | **free** |
| paper top-K | `paper_top1_acc_at_<tol>`, `paper_top3…`, `paper_top5…` | **free** (locali-massimi su `scores`) |
| operativo | `precision`, `recall`, `f1`, `fpr`, `pate` | dipendente |
| affiliation | `affiliation_precision`, `affiliation_recall`, `affiliation_f1` | dipendente |
| eventi | `event_precision`, `event_recall`, `event_f1`, `detection_delay_mean` | dipendente |
| macro (>1 entità) | `<metrica>_macro`, `n_entities_macro_avg` | free |

**Disciplina sull'headline.** Solo il blocco threshold-free è comparabile col deep: gli arm deep usano la
regola per-τ del paper, FLOOR necessariamente il fallback a quantile. Tutto il blocco dipendente dalla soglia
va in appendice con quel caveat in didascalia. `auroc` solo accanto a `floor_random` (satura ai tassi di
anomalia wsd 0.0005–0.0344) — e **`paper_top1/3/5` satura anche peggio**: mediana 1.00 per *ogni* arm deep su
wsd, quindi è una metrica di sanità, non di ranking. `cf_*` (RQ4) omesso: richiede `test_clean/`+`test_mask/`,
assenti su wsd_fed e ucr_split.

**Cosa resta fuori, e perché non è una lacuna.** `channel_localization_at_k`, `ips`,
`per_channel_metrics`, `joint_micro_metrics` e il blocco `cf_*`: i primi quattro sono gated su C≥2
(`metrics_core._is_univariate`) e questi dataset sono **univariati per costruzione**, quindi `detect()`
stesso li salta; `cf_*` richiede `test_clean/`+`test_mask/`, assenti su wsd_fed e ucr_split. Il floor
calcola quindi **esattamente** l'insieme che `detect()` calcola su questi dati, niente di meno.

**Costo.** Il blocco `_suite` costa ~10× il blocco flat (PATE ~5.5 s a chiamata, la suite la chiama ~8
volte): ~110 s/client contro ~10 s. `--suite on` (default) per gli arm che finiscono nel paper, `--suite off`
per gli sweep di robustezza, che restano comunque **completi sul flat**. Ogni riga dichiara in
`_metrics_omitted` quali metriche non erano finite, così una colonna assente non può essere scambiata per una
colonna completa (su `ucr_split` 9/1130 righe non hanno `affiliation_f1`).

**Disciplina sull'headline.** Solo il blocco threshold-free è comparabile: gli arm deep usano la regola per-τ
del paper, FLOOR necessariamente il fallback a quantile. `f1`, `precision`, `recall`, `fpr`,
`affiliation_f1`, `pate`, `detection_delay_mean` vanno in appendice con quel caveat in didascalia. `auroc`
solo accanto a `floor_random` (satura ai tassi di anomalia wsd 0.0005–0.0344). `cf_*` (RQ4) omesso: richiede
`test_clean/`+`test_mask/`, assenti su wsd_fed e ucr_split.

**Impulse.** `--impulse on` è l'impostazione **decidente** (matched a ogni arm deep, `config.py:298`);
`--impulse off` è risultato secondario di prima classe (§0.6). `_impulse` stampigliato su ogni record.

**Percorsi artefatti (come implementati).**

    artifacts/floor/records_<ds>.jsonl        # una riga per (arm, entity); riscritto DE-DUPLICATO a fine run
    artifacts/floor/<ds>__<arm>.json          # {meta, summary, records} per arm
    artifacts/floor/witness_<ds>.json         # residui di esattezza per (arm, cluster) + verdetto
    artifacts/floor/<ds>_all_models.csv       # floor_table.py: tidy floor + ogni albero deep
    artifacts/floor/<ds>_matrix.json          # floor_table.py: client × modello × metrica

`--emit-fed-tree` **non è implementato** (equivale a OFF permanente): scriverebbe in
`artifacts/fed_eval/<ds>/...`, il glob esatto che `scripts/aggregate_all.py:30` percorre, e quelle righe
portano una **regola di soglia diversa** — entrerebbero silenziosamente in ogni tabella dell'unico albero
fidato.

Schema record (`_schema: 2`): `{_arm, _head, _mode, _knobs, _cluster, _entity, _seed, _impulse,
_threshold_rule, _threshold, _unit, _n_windows, _n_fit_windows, _fit_stride, _eval_stride, _window,
_tolerance, _pooled_path, _param_relerr, _score_relerr, _excess_objective, _rank_deficient, _secs,
_metrics_omitted}` + tutte le metriche flat di Step 7 + `_suite` opzionale. Metriche non finite **omesse, mai
NaN** (`federated_eval.py:1547-1551`) **ma elencate in `_metrics_omitted`**. `_unit = cluster` su
`ucr_split`, così 1130 righe non possono essere mediate come 5× pseudo-repliche.

⚠️ **`_window` è parte dell'identità dell'arm.** La finestra fissa *sia* la geometria di accumulo *sia* la
lunghezza della MA dell'impulse term: un floor misurato a `W=128` **non è confrontabile** con arm deep
girati a un'altra finestra. `--window` cambia il tag dell'arm (`__w<W>`) proprio per impedire la collisione
silenziosa su resume. Su `ucr_split` questo è **operativo, non teorico**: il floor su disco è a `W=128`
perché a `W=128` sono gli arm deep su disco; se quelli vengono rifatti a `2×periodo`, il floor va rifatto.

---

## 4. File — stato reale

**Aggiunti:**

| Path | Stato |
|---|---|
| `scripts/floor_heads.py` | ✅ 7 teste + `fedprox_round`/`fedlocalgd`, zero import dal repo, `__main__` = 17 invarianti |
| `scripts/floor_eval.py` | ✅ 11 modi, witness, suite metrica completa, resume de-duplicato, `--selftest` |
| `scripts/run_floor.sh` | ✅ le batch A–F in ordine di dipendenza, resumable, log in `logs/floor/` |
| `scripts/floor_table.py` | ✅ (esisteva) ora **produce** `<ds>_matrix.json`, prima orfano; legge i `report.json` deep per avere anche top-K/event/best_f1 dal lato deep |
| `scripts/floor_selftest.py` | ❌ non serve: gli invarianti sono in `floor_heads.__main__` + `floor_eval --selftest` |
| `scripts/floor_stats.py` | ✅ Wilcoxon appaiato + CI bootstrap + TOST opzionale, `--unit entity\|cluster` (auto `cluster` su `ucr_split`), stampa il margine minimo per l'80% di potenza. Resta **non pre-registrata la scelta del margine** TOST di §6 (misurato: 0.045–0.062 su ucr, 0.102–0.210 su wsd). |

**Toccati dopo i numeri:** `federated_method.tex` — riga `Moving-average (10 min)` riempita (`0.442`,
`\multicolumn` sulle due colonne perché la testa è budget-invariante), §metrics e §results riscritte, punto
(i) di "Scope and honest negatives" riscritto. **Ancora da fare:** `documentation/RESEARCH_LEDGER.md`
(nuova voce + verdetto di fiducia).

**Da NON toccare:** nulla sotto `pipeline/`. Nessun nuovo ramo `elif arm ==`: `federated_eval.py:1489` fa
`state_dict`-save della coppia restituita e `cf_fidelity_from_ckpts:750` la ricarica con `strict=True`,
quindi un modello in forma chiusa dovrebbe fabbricare uno state_dict completo Stage1VQVAE+MaskGIT. Nemmeno
`scripts/summarize_converged.py` (usare `floor_stats.py` invece di allargare un file da cui dipendono le
tabelle fidate). Nessun codice di esperimento in esecuzione.

---

## 5. CLI (verificata)

> 📘 **I comandi operativi completi — wsd, ucr (tutte / prime 10 / le 82 selezionate), i 5 toy,
> `ucr_ad`/`ucr_pool`, sharding, post-processing e costi misurati — stanno in
> [`FLOOR_RUNBOOK.md`](FLOOR_RUNBOOK.md).** Qui sotto resta la forma minima di ogni comando.

    # tutto, in ordine di dipendenza, resumable
    bash scripts/run_floor.sh            # wsd (A-E) poi ucr_split (F)
    bash scripts/run_floor.sh wsd

    # gli invarianti, prima di credere a qualunque numero
    python scripts/floor_heads.py                     # 17 invarianti, standalone
    python scripts/floor_eval.py --selftest           # + i 2 contratti verso il repo

    # per-client, tutte le teste di decisione
    python scripts/floor_eval.py --dataset wsd_fed --clusters all \
        --heads ma_c,ma_causal,ar,pca --modes local --impulse on --jobs 4

    # percorso pooled INDIPENDENTE + gate di esattezza (fallisce su uno zero esatto)
    python scripts/floor_eval.py --dataset wsd_fed --heads ar,pca,gauss \
        --modes central,fed_exact --witness --witness-tol 1e-9

    # la cella portante: perdita di aggregazione (J(ŵ)-J(w*) finisce in _excess_objective)
    python scripts/floor_eval.py --dataset wsd_fed --heads ar,pca \
        --modes fed_fedavg,fed_fedavg_uniform

    # l'arm ex-controllo-positivo (§0.5): misura, non dimostra
    python scripts/floor_eval.py --dataset wsd_fed --heads pca --modes fed_naive,fed_naive_aligned

    # l'asse round × passi-locali (rispecchia gli arm deep veri)
    python scripts/floor_eval.py --dataset wsd_fed --heads ar --modes fed_localgd \
        --tau 16 --rounds 30                      # --eta assente = passo automatico 1/λ_max

    # diversità-vs-quantità
    python scripts/floor_eval.py --dataset wsd_fed --heads ar,pca --modes central_capN --cap-clients 2

    # sweep di k e ablazione dell'impulse
    python scripts/floor_eval.py --dataset wsd_fed --heads ma_c --k 20 --suite off
    python scripts/floor_eval.py --dataset wsd_fed --heads ma_c --impulse off

    # la matrice di confronto contro ogni albero deep
    python scripts/floor_table.py --dataset wsd_fed

⚠️ `--tau 1,4,16,64` e `--cap-clients 1,2,3,5,11` del design **non** sono liste: un valore per run (il tag
dell'arm porta il valore, quindi i run si accumulano senza collidere).

---

## 6. Regole di decisione pre-registrate

**Selezione della testa: su val, non su test.** L'headline è la testa **selezionata su val** per cluster
(criterio label-free: MSE one-step / MSE di ricostruzione su val), con il *massimo su tutte le teste*
pubblicato accanto ed etichettato esplicitamente come **inviluppo superiore**. Il massimo su 4 teste è un
oracolo su test set: distorto verso l'alto e una forking path che la pre-registrazione non neutralizza.

**Margine TOST: da stimare, non da indovinare.** Va derivato dalla SD delle differenze appaiate FLOOR-vs-deep
misurata nel pilota, non ereditato dalla banda 0.11–0.15 del ledger (che è una MDE local-vs-centralized: SD
appaiata 0.129, SE 0.0232 su n=31). Le SD appaiate misurate contro `movavg10` sono **0.20–0.24** per gli arm
buoni e **0.32–0.42** per gli `enc_*` (§0.3) ⇒ margini diversi per gruppo, e la potenza raggiunta va
dichiarata. Un margine troppo largo *fabbrica* meccanicamente l'esito "equivalente".

**L'albero deep va nominato.** Per ogni arm presente in più alberi, pubblicare lo stesso Δ contro **entrambi**
(§0.2). Il Δ deep è a **n_seeds=1** e la sua incertezza **esclude** la varianza di training: da dichiarare.
`floor_stats.py` deve pretendere un'attestazione di non-troncamento per (cluster, entity), perché il verdetto
TRUNCATED/NOT-CONVERGED va solo su stdout (`pipeline/federated.py:1664,:1952`), non nei JSON.

**Righe per-cluster: descrittive, mai inferenziali.** I cluster wsd sono c0=5, c1=11, c2=9, c3=6. Un Wilcoxon
a due code con n=5 ha p minimo raggiungibile 0.0625: non può *mai* toccare α=0.05. Righe per-cluster =
mediane + dot plot per entità + `n_k`, senza p-value. Ogni claim inferenziale al livello pooled n=31, con il
cluster come covariata di stratificazione.

**Esiti, e cosa significano:**

| Esito | Significato | Già osservato? |
|---|---|---|
| FLOOR **perde** contro un arm | quell'arm giustifica il suo costo | sì: `cb_ema`, `local`, `centralized` (p ≤ 0.005) |
| FLOOR **pareggia** un arm federato | quell'arm non ha valore dimostrabile su wsd; va riportato come tale, non come "promettente" | sì: `enc_fedproto`, `enc_commoninit`, `enc_fedprox`, e `cb_only` in mediana |
| FLOOR **batte** un arm | risultato negativo pubblicabile sull'arm | sì: `enc_fedavg` (−0.063 in mediana) |
| `fed_exact == central` entro tolleranza | l'asse perdita-di-aggregazione ha uno zero valido; ogni Δ deep `central−fed` è attribuibile all'ottimizzazione | sì: 1e-12…1e-14 (§0.4) |
| `fed_exact != central` | bug nel percorso pooled o nell'accumulo — **fermarsi**, non pubblicare | — |
| `fedavg ≈ fed_exact` in metrica, con gap parametrico | la media in coordinate canoniche non fa danno | sì (§0.5b), 18 confronti su 3 teste, zero significativi |
| `fed_naive` ≪ `fed_fedavg` sullo stesso input | il collasso weight-space è causato dal **gauge**, non dalla media | sì (§0.5): 0.1154 vs 0.4471, p<1e-4, 4/4 cluster |
| `floor_fed_scaleonly != floor_local` | leak nell'asse metrico — **fermarsi** | — |

**L'esito scomodo, dichiarato in anticipo.** Se dopo il set di teste completo la testa a **zero parametri**
risultasse indistinguibile anche da `local`/`centralized`/`cb_ema`, la conclusione da pubblicare è che su
`wsd_fed` il compito non discrimina i modelli, e ogni claim relativa sul dataset va ritirata. I numeri
attuali **non** dicono questo — le tre barre buone battono il floor con p ≤ 0.005 — ma la regola va fissata
prima, non dopo.

---

## 7. Ordine di implementazione

| # | Step | Definition of done (asserito, non ispezionato) |
|---|---|---|
| 1 | `floor_heads.py` + `floor_selftest.py`, `random` e `ma_c` | `ma_c` riproduce `detect._moving_average_paper` bit-identico su input casuali; `random` seed-riproducibile. |
| 2 | `floor_eval.py` percorso `local`, un client | Riproduce un numero del prototipo su `kpi_015` **calcolato con la stessa regola di accumulo di §3** (finestra intera, nessuno `skip`) — pinnare il valore atteso *dopo* averlo rigenerato, non prenderlo da log con geometrie diverse. |
| 3 | `local` su tutti i 31 client, 4 teste | Riproduce §0.1 entro 1e-6. |
| 4 | `central` via **percorso pooled indipendente** | La design matrix pooled è materializzata, non ricostruita da somme. |
| 5 | `fed_exact` + witness | `‖θ_fed − θ_central‖/‖θ_central‖ < 1e-9` **e** `max|Δ score|/max|score| < 1e-9`, per ogni cluster e ogni head additiva. Uno zero *esatto* fa **fallire** il gate: significa che i due percorsi condividono un oggetto (§0.4). |
| 6 | `fed_fedavg` + `fed_fedavg_uniform` + `J(ŵ)−J(w*)` | Il gap parametrico è non nullo e riportato; il Δ metrico ha il suo CI. |
| 7 | `fed_naive` (controllo positivo) | ✅ **Collassa**: 0.1154 vs `fed_fedavg` 0.4471, p<1e-4, tutti e 4 i cluster (§0.5). ⚠️ Il fixture del selftest deve essere **eterogeneo**: a sottospazi identici il gauge è innocuo e il controllo sembra fallire. |
| 8 | `fed_scaleonly` (regression test) | Δ **esattamente** 0 contro `floor_local`. |
| 9 | `fed_localgd`, `fed_prox`, `central_capN` | Limiti verificati: `μ=0 → local`, `μ→∞ → global`, `τ→1 → pooled`. |
| 10 | `floor_stats.py` + questo file congelato | Pre-registrazione committata **prima** del run finale. |
| 11 | `ucr_split` (226 cluster × 5, IID per costruzione) | `_unit = cluster`. Qui `centralized − federated` è perdita di aggregazione pura ⇒ il test più netto per `fed_exact`. |

**Pinnare lo stride di fit nella pre-registrazione.** L'analisi di rango cambia radicalmente con lo stride: su
`ucr_split` a stride 1 sono 60/1130 i client con `n < 2W` e 0 con `n < W`; a stride 13 diventano 734/1130 e
546/1130 (covarianza strettamente singolare). Su wsd il minimo è 2862 finestre a stride 1 ma 221 a stride 13.
**Raccomandazione: fittare a stride 1**, che è ciò che fa il training deep (`federated.py:1754`:
`c.n_windows = len(s1c.data.train_dataset)` con `window_stride=1`) e quindi rende coerenti anche i pesi
FedAvg `n_k`. Lo *scoring* resta a stride 13 come `detect`. `_fit_stride` va nello schema record.

---

## 8. Rischi e cosa non fare

1. **Non presentare `fed_exact` come un algoritmo federato nuovo.** È `k-FED`/Prop.1, già nel ledger come
   "non novel". Il suo valore è di *strumento di misura*: lo zero dell'asse.
2. **Non usare `floor_ema_shard` per spiegare `cb_ema`.** Errore di categoria (§2.2).
2b. **Non falsificare un meccanismo su un fixture omogeneo.** Il 2026-07-28 ho dichiarato morto il controllo
   positivo di §0.5 sulla base di tre fette dello stesso segnale — dove i client condividono il sottospazio e
   il gauge è per costruzione irrilevante. Su dati veri il controllo scatta con Δ = −0.33. Se un meccanismo
   vive nell'eterogeneità, il fixture che lo testa deve essere eterogeneo.
3. **Non usare `floor_fed_prox` per dichiarare chiuso il null di FedProx deep.** Dice solo che l'obiettivo
   convesso è ben comportato.
4. **Non emettere nell'albero `fed_eval` per default.** Regola di soglia diversa.
5. **Non chiamare arm `local`/`centralized`/`fa_*`.** Collide con `summarize_converged.py:64-65` e
   `aggregate_all.py:30`.
6. **Non far girare a >4 worker mentre `federated_eval.py` è vivo.** Il muro è la CPU a 16 core.
7. **Non riportare un singolo numero come "il floor".** È un intervallo per testa (0.35–0.51).
8. **La regola "train to convergence" non si applica** — non c'è training. Ma si applica al *lato deep* del
   Δ: usare solo arm attestati non troncati.
9. ~~**Il prototipo vive in una directory tmp di sessione** e verrà perso.~~ **È stato perso** (2026-07-28,
   `scripts/_floor_proto/` non è mai stato creato). Riscritto da zero in `scripts/floor_heads.py`. Regola
   che ne deriva: **nessun numero entra in questo documento se non esiste in-repo lo script che lo
   rigenera.** Ogni tabella qui sotto porta ora il comando che la produce.
10. **Non paragonare un floor a `W=128` con arm deep a un'altra finestra.** `W` fissa la geometria di
   accumulo *e* la MA dell'impulse term. Il tag dell'arm porta `__w<W>` proprio per rendere impossibile la
   collisione silenziosa.
11. **Non leggere `paper_top1/3/5` come metrica di ranking su wsd**: mediana 1.00 per ogni arm deep. È un
   controllo di sanità (il massimo assoluto cade dentro un'anomalia), non un discriminante.
12. **Non mediare le 1130 righe di `ucr_split` come repliche.** Le 5 shard di un cluster condividono il test
   set e la testa è invariante a riscalamenti positivi ⇒ le metriche threshold-free coincidono entro 1e-8
   (verificato su 226/226 cluster). L'unità è il cluster: n=226, non 1130. Solo `_threshold` e le metriche
   dipendenti dalla soglia variano davvero fra shard.

---

## 8b. I nove difetti trovati nell'audit del 2026-07-28 e cosa è stato fatto

| # | Difetto | Risoluzione |
|---|---|---|
| 1 | Del design esisteva solo `ma_c`: nessuna delle teste fittate, nessun modo federato, prototipo perso | `floor_heads.py` (7 teste + prox/localgd) e `floor_eval.py` (11 modi + witness) riscritti da zero; §0.1/§0.4/§0.5 tornano rigenerabili |
| 2 | §0.6 e §0.1b citavano numeri del prototipo, non riproducibili | batch B (impulse off) e C (sweep di k) di `run_floor.sh` li rimisurano con la geometria canonica; i numeri vecchi sono marcati |
| 3 | `head_random` seedata con `hash()` → **non** riproducibile fra processi, docstring falsa | seed via `zlib.crc32` su una chiave stabile; invariante asserito nel selftest |
| 4 | Il floor su `ucr_split` è agganciato a `W=128` senza che nulla lo dichiari | `--window` + `__w<W>` nel tag dell'arm + `_window` nello schema + avviso in §3 e §8.10 |
| 5 | 9/1130 righe senza `affiliation_f1`, indistinguibili da righe complete | campo `_metrics_omitted` su ogni riga |
| 6 | `_event_metrics` mai chiamato (il design lo prevedeva) | chiamato in `full_metrics`, insieme a top-K, blocco macro e suite unificata |
| 7 | `--out-dir` fuori dal repo → crash su `Path.relative_to` a fine run | `rel_repo()`; `--out-dir` assoluto supportato |
| 8 | `federated_method.tex` portava ancora `\pending` sulla riga della baseline | riga riempita (`0.442`, `\multicolumn` sulle due colonne) + §metrics/§results/§negatives riscritte |
| 9 | `wsd_fed_matrix.json` orfano: nessuno script lo produceva | `floor_table.py` è ora il produttore, e legge i `report.json` deep per avere anche top-K/event/best_f1 dal lato deep |

---

## 9. Provenienza

Design prodotto da un panel: 3 design indipendenti (SLAB / FLOOR / BaselineZoo), 3 giudici (valore
scientifico → BaselineZoo; fattibilità ingegneristica → FLOOR; reviewer ostile → FLOOR), sintesi su spina
FLOOR con innesti dagli altri due, poi 2 critici adversariali con verifica su disco (13 blocking). I numeri
di §0 sono stati ri-verificati direttamente contro `artifacts/converged_all`, `artifacts/converge60`,
`artifacts/fed_eval/wsd_fed` e i log del prototipo; le correzioni dei critici su conteggio eval (90, non
121), stride di fit, margine TOST, ambiguità dell'albero deep e circolarità del check di esattezza sono
recepite nel testo.
