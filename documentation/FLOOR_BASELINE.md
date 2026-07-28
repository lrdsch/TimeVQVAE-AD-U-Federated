# FLOOR — baseline closed-form: sanity floor + federazione esatta

**Stato:** design / pre-registrazione. Le sezioni §0–§2 riportano numeri **già misurati** su `wsd_fed` reale
attraverso il vero percorso `detect`; §3–§8 sono il progetto da implementare.

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

| head | parametri fittati | median `vus_pr` | mean |
|---|---|---|---|
| `movavg10` — box centrato k=10, via `detect._moving_average_paper` | **0** | **0.5054** | **0.4421** |
| `pca8_local` — PCA su finestre, K=8 | 128·8 | 0.4277 | 0.4228 |
| `pca8_fedexact` | idem | 0.4249 | 0.4268 |
| `ar32_local` — ridge AR(32) causale | 33 | 0.3597 | 0.3718 |
| `ar32_fedexact` | idem | 0.3557 | 0.3694 |
| `ar32_fedavg` | idem | 0.3479 | 0.3721 |
| `random` | 0 | 0.008 | 0.011 |

> **Geometria canonica.** La riga `movavg10` sopra (0.4421 mean / 0.5054 median) è stata rigenerata
> con la regola di accumulo di §3 su tutti i 31 client — **questo è il numero da mettere nel paper**.
> Le altre righe sono ancora quelle del prototipo (geometria non canonica) e vanno rigenerate.

### 0.1b Sensibilità a k (sweep a posteriori, n=31, geometria canonica)

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

### 0.3 Il risultato che decide: FLOOR (`movavg10`) vs ogni arm deep

Wilcoxon signed-rank appaiato su `(cluster, entity)`, n=31, α=0.05.

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

| head | `param relerr` | `score relerr` |
|---|---|---|
| AR(32) | 1.89e-12 | 2.37e-12 |
| PCA(8) | 1.51e-13 | 5.74e-14 |
| Gauss diag | 0.00e+00 | 1.75e-14 |

Confronto **non circolare**: statistiche calcolate sulle finestre **concatenate** (`stats(pooled)`) contro la
**somma** delle statistiche per-client (`Σ_k stats_k`), poi risolte e scorate indipendentemente. Le
magnitudini 1e-12…1e-14 sono arrotondamento float64 — la firma corretta di una verifica reale.

> ⚠️ **Trappola da non ripetere.** Un prototipo precedente riportava `max|Δvus_pr| = 0.00e+00` tra
> `fed_exact` e `centralized`. Era **circolare**: entrambi gli arm ricevevano lo stesso oggetto `w_central`,
> già calcolato sommando statistiche. Quel run dimostra determinismo della pipeline, non esattezza della
> federazione. Uno zero *esatto* invece di ~1e-13 è il campanello d'allarme. Il gate di §7 richiede un
> **percorso pooled indipendente**.

### 0.5 Il contrasto meccanicistico (il contributo positivo)

`ar32_fedavg` 0.3479 vs `ar32_fedexact` 0.3557: media dei pesi ≈ soluzione esatta **a livello di metrica**,
mentre i parametri differiscono del **15–33% relativo** (`‖w_central−w_fedavg‖/‖w_central‖`: 0.143/0.564 su
c0, 0.102/0.308 su c1, 0.075/0.497 su c2).

Nel caso deep lo stesso repo misura `NaiveAvg` 0.048 contro `Ensemble` 0.326 su chiavi identiche
(`documentation/MATCHED_FUSION_FINDINGS.md`). Su un modello **convesso** la media dei pesi non costa quasi
nulla; su quello deep distrugge tutto. ⇒ **il collasso weight-space è una proprietà del modello deep**
(simmetria di permutazione/segno di codebook+encoder), **non della media in sé**.

Questo trasforma la baseline da semplice pavimento a contributo positivo. Serve però un **controllo positivo**
(`floor_fed_naive`, §2): un modello convesso a cui si rompe deliberatamente l'allineamento segno/permutazione
*deve* collassare. Senza, resta un'inferenza da un null.

### 0.6 L'impulse term vale più della baseline

Su `kpi_015`, head `ma10`: impulse OFF 0.3742 → ON 0.7875. Su AR: 0.306 → 0.719. Il post-processing di
`detect` (`0.5·(raw + _moving_average_paper(raw,128))`, `config.py:298`) vale fino a **+0.41 VUS-PR a un
filtro da 8 parametri**. Nessuno nel progetto l'ha mai misurato: riformula il significato di ogni numero
assoluto riportato finora e va pubblicato come risultato secondario di prima classe.

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
| `floor_fed_naive` | la patologia `NaiveAvg` deep | matrici `U_K` grezze (solo PCA/Gauss) | media **senza** allineamento segno/permutazione, poi ri-ortonormalizza | NO — **deve collassare** | **Controllo positivo.** Un modello convesso che collassa quando si rompe l'allineamento inchioda il collasso deep alla simmetria, non alla media. |
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

`ma_c`, `ma_causal`, `random` hanno **zero parametri fittati** ⇒ local == central == ogni modo federato,
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

**Step 7 — metriche.** `preds = (scores > thr).astype(np.int64)`, poi

    rep = {}
    rep.update(detect._detection_metrics(labels, preds, scores, cfg))   # detect.py:831
    rep.update(detect._event_metrics(labels, preds))

Il buffer VUS/PATE è letto da `cfg.evaluation.paper_metrics_tolerance`. Output = le chiavi flat che legge la
tabella RQ1: `vus_pr`, `auprc`, `auroc`, `pate_f1` = `federated_eval.METRIC_KEYS`
(`federated_eval.py:58`).

**Disciplina sull'headline.** Solo il blocco threshold-free è comparabile: gli arm deep usano la regola per-τ
del paper, FLOOR necessariamente il fallback a quantile. `f1`, `precision`, `recall`, `fpr`,
`affiliation_f1`, `pate`, `detection_delay_mean` vanno in appendice con quel caveat in didascalia. `auroc`
solo accanto a `floor_random` (satura ai tassi di anomalia wsd 0.0005–0.0344). `cf_*` (RQ4) omesso: richiede
`test_clean/`+`test_mask/`, assenti su wsd_fed e ucr_split.

**Impulse.** `--impulse on` è l'impostazione **decidente** (matched a ogni arm deep, `config.py:298`);
`--impulse off` è risultato secondario di prima classe (§0.6). `_impulse` stampigliato su ogni record.

**Percorsi artefatti.**

    artifacts/floor/<ds>/<cluster>/seed<N>/<arm-tag>/<entity>/report.json
    artifacts/floor/<ds>/<cluster>/seed<N>/<arm-tag>/<entity>/scores.npz   # train/test_scores, test_labels
                                                                          # + eval_buffer, eval_q, eval_pot_level, dataset_name
    artifacts/floor/<ds>_<cluster>.json           # {meta, summary, records}, forma federated_eval.py:1543-1575
    artifacts/floor/records_<ds>.jsonl            # righe flat append-only (convenzione naiveavg_eval.py:135-138)
    artifacts/floor/witness_<ds>.json             # residui di esattezza per cluster + gate
    artifacts/floor/suffstat_<ds>_<cluster>.npz   # statistiche riusabili (ogni head futuro rifitta senza dati)

`--emit-fed-tree` (**default OFF**) scriverebbe anche in `artifacts/fed_eval/<ds>/...`, il glob esatto che
`scripts/aggregate_all.py:30` percorre. Off per default perché quelle righe portano una **regola di soglia
diversa** ed entrerebbero silenziosamente in ogni tabella dell'unico albero fidato.

Schema record: `{_arm, _head, _mode, _knobs, _cluster, _entity, _seed, _impulse, _threshold_rule, _unit,
_n_windows, _rank_deficient, _fit_stride, vus_pr, auprc, auroc, pate_f1, affiliation_f1, f1}`. Metriche non
finite **omesse, mai NaN** (`federated_eval.py:1547-1551`). `_unit = cluster` su `ucr_split`, così 1130 righe
non possono essere mediate come 5× pseudo-repliche.

---

## 4. File da aggiungere / toccare

**Aggiungere (6):**

| Path | Riga |
|---|---|
| `scripts/floor_heads.py` | Teste pure-numpy (`stats`/`solve`/`score`/`val_crit`, `additive`, `granularity`); **zero import dal repo**, testabile standalone. |
| `scripts/floor_eval.py` | Il framework: `_build_cfg`, costruzione arm per cluster (incluso il percorso pooled indipendente), accumulo detect-fedele + finalize + soglia + metriche, calcolo witness, writer, pool `--jobs`. |
| `scripts/floor_stats.py` | Statistiche matched a livello client: appaia record `floor_*` contro un arm deep nominato su chiavi `(cluster, entity)`, Wilcoxon n=31 + **TOST al margine pre-registrato** + CI bootstrap. Serve perché `fed_aggregate.py` testa a livello **seed** (inutile a n_seeds=1 per una baseline deterministica) e `summarize_converged.py:64-65` appaia solo `local`/`centralized` letterali. |
| `scripts/floor_selftest.py` | I cinque invarianti asseriti di §7 come entry point CI-style. |
| `scripts/run_floor.sh` | Job packing per (dataset, cluster), resumable (salta se `report.json` esiste), `--jobs` limitato, rifiuta di partire con >4 worker se un processo `federated_eval.py` è vivo. |
| `documentation/FLOOR_BASELINE.md` | **Questo file** — la pre-registrazione, committata *prima* del run finale. |

**Toccare solo dopo che i numeri sono a terra (3):** `federated_method.tex:279-281` (riempire la riga
`\pending`) e `:340-341`; `documentation/RESEARCH_LEDGER.md` (nuova voce + verdetto di fiducia).

**Da NON toccare:** nulla sotto `pipeline/`. Nessun nuovo ramo `elif arm ==`: `federated_eval.py:1489` fa
`state_dict`-save della coppia restituita e `cf_fidelity_from_ckpts:750` la ricarica con `strict=True`,
quindi un modello in forma chiusa dovrebbe fabbricare uno state_dict completo Stage1VQVAE+MaskGIT. Nemmeno
`scripts/summarize_converged.py` (usare `floor_stats.py` invece di allargare un file da cui dipendono le
tabelle fidate). Nessun codice di esperimento in esecuzione.

---

## 5. CLI

    # floor_local — riferimento per-client, tutte le teste di decisione
    python scripts/floor_eval.py --dataset wsd_fed --clusters all \
        --heads ma_c,ma_causal,ar,pca --modes local --impulse on --jobs 8

    # floor_central — percorso pooled INDIPENDENTE (il riferimento del witness)
    python scripts/floor_eval.py --dataset wsd_fed --clusters all --heads ar,pca \
        --modes central --pooled-path independent

    # floor_fed_exact — federazione a un round via statistiche sufficienti + gate di esattezza
    python scripts/floor_eval.py --dataset wsd_fed --clusters all --heads ar,pca \
        --modes fed_exact --witness --witness-tol 1e-9

    # la cella portante: perdita di aggregazione, con quantificatore in forma chiusa
    python scripts/floor_eval.py --dataset wsd_fed --heads ar,pca \
        --modes fed_fedavg,fed_fedavg_uniform --report-excess-objective

    # controllo positivo: DEVE collassare
    python scripts/floor_eval.py --dataset wsd_fed --heads pca --modes fed_naive

    # l'asse round × passi-locali (rispecchia gli arm deep veri)
    python scripts/floor_eval.py --dataset wsd_fed --heads ar --modes fed_localgd \
        --tau 1,4,16,64 --rounds 30 --eta auto

    # curva diversità-vs-quantità a due manopole
    python scripts/floor_eval.py --dataset wsd_fed --heads ar,pca --modes central_capN \
        --cap-clients 1,2,3,5,11 --cap-windows fixed

    # analisi matched contro una barra deep NOMINATA
    python scripts/floor_stats.py --dataset wsd_fed \
        --deep-arm local --deep-tree artifacts/converged_all \
        --floor-head val_selected --tost-margin <da §6>

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
| `fedavg ≈ fed_exact` in metrica, con gap parametrico | il collasso weight-space è del modello deep, non della media | sì (§0.5), da confermare col controllo positivo |
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
| 7 | `fed_naive` (controllo positivo) | Collassa. Se non collassa, il meccanismo di §0.5 non è pubblicabile. |
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
3. **Non usare `floor_fed_prox` per dichiarare chiuso il null di FedProx deep.** Dice solo che l'obiettivo
   convesso è ben comportato.
4. **Non emettere nell'albero `fed_eval` per default.** Regola di soglia diversa.
5. **Non chiamare arm `local`/`centralized`/`fa_*`.** Collide con `summarize_converged.py:64-65` e
   `aggregate_all.py:30`.
6. **Non far girare a >4 worker mentre `federated_eval.py` è vivo.** Il muro è la CPU a 16 core.
7. **Non riportare un singolo numero come "il floor".** È un intervallo per testa (0.35–0.51).
8. **La regola "train to convergence" non si applica** — non c'è training. Ma si applica al *lato deep* del
   Δ: usare solo arm attestati non troncati.
9. **Il prototipo vive in una directory tmp di sessione** e verrà perso. Ogni numero citato qui è inline
   proprio per questo. Se serve l'evidenza eseguibile, copiare
   `proto_floor.py`/`proto2.py`/`pilot2.py`/`align.py`/`exactfed_*.py`/`floor.log`/`exactfed_results.json` in
   `scripts/_floor_proto/` prima che tmp venga ripulita.

---

## 9. Provenienza

Design prodotto da un panel: 3 design indipendenti (SLAB / FLOOR / BaselineZoo), 3 giudici (valore
scientifico → BaselineZoo; fattibilità ingegneristica → FLOOR; reviewer ostile → FLOOR), sintesi su spina
FLOOR con innesti dagli altri due, poi 2 critici adversariali con verifica su disco (13 blocking). I numeri
di §0 sono stati ri-verificati direttamente contro `artifacts/converged_all`, `artifacts/converge60`,
`artifacts/fed_eval/wsd_fed` e i log del prototipo; le correzioni dei critici su conteggio eval (90, non
121), stride di fit, margine TOST, ambiguità dell'albero deep e circolarità del check di esattezza sono
recepite nel testo.
