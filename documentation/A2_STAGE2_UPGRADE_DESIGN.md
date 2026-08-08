# A2 → stage-2 upgrade: FedSGD(τ) · FedAdam · FedAvgM · FedProx-sul-prior

**Studio di design, 2026-08-07.** Obiettivo: migliorare A2 cambiando SOLO l'aggregazione dello
stage 2 (il prior). Lo stage 1 — encoder FedAvg + BN pooled + codebook suffstat — resta
byte-identico in ogni variante. Fonte: lettura integrale di `federated_stage2` e della
dispatch, più i log misurati di `zn_a2`. Ogni claim ha file:riga.

> 🔴 **LEGGERE PRIMA LA §7 (verifica avversariale, stesso giorno).** L'audit del valore
> atteso ha bocciato la premessa «ottimizzare meglio il prior ⇒ detection migliore» sui
> nostri stessi dati, e ha riordinato tutto il piano: prima i DIAGNOSTICI (gate
> fedtok-centralprior, fix dell'oracolo val, ctrl strumentato), poi — solo se i gate
> passano — le varianti. FedAvgM β=0.9 è NO_GO. Le onde della §5 sono SUPERATE dalla §7.

---

## 0. La tassonomia — sono quattro meccanismi diversi su TRE leve ortogonali

| leva | metodo | cosa cambia | costo vs A2 |
|---|---|---|---|
| regime del client | **FedSGD(τ)** | τ passi di optimizer per round invece di 10 epoche | round ↑ molto per τ piccoli |
| regolarizzatore del client | **FedProx sul prior** | `+ (μ/2)‖w−w^t‖²` nella loss locale | uguale |
| ottimizzatore del server | **FedAvgM** / **FedAdam** | momentum / Adam sullo pseudo-gradiente `Δ` | uguale |

- **FedAvgM e FedAdam sono mutuamente esclusivi** (`if server_opt=="fedadam": … elif
  server_momentum>0:` — federated.py:2132/2139). Se li imposti entrambi, **fedadam vince in
  silenzio e il banner FedAvgM stampa lo stesso** (2088-2091): il log mentirebbe.
- **τ compone liberamente** con entrambi i rami server (la scelta client a 2127-2128 è
  indipendente dall'if/elif del server). τ + FedAdam è la coppia raccomandata dal commento
  stesso del codice (2063-2064).
- **FedProx è ortogonale a tutto** e oggi NON esiste per lo stage 2: è codice nuovo
  (~120-150 righe, 2 file), non plumbing.

A2 oggi = client libero (10 epoche) + server rimpiazzo secco con la media pesata
`n_windows`. Il punto più naïf di entrambi gli assi.

## 1. Semantica esatta di ciò che è GIÀ implementato

### FedSGD(τ) — `tau_steps` (federated.py:2006-2033, dispatch 2127-2128)
- τ>0 ⇒ `_local_train_prior_steps`: esattamente τ passi AdamW per round; `--local-epochs`
  viene **ignorato in silenzio**.
- Lo stream di minibatch è un generatore **persistente sul client** (`_cycle`, 2013-2016):
  i round non riavviano l'epoca — «τ=1 ≈ un minibatch centralizzato grande per
  aggregazione».
- **Le teste locali steppano anche loro** (l'AdamW è su TUTTI i parametri, 1967): a τ=1
  `output_bias` prende 1 passo Adam per round — la capacità della testa si accoppia al
  numero di round.
- ⚠️ L'arm `federated_fedsgd` esistente NON è un precedente utilizzabile: ha
  `local_prefixes=()` (prior interamente condiviso, zero teste) e stage 1 cb-only
  (federated_eval.py:1854-1863). Copiarne la dispatch cambierebbe lo stage 1 di A2.

### FedAdam — `server_opt="fedadam"` (federated.py:2097-2105, 2132-2138)
```
Δ = media_client − x          (convenzione "ascent verso i client")
m = 0.9·m + 0.1·Δ ;  v = 0.99·v + 0.01·Δ²
x = x + server_lr · m/(√v + 1e-3)          ← x SOSTITUISCE la media nel broadcast
```
- b1/b2/eps **hardcoded** (2103), niente bias-correction (= FedOpt, Reddi et al.).
- Il passo per-coordinata satura a ~`server_lr` (sign-SGD-like): con `--server-lr 0.01` il
  server *insegue* la media, non la installa mai direttamente.

### FedAvgM — `server_momentum` (federated.py:2087-2091, 2139-2144)
```
δ = x − media_client          (convenzione DESCENT — segno OPPOSTO a FedAdam)
v = β·v + δ   (heavy-ball, SENZA damping (1−β))
x = x − v
```
- β=0 ⇒ bit-identico a FedAvg. β=0.9 ⇒ passo efficace fino a **10×** lo pseudo-gradiente
  (1/(1−β)) — overshoot lungo la direzione di drift persistente. Niente `server_lr` qui
  (implicito 1.0).

### Val, patience, restore (federated.py:2156-2213)
- Val **ogni round**, per client sul proprio `val_loader`, media UNIFORME (non pesata),
  **sotto autocast fp16** (491) — a differenza dello stage 1 che valida in fp32 (470-474).
  Rumore fp16 contro `min_delta=1e-4`: già oggi, ma diventa rilevante per τ piccoli.
- `best_states` = **prior COMPLETO per client** (corpo broadcast post-server-optimizer +
  teste dello stesso round — una coppia che è coesistita davvero, 2165-2171).
- Lo stato del server (`x, m, v`) è **perso a fine run**: mai checkpointato; lo stage 2 non
  ha resume (2196-2197).
- ⚠️ Asimmetria: `best_states` si aggiorna su `val < best_val` ma la patience si resetta
  solo su `val < best_val − min_delta` (2168-2171).

### LR e schedule dello stage 2
- AdamW `lr = cfg.training.lr = 1e-3` **costante, per sempre**: `Stage2ClientState.sched`
  non viene MAI assegnato — `enc_sched` non raggiunge lo stage 2. Un run τ=1 da migliaia di
  round si allena a 1e-3 fisso: la convergenza può venire solo dalla selezione su val.
- Hidden knob: l'AdamW dello stage 2 prende il **weight_decay di default torch = 0.01**
  (1967 non lo passa) — un secondo "prox" verso 0, non verso `w^t`. A μ≈0.01 loss-form i
  due sono numericamente della stessa forza.
- Staleness dei momenti Adam: il server sovrascrive il corpo ogni round ma i momenti
  `m, v̂` del client sopravvivono al salto (nessun reset, 2145-2146). A τ=1 è la dinamica
  dominante — di fatto un Adam a lungo orizzonte su una traiettoria perturbata dal server.
  Emergente, non progettato: da tenere d'occhio, non da "riparare" al buio.

## 2. Le TRAPPOLE (violarle produce run sbagliate in silenzio)

1. 🔴 **`--server-momentum` ha default argparse 0.9, NON 0** (federated_eval.py:1132-1133;
   il default della funzione è 0.0 a :675). Inoltrarlo alla cieca dal ramo enc-trio
   trasforma ogni rerun di A2 in FedAvgM β=0.9 senza che nulla lo dica. ⇒ o flag dedicati
   `--s2-*` con default 0, o forwarding condizionale esplicito.
2. 🔴 **L'out-json è cieco ai knob**: `cohort.py:256` scrive `<cl>__federated_enc_fedavg.json`
   per QUALSIASI variante. Stesso tag ⇒ launch.sh **salta** la cella (`-f` a :406) e
   spedisce i numeri della variante vecchia, mentre RUN.json viene riscritto coi flag
   nuovi: il manifest certificherebbe knob mai girati. ⇒ **UN TAG PER VARIANTE, sempre.**
   Il `cohort_fingerprint` non copre nessuno di questi knob (solo datasets+seeds).
3. 🔴 **La dir dei checkpoint oggi non distingue le varianti**: `_arm_tag` (980-1009) non
   emette bit per tau/server/pmu ⇒ due varianti condividerebbero
   `federated_enc_fedavg_bn-shared_prior-partial/` mischiando ckpt, `knobs.json` e
   `_fed_resume.pt` (che il resume adotterebbe cross-variante in silenzio). ⇒ estendere
   `_arm_tag` con bit terminali `tau{N}`, `srv-fedadam_slr{lr}`, `sm{β}`, `pmu{μ}[_dec]`,
   emessi SOLO se non-default (nessuna dir esistente cambia nome).
4. 🔴 **`FLAG_OWNER_ARMS` non conosce i 4 knob**: oggi `--tau-steps 8 --arms
   federated_cb_only` è accettato e ignorato. Aggiungere le entry ⇒ SystemExit standard
   (1442-1452) su ogni flag senza arm proprietario.
5. ⚠️ Guardie semantiche da aggiungere nel ramo enc-trio: rifiutare ogni knob stage-2 con
   `--fed-enc-prior local` (shared_keys=[] ⇒ federazione no-op vestita da trattamento; e
   τ>0 butta il run fuori dal percorso converged per-client a :730); rifiutare
   `fedadam + momentum>0` (precedenza silenziosa + banner bugiardo).
6. ⚠️ `fed_echo`/`knobs.json` va esteso con i knob nuovi o su disco nulla distinguerà le
   varianti.
7. ⚠️ `mean_loss` ritornata dai trainer deve restare la **task loss non penalizzata**
   (contratto stage-1 a 306-315), o le curve `[fed-s2] prior_loss` diventano incomparabili.

## 3. FedProx sul prior — progetto del port

- **I 55 shared key sono TUTTI parametri, zero buffer** (verificato: `MaskGITPrior3DPos`
  non ha `register_buffer`; 57 chiavi = 57 parametri, −2 teste locali = 55). L'ancora
  copre naturalmente solo il corpo: le teste sono fuori da `shared_keys` per costruzione.
- **Snapshot `w^t`**: in cima al round loop (dopo il `for r in range(rounds):` a 2123),
  per-client dal SUO modello (ref fp32 sul device del client, specchio di 1630-1636).
  Sotto FedAdam/FedAvgM l'ancora è il **broadcast post-server-optimizer** — corretto: è lo
  stato da cui i client partono davvero.
- **AMP**: prox calcolato FUORI da `_amp()`, in fp32, aggiunto alla loss autocast — è la
  scelta documentata dello stage 1 (349-352: «an fp16 reduction loses the low-order bits
  the penalty is made of»).
- **Telemetria** (solo loss-form): `scaler.unscale_(opt)` DOPO backward e PRIMA di step,
  poi `_prox_grad_ratio` (contratto a 1098); mai la ratio in decoupled (anti-monotona,
  376-380) — lì il diagnostico è `prox_pull_frac`.
- **Forma primaria: DECOUPLED.** La patologia del precondizionatore vale identica (stesso
  AdamW fused persistente, 1967 ≡ 299): in loss-form μ(w−w^t) viene diviso per √v̂ e μ
  piccolo è un no-op silenzioso, rilevabile solo a posteriori. La decoupled ha autorità
  calcolabile a priori: il knob vero è `lr·μ` per passo; a lr=1e-3 e 50-500 passi/round,
  μ=1 rimuove ~5-39% del drift per round, μ=10 ~39%+.
- **Griglie**: loss-form μ ∈ {0.01, 0.1, 1, 10} (mai attorno a un punto solo), gate
  `prox_grad_ratio ≥ 1e-2` **sui round tardivi** (nei primi round la ratio è inaffidabile,
  il denominatore collassa ~1000×); decoupled μ ∈ {1, 10} con `prox_pull_frac ≥ 5%`.
- **Diff**: ~120-150 righe su 2 file (`federated.py`: Stage2ClientState +campi, i due
  trainer +hook — meglio fattorizzare il corpo del passo in un helper condiviso —,
  `federated_stage2` +2 kwargs +snapshot +history; `federated_eval.py`: 2 argparse
  `--fed-prior-prox-mu/--fed-prior-prox-form`, forwarding, fed_echo, FLAG_OWNER_ARMS).
  μ=0 ⇒ percorso byte-identico a oggi (l'addizione è condizionale, non `+0.0`).

## 4. Budget — misurato dai log di `zn_a2`

Convergenza stage 2 (mediana client, passi al best round): coorte **S ≈ 8 265**; probe:
011=9 280 · 014=6 400 · 043=7 250 · 170=6 240. Outlier: 222=51 900 (6,3× la mediana).
Costo cella A2 misurato: s1 58% · s2 37% · detect 3% di 37,7 job-h/coorte.

**Val = 1,5-2% di un round-epoca ma DOMINANTE a τ=1** (un round τ=1 = 0,3 s di train vs
0,7-9,7 s di val). Il codice **non sa valutare meno spesso** (nessun knob di cadenza,
2159-2163) ⇒ serve `--fed-s2-val-every K` (aggregazione ogni round, selezione/patience solo
sui round di eval).

| τ | round necessari (011) | wall-clock vs A2 (val ogni round) | con val-every-k |
|---|---|---|---|
| 1 | 9 280 | ×4,1-6,2 | k=16 → **×1,65-1,87** |
| 4 | 2 320 | ×1,7-2,1 | k=4 → ×1,21-1,36 |
| 16 | 580 | ×1,1-1,2 | k=2 → ×0,95-1,10 |
| 64 | 145 | ~×1,0 | k=1 |
| 256 | 37 | ~×1,0 — quasi il regime a epoche | — |

**Patience**: 6 round-epoca ≈ 1 740 passi su 011; a τ=8 sarebbero 48 passi (36× più
stretta ⇒ CONVERGED prematuro garantito). Regola: patience in EVAL con passo-patience
costante ⇒ `patience_evals = ceil(1800/(τ·k))` a livello di coorte. Serve un
`--fed-s2-patience` separato: oggi UNA patience va a entrambi gli stage (723, 757) e
cambiarla romperebbe il contratto stage-1-byte-identico.

**Riuso dello stage 1**: `--resume-from <arm dir> --s1-rounds 0` esiste
(federated_eval.py:1212-1218) e i 10 `_fed_resume.pt` di zn_a2 sono su disco. MA lo
stage 2 ripartito non è bit-comparabile col zn_a2 originale (RNG consumato diverso alla
costruzione del prior) ⇒ **serve l'arm di controllo `zn_a2s2_ctrl`**: replica FedAvg via
resume. Se il ctrl riproduce zn_a2 entro il rumore, ogni contrasto si fa **contro il ctrl**
e lo stage 1 non si ripaga mai più (risparmio 21,8 job-h/variante a coorte piena,
4,75 job-h a probe).

## 5. Piano campagna

**Serie probe: {011, 014, 043, 170}** — 011 e 043 = le due vittorie oltre-rumore di A2
(guardie di regressione: chi le perde è morto) e i due gap più chiudibili verso
centralized (−0,156, −0,238); 170 = il caso di fallimento, bersaglio esplicito; 014 =
guardia di near-parity (−0,025) ed è la cella più veloce. Escluse: 082 degenere, 083
ceiling, 222/229 floor, 086 headroom 0,013, 001 il peggior rapporto discriminazione/ora.

- **Wave 0 (patch)**: forwarding dei 4 knob nel ramo enc-trio + flag `--s2-*` dedicati con
  default nulli + guardie + `_arm_tag` + `fed_echo` + `FLAG_OWNER_ARMS`. Poi il port prox.
- **Wave 1** (costo per cella = A2, 4 serie × 6 tag): `zn_a2s2_ctrl` (gate di validità),
  `zn_a2s2_adam` (slr 0.01), `zn_a2s2_avgm09`, prox decoupled μ∈{1,10} + loss μ∈{0.1,1}.
- **Wave 2** (dopo la patch val-every/patience): `tau1+fedadam`, `tau16+fedadam`, `tau64`,
  `tau1+sgd` (per isolare il contributo del server adattivo a τ=1).
- **Wave 3**: ≤2 vincitori × le 6 serie restanti, stessi tag (le celle si appaiano da sole).

**zn_report**: una riga ARMS per tag (stessa regex-dir di A2 finché `_arm_tag` non è
patchato — è il tag a separare); COPPIE pre-registrate: `ctrl − A2` (DEVE essere un null),
poi `variante − ctrl` per ognuna, e per il vincitore `winner − local_conv` e
`winner − centr_conv`. Metrica primaria AUPRC appaiata per serie; VUS-PR secondaria mai
aggregata. **Gate**: non perdere 011 né 043 oltre il rumore appaiato. **Target**: chiudere
il gap verso centralized su 011/043 e/o sollevare 170.

## 6. Ordine dei lavori (SUPERATO — vedi §7)

1. Patch di plumbing (wave 0) — piccola, tutta guardie + naming; nessun comportamento
   esistente cambia (bit da `_arm_tag` solo se non-default, flag nuovi con default nulli).
2. Port FedProx-sul-prior (~120-150 righe, μ=0 no-op provabile).
3. `zn_a2s2_ctrl` sul probe → se null vs zn_a2, via alla wave 1 col resume.
4. Wave 1 → wave 2 (dopo `--fed-s2-val-every`/`--fed-s2-patience`) → wave 3.

---

# §7. VERIFICA AVVERSARIALE (2026-08-07, stesso giorno) — verdetti e piano rivisto

Quattro revisori indipendenti: uno scettico per metodo (mandato: refutare codice E tesi di
miglioramento) più un **audit del valore atteso** misurato da disco. Esito: il piano delle
onde in §5 è superato.

## 7.1 🔴 L'audit del valore atteso boccia la premessa

La domanda «un prior ottimizzato meglio produce detection migliore?» è stata misurata su
**16 arm × 6 serie** (Spearman fra best-val di stage 2 e AUPRC, per serie, con le famiglie
separate per confondente-tokenizer):

- Nelle famiglie **oneste** (stesso stage 1) la correlazione è **nulla o perversa**:
  fed-family ρ = +0,09 / −0,03 / −0,26 / +0,09 / −0,20 / **+0,60 (ucr_170)**. Mediana ~0.
- Su **ucr_170** è fortemente nel verso SBAGLIATO a ogni raggruppamento (fino a ρ=+0,90):
  più bassa la val, peggiore la detection.
- **Within-arm** (stesso arm, più ottimizzazione: main→es→ot): più training FA MALE oltre
  soglia in 4 celle (043c −0,397 · 086l −0,287 · 170l-es −0,269 · 014l −0,204), aiuta in
  1 borderline (170l-ot +0,188).
- Il caso in cui val e detection si muovono insieme (A1→A2: val ↓ 6/6, AUPRC ↑ 5/6) è un
  cambio di **pooling dei dati** (0,2-0,6 nats), non di qualità dell'optimizer — i knob
  server operano nel regime 0,01-0,1 nats. E su ucr_011 **la val di A2 è GIÀ più bassa di
  quella di centralized**: il gap di −0,151 AUPRC **non è un gap di val**.

⇒ Il canale val è quasi saturo. I quattro metodi migliorano un proxy la cui relazione con
la detection è nulla-o-perversa sui nostri dati. **Ogni guadagno AUPRC da queste varianti
va trattato come esplorativo, non come atteso.**

## 7.2 Verdetti per metodo

| metodo | verdetto | P(batte ctrl oltre rumore) su 011/043 | perché |
|---|---|---|---|
| **FedSGD τ** | GO_WITH_CHANGES, **gated** | ~20-30% / ~15-25% | il meccanismo è vero ma la residenza del gap nello stage-2 è **non provata**; τ=1 rimosso dalla prima ondata; partire da τ=64/16 e pretendere monotonia |
| **FedAdam** | LOW_PRIORITY | ~8% / ~10% | risolve un fallimento **assente**: la val di A2 scende liscia e monotona su 4/4 probe (zero oscillazioni da smorzare). Utile solo come cella di controllo del 2×2 regime×server. slr {0.01, 0.03}, mai 1.0. Costo reale ×1,2-1,6, non «uguale» |
| **FedAvgM β=0.9** | **NO_GO** come da piano | ~4-6%, e P(perdere un gate) 30-45% | patologia misurata: al round 0 la media è DISTRUTTIVA — val(agg₀)=4,59/4,54 > floor uniforme ln(64)=4,16 mentre i client stanno a ≤3,5 — e il momentum senza damping memorizza proprio quella direzione, amplificata fino a 10×. Solo β∈{0.2,0.3} + skip del round 0 |
| **FedProx-prior** | LOW_PRIORITY | ~10-15% / ~10-15% | la premessa (drift intra-round ⇒ media degradata) è **contraddetta dalle nostre fed_history**: fra round consecutivi mean_loss sale al massimo di +0,0057 (zero risalite su 043/170) — l'aggregazione di A2 è loss-neutra. La barriera D3 non compare nei run veri. Gate: `aggregation_penalty` misurata nel ctrl. Se gira: SOLO decoupled, μ∈{1,3} (μ=10 = 94,6% di pull al client mediano = un τ-piccolo travestito) |

## 7.3 Scoperte di codice della verifica (reali oggi, non solo per le varianti)

1. 🔴 **L'oracolo val dello stage 2 è rumoroso per le MASCHERE, non per l'fp16**:
   `_val_loss_prior` ri-campiona maschere casuali dal RNG globale a ogni eval, senza seed
   (federated.py:482-498 + model/prior.py:277-287). σ misurata ≈ 0,006-0,018 nats contro
   `min_delta=1e-4`. Al regime a epoche è tollerabile; a τ piccoli la selezione diventa un
   processo di record di rumore e il falso-CONVERGED è silenzioso (la patience-6-round = 6
   step di staleness). **Fix prerequisito a ogni run τ: maschere val deterministiche +
   val fp32** (specchio della scelta già fatta per lo stage 1 a 466-474).
2. Correzione al §1: a τ=1 coi default il run **non tronca a 300 round — falso-converge
   prima**, e l'allarme TRUNCATED non può scattare (il best rumoroso non è quasi mai
   l'ultimo round).
3. Correzione al §3: il confondente weight-decay è **controfattuale oggi** — anche le
   baseline usano wd=0,01 (federated_eval.py:338-339; config.py:283). La VERA asimmetria
   di ricetta è un'altra: **le baseline hanno warmup+cosine, il fed stage-2 mai** — e la
   parte che morde è il warmup (le baseline spendono il 55-80% del run sotto il picco LR).
   Un arm di parità di schedule stage-2 è candidato più pulito del prox.
4. Correzione al §0: FedAdam sotto-regime-epoche costa ×1,2-1,6 per cella (converge in
   ×1,5-2,5 round), non «uguale». E il cap del passo transita fino a 2·server_lr (picco
   |m|/√v ≈ 2,13 al round ~13, senza bias-correction), non server_lr.
5. A2 batte A1 **7/10 con 2 pareggi degeneri** (082, 229) e 1 sconfitta (001), non 8/10.

## 7.4 Il piano rivisto: prima MISURARE il soffitto, poi comprare

**Gate 0 — `fedtok-centralprior` (il diagnostico che manca a tutto il piano).** Gli encoder
di A2 finiscono il run IDENTICI fra client (`enc_max_cross_client_delta=0.0` in knobs.json).
Quindi: resume dello stage 1 di A2 (`--resume-from … --s1-rounds 0`), UN prior allenato
centralmente sul flusso di token pooled, ricetta delle baseline. Quel numero è il **tetto di
OGNI upgrade dell'aggregazione stage-2**, per costruzione, a costo di ~1 cella. Se non batte
il prior federato di A2 su 011/043, l'intera direzione stage-2 è morta e i soldi vanno allo
stage 1 (dove vive il danno di 170).

**Gate 1 — `zn_a2s2_ctrl` strumentato** (non più solo null-test del resume):
- probe di detection **lungo la traiettoria**: ~5 checkpoint del prior (best/4, best/2,
  3best/4, best, last) + detect su ognuno ⇒ misura DIRETTA del legame val↔detection round
  per round — il dato che decide la wave 2 al posto dell'AUPRC-only;
- contrasto **best-val vs last-round** a costo zero (su 170 la selezione su val è essa
  stessa indiziata: es 0,060 vs kept-last 0,517);
- log della **aggregation_penalty** per round (val dei corpi client di fine round PRIMA
  della media vs val della media) ⇒ decide FedProx con un numero, non con D3;
- margini sintetici su val corrotta (in val non ci sono anomalie etichettate: iniezione di
  spike/permutazioni come proxy del margine).

**Poi, solo se i gate passano:** τ=64 → τ=16 (val-every-k + patience in eval + maschere
fisse), con FedAdam slr 0,03 come controllo del 2×2; prox decoupled μ∈{1,3} solo se
l'aggregation_penalty è >0; arm di parità schedule stage-2 (warmup+cosine) come candidato
economico aggiunto. FedAvgM solo a β≤0,3 con skip round-0, se mai.

**Cancellati:** τ=1 in prima battuta, prox loss-form (no-op silenzioso col precondizionatore,
il precedente encoder ha richiesto un audit post-hoc da 132 round), FedAvgM β=0.9,
l'obiettivo «sollevare ucr_170 con lo stage 2» (il danno è nello stage 1; 170 resta solo
come guardia don't-collapse).

**Pre-registrazione:** primario = ctrl−A2 null + no-harm su 011/043 + lettura di meccanismo
(val↓ co-mossa con margine↑ su almeno una serie). Se una variante abbassa la val con AUPRC
piatta, è l'esito PREDETTO dall'audit (canale val saturo), non «servono più round».
