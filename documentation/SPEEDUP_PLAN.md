# Piano di ottimizzazione: `-U-Federated` ← `-M-Real`

> ⚠️ **Nota 2026-07-10:** i benchmark di timing citati su `toy_fed` si riferiscono al dataset C=8 multivariato, ora **cancellato** (linea -M chiusa). Le ottimizzazioni implementate restano valide e attive; i numeri `toy_fed` sono storici. Dataset attivi: `toy_fed_uni` + `wsd_fed`.

> **STATO (2026-07-09): Fase 0/1/2 IMPLEMENTATE** sul branch `speedup-port`.
> Vedi §6 per ciò che resta. Misure end-to-end sui moduli reali di questa repo:
>
> | modulo | fp32 | fp16 | bf16 (vecchio default) |
> |---|---|---|---|
> | Stage 1 VQ-VAE (B=64, C=38, T=256) | 598 ms/step | **205 ms (2.92×)** | 6708 ms (**11× più lento**) |
> | Stage 2 prior (B=8, seq_len=3648) | 285 ms/step | **84 ms (3.40×)** | 427 ms (0.67×) |
> | `federated.py` toy_fed end-to-end | 65.8 s | 73.3 s | 282.7 s |
>
> Il 3.40× su stage2 riproduce indipendentemente il "3.64×" documentato da `-Real`.
> Su `toy_fed` fp16 **non** batte fp32 (73 vs 66 s): il modello è minuscolo e il tempo è
> dominato da data-loading e setup dei 6 client, non dalla GPU. Il guadagno è reale solo
> quando il forward è compute-bound — cioè sui dataset veri.

Analisi comparativa delle ottimizzazioni di velocità tra `TimeVQVAE-AD-M-Real` (riferimento)
e `TimeVQVAE-AD-U-Federated` (target), con piano di porting.

Metodo: 10 sonde parallele per sottosistema (stage1, stage2, config, utils, data, model-conv,
model-vq-prior, detect, orchestration, docs), ciascuna verificata con grep indipendente su
entrambe le repo → 108 finding. I punti che reggono l'ordinamento del piano sono stati
ri-verificati a mano e **misurati sulla GPU reale**.

I numeri di riga sono al 2026-07-09; ri-controllare con grep prima di editare.

---

## 0. Il vincolo che governa tutto: la GPU è Turing

```
Quadro RTX 8000 ×2 — compute capability 7.5 (Turing)
```

I tensor core Turing accelerano **fp16 e int8. Non bf16.** `-Real` lo documenta esplicitamente
(`config.py:322`):

> `NOTE: this GPU is Turing (Quadro RTX 8000) → fp16, NOT bf16 (no hw bf16).`

Benchmark eseguito su questo nodo (torch 2.11.0+cu128):

| operazione | fp32 | fp16 | bf16 |
|---|---|---|---|
| matmul 4096³ | 22.96 ms (6.0 TFLOP/s) | **3.37 ms (40.8 TFLOP/s)** | 40.23 ms (3.4 TFLOP/s) |
| conv2d 64×64×64×64 | 2.96 ms | **2.07 ms** | 13.78 ms |
| `nn.TransformerEncoderLayer` eval | 2.95 ms | **1.01 ms** | 4.62 ms |

**bf16 su questa GPU è 1.6×–4.7× più lento di fp32, e ~3–12× più lento di fp16.**

Trappola: `torch.cuda.is_bf16_supported()` ritorna **`True`** su questa GPU (supporto via
emulazione, senza tensor core). Un guard basato su quella funzione non protegge. L'unico
guard corretto è `torch.cuda.get_device_capability()[0] >= 8`.

### Conseguenza diretta: il path federato è in bf16

`pipeline/federated.py:107-116`:

```python
def _amp():
    if os.environ.get("FEDVQ_BF16", "1") != "0" and torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()
```

Attivo **di default** (`FEDVQ_BF16=1`) e usato in:

- `federated.py:158` — forward stage1 (encoder/decoder)
- `federated.py:396` — forward del prior transformer
- `federated_eval.py:81`, `:97` — **entrambi i bracci** del confronto RQ1

Quindi oggi il training federato gira **più lento che in fp32**. Il confronto RQ1 resta
metodologicamente equo (entrambi i bracci pagano lo stesso pedaggio) ma entrambi sono
penalizzati. `federated_eval.py:418` registra `"bf16": True` nella provenance dei risultati.

Le guardie fp32 dentro `model/vector_quantizer.py:164/192/211` (`torch.autocast(enabled=False)`
per EMA, statistiche sufficienti, merge — esattezza della Prop. 1) sono **dtype-agnostiche**:
continuano a funzionare identiche sotto autocast fp16. Il passaggio bf16 → fp16 non tocca
l'esattezza del merge.

---

## 1. Ottimizzazioni presenti in `-Real` e assenti/diverse in `-U-Federated`

Ordinate per rapporto beneficio/rischio.

| # | Ottimizzazione | `-Real` | `-U-Fed` | Guadagno | Rischio | Sforzo |
|---|---|---|---|---|---|---|
| 1 | **AMP fp16 autocast + GradScaler** in stage1/stage2 | `stage1.py:531-579`, `stage2.py:683-750`, `config.py:317-323` | assente (fp32 puro) | **3.64× su stage2** (misurato da `-Real`) | basso | S |
| 2 | **grad-clip + guard loss non-finita** | `stage1.py:530,558-571`, `config.py:316` | assente | abilita (1) in sicurezza | basso | XS |
| 3 | **Early-stop su `warmup_steps`, non `min_epochs`** | `stage1.py:690` | `stage1.py:621`, `stage2.py:709` | vedi §1.1 | basso | XS |
| 4 | **`quality_stage1` su GPU** | `quality_stage1.py:314` (cuda-if-avail) | `quality_stage1.py:311` (`"cpu"`) | 10–50× su quello stage | basso | XS |
| 5 | **`torch.compile` del prior** (handle separato) | `stage2.py:692-700`, `config.py:324-329` | assente | ~1.2–1.5× su (1) | medio | S |
| 6 | **`score_mask_chunk`**: scoring mascherato batchato | `prior.py:818-846`, `config.py:238` | assente | grande su CF/`cf_eval` (B=1); no-op su detect | basso | M |
| 7 | **`train_eval_stride_multiplier`**: stride grezzo sul TRAIN | `config.py:100`, `detect.py:81-89` | assente | ~M× sul pass TRAIN di detect | basso (opt-in, ~0.4% approx) | M |
| 8 | **Token cache compattata (uint8/int16) + write atomico** | `data.py:1372-1391` | `data.py:1257` (`torch.save` diretto, int64) | −4…8× disco/RAM, crash-safe | basso | XS |
| 9 | ~~Fast-path `np.quantile`~~ **da NON portare** | `detect.py:656-665` | assente | vedi §1.4 | — | — |
| 10 | **`--skip-existing`** (resume dello sweep) | `run.py:435,581-611` | assente | evita re-run di ore | basso | S |
| 11 | **GroupNorm** al posto di BatchNorm | `common.py:71-89` (`_make_norm`) | `common.py:93,110,124,143,160,182` (`BatchNorm2d`) | **~0 in velocità** — vedi §1.2 | medio | S |
| 12 | Helper schedule/LR (`resolve_lr`, `make_lr_lambda`, WSD, budget) | `utils.py:520-608` | assenti | indiretto | medio | M |
| 13 | Risoluzione path da project-root (non `cwd`) | `config.py:542-543` | `config.py:336-337` (`Path.cwd()`) | riproducibilità di `window_length` | basso | XS |

### 1.1 Il gate di early-stopping (#3) — differenza reale

`-Real` (`stage1.py:690`):
```python
elif (cfg.training.early_stopping and step >= warmup_steps
      and step - best_step >= patience_steps):
```

`-U-Federated` (`stage1.py:621`, identico in `stage2.py:709`):
```python
elif (cfg.training.early_stopping and (epoch + 1) >= min_epochs
      and step - best_step >= patience_steps):
```

Con `stage{1,2}_min_epochs = 15` (`config.py:209-210`), e poiché `min_epochs` **alza
`max_steps`** per estendere l'orizzonte del coseno LR, il commento di `-Real` (`config.py:362`)
spiega perché quel gate è sbagliato:

> `The C4 fix moved the early-stopping floor to warmup_steps ...; min_epochs no longer gates it.`
> *(altrove: gating su `min_epochs` "would otherwise make early stopping fire only at the last epoch")*

In `-U-Federated` l'early stopping è quindi **di fatto disattivato**: non può scattare prima
dell'epoca 15, e a quel punto lo span LR è già quasi esaurito. Ogni entità che converge presto
paga comunque le 15 epoche piene. Il guadagno dipende dall'epoca di convergenza reale — da
misurare, non da assumere.

### 1.2 GroupNorm (#11) non è una ottimizzazione di velocità

Va detto chiaramente, perché è facile classificarlo male: su singola GPU il BatchNorm fuso di
cuDNN è **più veloce** di GroupNorm. Il beneficio di GroupNorm qui è di **stabilità** (niente
statistiche per-batch su output encoder non limitato; niente crash a batch=1, regime tipico dei
client federati piccoli) e marginalmente di payload FedAvg (spariscono `running_mean` /
`running_var` / `num_batches_tracked`).

Nota che nessuna delle due repo usa `SyncBatchNorm` o `DistributedDataParallel` — il
parallelismo è per-processo, quindi non c'è nessun all-reduce di BatchNorm da eliminare.

⚠️ **Gotcha FedAvg**: GroupNorm cambia le chiavi dello `state_dict`. `_shared_keys`
(`federated.py:374`) e `_fedavg_shared` (`:409-418`) vanno rigenerati, e i checkpoint
BatchNorm esistenti falliranno il load strict.

### 1.4 Il fast-path `np.quantile` di `-Real` NON va portato (misurato)

`-Real` (`detect.py:656-665`) devia `median`/`quantile` su `np.quantile`, motivandolo con:
*"torch.quantile raises when the input has > 2**24 (~16.7M) elements"*.

Verificato su torch 2.11: **il limite riguarda la dimensione RIDOTTA, non il numel totale.**
Un reduce su `dim=1` con 67M elementi passa; è la forma 1-D a 2²⁵ che solleva
`quantile() input tensor is too large`. In `_aggregate` la dimensione ridotta è quella dei
**canali** (C ≲ 55), quindi il crash non si verifica mai. Le soglie usano già `np.quantile`.

E il path numpy è **più lento**: su un input poolato (700k, 38), mediana per canale →
torch **446 ms** vs numpy **1078 ms** (2.4× più lento). Output identici.

Conclusione: ottimizzazione apparente, in realtà una pessimizzazione su questo torch.
Non portata. Vale la pena rimuoverla anche da `-Real`.

### 1.3 I batch profile (`_BATCH_PROFILES`) NON sono uno speedup

Questo è il punto dove l'analisi automatica si contraddiceva e che ho verificato a mano.
`-Real` ha `_BATCH_PROFILES` (`config.py:658-678`) assente in Federated, ma il commento di
`-Real` riporta la **misura**:

> `MEASURED on the stage2 prior (fp16, RTX 8000, seq_len=3648), throughput is FLAT in batch size`
> `because attention is O(seq_len^2) and already saturates the GPU at B=16 —`
> `ms/window: B16 10.85 | B32 9.88 | B64 9.94 | B128 10.20 | B192 10.66 | B256 10.78 | B384 10.77`
> `so 192→384 bought −1.0% throughput for 2x the VRAM (17.1 → 34.1 GB).`

E in `data.py:1120`: *"measured throughput on this model is flat in batch size"*.

Quindi i batch profile sono una leva su **VRAM e granularità di early-stop/validazione**, non
sul throughput. Lo stage1 (conv VQ-VAE) *"is left untouched ... never profiled this way"*: non
esiste una misura che giustifichi lì un guadagno. **Non mettere questo in cima al piano.**

L'ottimo di throughput misurato è a B=32–64 (+8.4% vs 384), ma richiede `lr *= sqrt(B/B0)` e un
A/B su AUROC.

---

## 2. Ciò che `-U-Federated` ha e `-Real` no

| Cosa | Giudizio |
|---|---|
| `_amp()` bf16 nel path FL (`federated.py:107`) | ❌ **Pessimizzazione** su Turing (§0). L'infrastruttura è giusta, il dtype è sbagliato. |
| `opt.zero_grad(set_to_none=True)` (`federated.py:161,401`) | ✅ corretto (ma è già il default in torch ≥2.0) |
| `torch.cuda.empty_cache()` fra client (`federated_eval.py:179,249,378`) | ⚠️ igiene di memoria, **leggermente negativo** per la latenza; tenere solo se si va in OOM |
| `torch.backends.cudnn.benchmark = False` (`federated.py:242`) | ❌ **auto-sabotaggio**: le shape sono fisse per client, l'autotuner conviene. `federated_stage1()` non chiama mai `apply_determinism()` |

---

## 3. Ottimizzazioni assenti da **entrambe** le repo

Verificate con grep su entrambi gli alberi: zero hit.

### Gratis, rischio ~nullo
| Cosa | Dove | Guadagno |
|---|---|---|
| `fused=True` su AdamW | ogni `torch.optim.AdamW` (FED: `stage1.py:404`, `stage2.py:544`, `federated.py:145,384`, `federated_eval.py:76,92`) | ~1.1–1.3× sullo step ottimizzatore |
| `non_blocking=True` sui `.to(device)` | FED `stage1.py:509,553`, `stage2.py:647,649,677,679`, `federated.py:394`, `detect.py:286-287`, `data.py:1226` | pochi % (`pin_memory` è già attivo) |
| `torch.inference_mode()` invece di `no_grad()` | `detect.py:291`, `prior.py:786,791,824`, val loop `stage1.py:551` | pochi % |
| `prefetch_factor` / `drop_last` (solo train!) | `data.py:1075-1084` `_build_loader` | pipeline di input più stabile |
| `pin_memory` anche con `num_workers=0` | idem (oggi è `pin_memory=num_workers>0`) | H2D più veloce |

### Leve grosse, mai sfruttate
| Cosa | Guadagno | Rischio |
|---|---|---|
| **autocast fp16 in `detect.py`** — il forward di eval è fp32 in *entrambe* | **~2.9× sul prior** (misurato §0); detect è dominato dallo scoring | medio |
| **autocast nel token-encode** (`data.py`, pass one-shot) | ~1.5–2× su quel pass; solo argmax → nessun impatto numerico | basso |
| **`_mask_tokens` vettorizzato**: oggi `for row in range(B)` + `randperm` per riga (`prior.py:282-285`, e 4 altre varianti) | rimuove B kernel launch + una sync host per step | basso |
| **VQ: `cdist` → `addmm`** (`vector_quantizer.py:282-283`) | ~10–30% sullo step VQ; scala con K | basso |
| **short-circuit di `s_local`** quando `weight_s_local == 0` (`detect.py:294`) | salta un intero pass di ricostruzione | basso |
| **F.scaled_dot_product_attention** nel prior (`prior.py:242-248` usa `nn.TransformerEncoderLayer` con `norm_first=True` → fast-path *non* preso) | vedi ⚠️ sotto | medio |
| **Client FL seriali → shardati sulle 2 GPU** (`federated.py:268`: `for ci, c in enumerate(clients)`) | fino a #GPU× sul wall-clock per round | alto |
| **Riduzione `seq_len` / attention assiale** | la leva più grande in assoluto (attention è O(seq_len²)) | alto, greenfield |

⚠️ **FlashAttention non è disponibile su questa GPU.** Verificato:
`Flash attention only supports gpu architectures in the range [sm80, sm121]. Attempting to run on a sm 7.5 gpu.`
Backend SDPA disponibili su sm_75: `EFFICIENT_ATTENTION` e `MATH`. Un refactor verso SDPA dà
comunque il kernel mem-efficient, ma non aspettarsi i numeri di flash.

### Da NON fare
- **`channels_last`**: benefici solo con conv2d su tensor core; qui i canali sono pochi (~4–64),
  ci sono conv raggruppate e kernel (1,3)/(1,4). Atteso ~0 o negativo. Il `Conv3d` non lo
  supporta comunque.
- **DDP / DataParallel**: assenti da entrambe, e **giustamente**. Il parallelismo è per-processo
  (`run.py` genera un subprocess per `(dataset,entity)` con `CUDA_VISIBLE_DEVICES` round-robin).
  I modelli sono piccoli; DDP sarebbe una regressione e collide con lo scheduler esistente.

---

## 4. Piano di implementazione

### Fase 0 — Un minuto, zero codice
```bash
export FEDVQ_BF16=0
```
Disattiva l'autocast bf16 → torna a fp32. Recupera **1.6×–4.7×** sul path federato.
Da fare *prima* di qualsiasi misura di baseline, altrimenti si profila un artefatto.

### Fase 1 — La precisione giusta (il grosso del guadagno)

1. **`_amp()` → fp16 + GradScaler.** In `federated.py:107-116`:
   - `dtype=torch.float16`, guard su `torch.cuda.get_device_capability()[0] >= 8` per scegliere
     bf16 *solo* su Ampere+ (rende il codice portabile senza ri-pessimizzare qui).
   - fp16 **richiede** un `GradScaler` nei loop di training (`_local_train_stage1`,
     `_local_train_prior`); bf16 no. Questo è il vero costo del cambio.
   - ⚠️ Lo stato del `GradScaler` è **per-client** e **non va mai aggregato**: FedAvg tocca solo
     i parametri del modello.
   - Le guardie fp32 in `vector_quantizer.py:164/192/211` restano valide invariate.
   - Aggiornare la provenance `federated_eval.py:418` (`"bf16"` → `"amp_dtype"`).

2. **Portare AMP nei loop canonici** `stage1.py` / `stage2.py` da `-Real`, insieme al
   grad-clip e al guard `torch.isfinite(loss)` (#2) — è ciò che rende AMP sicuro. Aggiungere
   `amp: bool = True` e `grad_clip_norm: float = 1.0` a `TrainingConfig` + override env `AMP=0/1`.

3. **Usare la API moderna.** `-Real` usa `torch.cuda.amp.autocast` / `GradScaler`, entrambe
   deprecate su torch 2.11 (`FutureWarning`). Nel porting scrivere direttamente
   `torch.amp.autocast('cuda', ...)` e `torch.amp.GradScaler('cuda')`.

**Verifica:** con `AMP=0` il training deve tornare bit-identico a fp32
(`GradScaler(enabled=False)` è un pass-through puro).

### Fase 2 — Win gratuiti (una PR, rischio ~0)
- Rimuovere `torch.backends.cudnn.benchmark = False` (`federated.py:242`), o chiamare
  `apply_determinism(deterministic=False)`.
- `quality_stage1.py:311`: `"cpu"` → `"cuda" if torch.cuda.is_available() else "cpu"`.
- Gate early-stop: `(epoch+1) >= min_epochs` → `step >= warmup_steps` in `stage1.py:621` e
  `stage2.py:709`. Tenere `patience_steps` nel fingerprint; il cambio è resume-compatibile.
- `fused=True` su tutti gli AdamW (guard `torch.cuda.is_available()`).
- `non_blocking=True` su tutti i `.to(device)`; `prefetch_factor=4`; `drop_last=True` **solo sul
  loader di train** (mai val/test: scarterebbe finestre etichettate e falserebbe le metriche).
- `torch.inference_mode()` nei path di pura inferenza.
- Token cache: `_compact()` (uint8/int16) + write atomico `tmp` + `os.replace`
  (`data.py:1257`). Sicuro: il consumer fa già `.long()` (`stage2.py:176`).
- Fast-path `np.quantile` (`detect.py:451`) — **conta più in Federated**, dove l'eval poolato è
  esattamente il regime >2²⁴ elementi in cui `torch.quantile` crasha.

### Fase 3 — Strutturali (richiedono A/B)
- **`torch.compile` del prior**, handle separato (`train_prior = torch.compile(model.prior, dynamic=False)`),
  con fallback eager via `torch._dynamo.config.suppress_errors = True`.
  ⚠️ **Gotcha FedAvg critico**: `torch.compile` prefissa ogni chiave dello `state_dict` con
  `_orig_mod.`. **Non riassegnare mai `model.prior`.** `_shared_keys` (`federated.py:374`) e
  `_fedavg_shared` (`:409`) leggono `model.prior.state_dict()` e si romperebbero in silenzio.
  Preferire `load_state_dict` in-place sul modulo persistente al rebinding, altrimenti si
  ricompila ad ogni round.
- **autocast fp16 in `detect.py`** dentro il `no_grad` esistente (`detect.py:291`), lasciando
  accumulo score e soglie in fp32. Validare che gli score non si spostino oltre la tolleranza
  prima di fidarsi delle soglie.
- **`score_mask_chunk`** (`prior.py:805-818` → versione chunked di `-Real`). `-Real` lo dichiara
  **bit-identico** (`max|diff| = 0.0`) e `mb` collassa a 1 a B grande → no-op su detect,
  guadagno grosso su `cf_eval` (B=1).
- **`train_eval_stride_multiplier`** — opt-in, default 1 = bit-identico. ⚠️ In federato: il
  moltiplicatore deve essere **identico su tutti i client**, altrimenti le soglie per-client
  diventano non confrontabili prima dell'aggregazione. Il fingerprint della score-cache deve
  includerlo.

### Fase 4 — Ricerca
- Vettorizzare `_mask_tokens` (tutte e ~6 le varianti in `prior.py`). ⚠️ Cambia lo stream RNG:
  se serve masking riproducibile cross-client, ri-seedare deterministicamente.
- `cdist` → `argmin(||E||² − 2·x·Eᵀ)` nel VQ. ⚠️ Le assegnazioni devono restare bit-consistenti
  fra client per l'esattezza della Prop. 1: stesso dtype, stesso ordine, dentro il `no_grad`.
- **Shardare i client FL sulle GPU.** È il collo di bottiglia dominante del path federato:
  `federated.py:268` allena i client **strettamente in serie** su un solo device, quindi il
  wall-clock per round scala linearmente con #client (14 nel benchmark univariato). Il merge del
  server (`_server_merge`, `:276`) è una barriera di sincronizzazione: qualunque parallelismo
  deve essere completa-poi-fondi.
- Riduzione `seq_len` / attention assiale: la leva più grande, ma è un cambio di modello.

---

## 7. Secondo giro (2026-07-09): score_mask_chunk, --skip-existing, detect fp16, multi-GPU

| # | Cosa | Dove | Stato |
|---|---|---|---|
| 1 | `score_mask_chunk` (scoring mascherato batchato) | FED `model/prior.py`, `config.py`, `pipeline/stage2.py` | ✅ validato |
| 2 | `--skip-existing` | FED `run.py`, `run_matrix.py` | ✅ validato |
| 3 | `detect_amp` (fp16 in detect, opt-in, default OFF) | **entrambe**: `config.py`, `pipeline/detect.py` | ✅ validate separatamente |
| 4 | Multi-GPU opzionale | **entrambe**: `utils.resolve_device` (`TVQ_DEVICE`); FED `_client_devices` (`FEDVQ_DEVICES`) | ⚠️ **NON validato**, default-off per costruzione |

### 7.1 `score_mask_chunk` — NON è bit-identico su CUDA

`-Real` (`config.py:238`) afferma *"VERIFIED bit-identical (max|diff|=0.0)"*. **Falso su GPU.**
Misurato (W=16, B=1, fp32): CPU max|diff| = 0.0, **CUDA max|diff| = 6.7e-6** su punteggi di
scala ~30 → ~2 ULP fp32 (eps·max = 3.6e-6). Batchare le colonne cambia la forma della GEMM e
quindi l'ordine di accumulazione. Ogni path è deterministico in sé.

Impatto reale: **Spearman = 1.000000000000**, insieme flaggato invariato a q = 0.90/0.99/0.999,
e su una pipeline completa **tutti i 145 leaf numerici del report di detect sono identici**
(AUROC, best_f1, PATE, soglie: worst diff = 0.0). `mb=1` (o B ≥ target_rows) è esattamente
bit-identico. Sicuro come default su questa base.

Guadagno: **8.96×** a B=1 (il regime di `cf_eval` / `cf_channel`). **Zero su detect**: con
`batch_size_eval=1024` `mb` collassa a 1 e si esegue il path originale. Misurato: detect 0.98×.

### 7.2 `detect_amp` — fp16 nel forward di detect (default OFF)

Env `DETECT_AMP=off|auto|fp16|bf16`; `auto` = fp16 sotto Ampere, bf16 da Ampere in su.
Registrato nella fingerprint della score-cache (`schema_version` 3→4 in Real, 1→2 in FED) così
una cache fp32 non può servire un run fp16.

| | GPU scoring loop | AUROC Δ | note |
|---|---|---|---|
| `-Real`, smap/T-1 (reale, 25 ch, seq_len=2400) | **3.34× più veloce** | 3.1e-3 | AUPRC Δ 3.6e-3 |
| `-U-Federated`, toy_fed | **2.03× più veloce** | 2.1e-5 | best_f1 e vus_roc identici |

`DETECT_AMP=off` verificato **bit-identico** in entrambe (run ripetuti); fp16 è deterministico
run-to-run. **Default OFF perché la Δ AUROC di 3e-3 su `-Real` è materiale per un paper.**

⚠️ **Non misurare lo speedup sul wall-clock di `detect()`**: su dataset piccoli è dominato dalle
metriche CPU (VUS-ROC, PATE, affiliation) e ha varianza 1.65× fra run identici. Misurare
`_compute_entities_raw`.

### 7.3 Due gotcha della letteratura interna che si sono rivelati falsi

1. **"`torch.cdist` non è protetto, in fp16 i token flippano"** — no: `cdist` è nella lista
   **fp32-promote** di autocast, quindi le distanze sono già fp32 anche sotto autocast fp16
   (verificato su torch 2.11: input fp16 → output fp32). Un guard esplicito è codice morto.
   I token flippano lo stesso, ma dello **0.067%** (smap/T-1), perché è `projected` — l'output
   dell'encoder — a essere fp16. La divergenza degli score viene quasi tutta dai logit fp16 del
   prior (max|Δ| 2.3e-2 su scala 2.38 a token identici).
2. Non correlato ma emerso: **i checkpoint esistenti di `-Real` non caricano più** (sono
   dell'era BatchNorm, hanno `running_mean`; il codice attuale usa GroupNorm). Condizione
   preesistente, non causata da queste modifiche.

### 7.4 Multi-GPU (NON validato)

- **Entrambe**: `utils.resolve_device()` + env `TVQ_DEVICE=cuda:1|cpu`. Sostituiti i 7 siti che
  fissavano `torch.device("cuda" if ...)`. Con env non impostata ritorna esattamente il valore
  di prima. Solo per run diretti: sotto `run.py` i subprocess sono già pinnati con
  `CUDA_VISIBLE_DEVICES`, che **rimappa gli indici**.
- **FED**: `_client_devices()` + env `FEDVQ_DEVICES=0,1|all` distribuisce i client FL sulle GPU.
  Tutti i merge e gli assert passano da **CPU** (`torch.allclose` e `torch.stack` fra device
  diversi sollevano, non ritornano False). I client stage2 sono **co-locati** con il rispettivo
  encoder stage1 congelato. I client restano allenati **in serie**: è un guadagno di *capacità
  di memoria*, non di wall-clock. Non parallelizzare il loop con thread/stream: cambierebbe
  l'interleaving RNG e romperebbe il determinismo su cui poggiano le ablazioni.
  `federated_eval.py` **non** è stato adattato: non usare `FEDVQ_DEVICES` con la harness RQ1.

Cosa controllerebbe una validazione futura: invarianza default-off; equivalenza `FEDVQ_DEVICES=0`
vs unset; codebook aggregato identico fra 1-GPU e 2-GPU; i 4 assert di correttezza; co-locazione
stage1/stage2.

---

## 6. Cosa è stato fatto e cosa resta

### 6.1 Implementato (branch `speedup-port`)

| Cosa | File | Verifica |
|---|---|---|
| `_amp()` bf16 → dtype per compute-capability (fp16 su Turing) | `pipeline/federated.py:109-158` | `FEDVQ_AMP=auto\|fp16\|bf16\|off`, testato |
| GradScaler per-client (mai aggregato), in entrambi i loop FL | `federated.py`, `federated_eval.py` | invarianti Prop.1 verdi in fp16 |
| AMP + GradScaler + guard `sched.step()` nei loop canonici | `stage1.py`, `stage2.py` | smoke AMP on/off, checkpoint ok |
| `amp: bool = True` + override env `AMP=0/1` | `config.py` | `AMP=0` → fp32 |
| `amp` escluso dall'hash della token cache | `data.py::_tokens_config_hash` | hash **identico** a prima → nessuna cache invalidata |
| Early-stop su `warmup_steps` (era `min_epochs`) | `stage1.py`, `stage2.py` | scatta a epoch=1/step=26 invece di 195 step |
| `cudnn.benchmark = False` → `True` | `federated.py:242` | — |
| `quality_stage1` default su GPU | `quality_stage1.py:311` | — |
| `fused=True` su tutti gli AdamW (legato al device reale) | 6 siti | testato con GradScaler+fp16 |
| `non_blocking=True` + `pin_memory` + `prefetch_factor=4` | `data.py`, `stage*.py`, `detect.py`, `federated*.py` | — |
| Token cache compattata (uint8/int16) + write atomico | `data.py` | uint8 verificato, nessun `.tmp` residuo |
| `Path.cwd()` → project root | `config.py` | — |

### 6.2 Resta da fare — presente in `-Real`, assente in Federated

| Cosa | Guadagno atteso | Rischio | Perché non è stato fatto ora |
|---|---|---|---|
| `torch.compile` del prior (handle separato) | ~1.2–1.5× sopra AMP | medio | ⚠️ `_orig_mod.` rompe `_fedavg_shared`/`_shared_keys` in silenzio. Serve cura. |
| `score_mask_chunk` (scoring mascherato batchato) | grande su `cf_eval` (B=1); no-op su detect | basso | `-Real` lo dichiara bit-identico; va portato + validato `max\|diff\|=0` |
| `train_eval_stride_multiplier` | ~M× sul pass TRAIN di detect | basso | opt-in, approssima (~0.4%). In FL: stesso M su tutti i client |
| GroupNorm ← BatchNorm | **~0 in velocità**; stabilità a batch=1 | medio | cambia le chiavi dello `state_dict` → invalida checkpoint + set FedAvg |
| Helper schedule/LR (WSD, budget, `resolve_lr`) | indiretto | medio | cambia la traiettoria; serve A/B su AUROC |
| `--skip-existing` (resume dello sweep) | ore risparmiate sui re-run | basso | orchestrazione, ortogonale |
| `grad_clip_norm` | sicurezza sotto fp16 | basso | default 1.0 in `-Real` cambierebbe la traiettoria; il GradScaler già salta gli overflow |
| `_BATCH_PROFILES` | **nessuno** (throughput piatto in B) | basso | leva su VRAM/granularità, non su velocità |
| `np.quantile` fast-path | **negativo** (2.4× più lento) | — | vedi §1.4 — da non portare |

### 6.3 Resta da fare — assente da **entrambe** le repo

| Cosa | Guadagno atteso | Rischio |
|---|---|---|
| **autocast fp16 in `detect.py`** (il forward di eval è fp32 in entrambe) | ~2.9–3.4× sullo scoring, che domina detect | medio: sposta gli score, serve A/B sulle soglie |
| **Client FL seriali → shardati sulle 2 GPU** (`federated.py:268`) | fino a #GPU× per round | alto: il merge del server è una barriera |
| `_mask_tokens` vettorizzato (oggi `for row in range(B)` + `randperm`) | B kernel launch + 1 sync host per step | basso; cambia lo stream RNG |
| VQ: `cdist` → `addmm` (`vector_quantizer.py:282`) | ~10–30% sullo step VQ | basso; assegnazioni devono restare bit-consistenti fra client |
| Short-circuit di `s_local` quando `weight_s_local == 0` | salta un pass di ricostruzione | basso |
| `F.scaled_dot_product_attention` nel prior | ⚠️ **niente FlashAttention su sm_75**; solo `EFFICIENT`/`MATH` | medio |
| autocast nel token-encode (pass one-shot) | ~1.5–2× su quel pass | basso |
| `torch.inference_mode()` al posto di `no_grad()` | pochi % | basso |
| `drop_last=True` sul solo loader di train | evita un batch ragged | basso; cambia #step/epoca |
| Riduzione `seq_len` / attention assiale | la leva più grande (attention è O(seq_len²)) | alto, greenfield |
| `channels_last` | **~0 o negativo** (canali pochi, conv raggruppate) | — |
| DDP / DataParallel | **da non fare** (parallelismo per-processo, modelli piccoli) | — |

---

## 5. Protocollo di verifica

Prima di dichiarare qualunque speedup:

1. **Baseline con `FEDVQ_BF16=0`.** Misurare in bf16 significa misurare un bug.
2. Ogni knob deve avere un off-switch che restituisce il comportamento precedente
   (`AMP=0`, `COMPILE=0`, `DATASET_BATCH_PROFILE=off`, `score_mask_chunk=1`,
   `train_eval_stride_multiplier=1`).
3. Per i cambi dichiarati *bit-identici* (`score_mask_chunk`, stride=1, `AMP=0`): asserire
   `max|diff| == 0` sugli score di detect, non "sembra uguale".
4. Per i cambi che spostano la numerica (AMP on, GroupNorm, `cdist`→`addmm`, `_mask_tokens`
   vettorizzato): A/B su AUROC, non solo su val-loss.
5. `logs/vram.csv` (via `utils.log_gpu_peak`) esiste in `-Real` per il tuning dei batch. Il nodo
   è **condiviso** (altri utenti girano Jupyter): lasciare margine di VRAM.
