# `local` model vs. original TimeVQVAE‑AD — verified differences

**What this is.** A component‑by‑component comparison of this repo's **`local`** model
(the non‑federated baseline every federated arm is measured against) with the **original
upstream** [ML4ITS/TimeVQVAE‑AnomalyDetection](https://github.com/ML4ITS/TimeVQVAE-AnomalyDetection)
(Lee et al., *"Explainable time series anomaly detection using masked latent generative
modeling"*).

**How it was produced (2026‑07‑15).** The `local` side was read directly from source in this
repo (`model/`, `pipeline/`, `config.py`). The upstream side was read from the **actual upstream
source files** on branch `main` (not from memory or the paper), then **adversarially
re‑verified** file‑by‑file. Every corrected item below survived that verification pass; the
provenance (exact upstream files + quoted lines) is in the last section.

> ⚠️ **Correction to repo lore.** A prior note in this repo called the time‑domain
> `RefinementHead` a *"non‑upstream addition."* **That is wrong.** The upstream decoder has the
> identical residual head (`out = out + self.linear(out)` after iSTFT). It is present and
> load‑bearing in *both*. (See the `local` reconstruction autopsy, `scripts/local_recon_autopsy.py`.)

---

## What "local" means here

`local` is the **standard two‑stage TimeVQVAE‑AD** (STFT VQ‑VAE tokenizer → MaskGIT prior),
trained on **one client/series at a time** with the default `config.py`, with no federation
active. Relative to upstream it is a **heavily rewritten, flat‑code, univariate‑only (C=1)
port**: a plain‑torch training loop instead of PyTorch Lightning/Hydra/wandb, a single
`config.py` dataclass instead of `configs/config.yaml`, plus federation machinery
(suff‑stat / k‑FED codebook merge, prior anchors) that is **dormant** in the `local` arm.

The **design is faithfully ported**. What actually changes model behaviour is a handful of
**default hyperparameters** and a few **structural swaps** — *not* the federation code, which
never runs in `local`.

---

## TL;DR — differences that actually change behaviour (ranked)

1. **Tokenizer latent width: `4` (local) vs `64` (upstream).**
2. **Prior stack + positional embeddings:** stock `nn.TransformerEncoder` + LayerNorm + embed **128** + **2‑D‑factorised (freq+time)** positions (local) vs x‑transformers + RMSNorm + embed **64** + **1‑D‑flat** positions (upstream).
3. **VQ codebook dynamics:** EMA **0.99** + k‑means init + dead‑code revival (local) vs EMA **0.8** + no k‑means + no revival (upstream).
4. **Optimiser / precision / batch:** Adam (coupled wd) / **fp16** / batch **64** (local) vs AdamW (decoupled wd) / **fp32** / batch **256↓128** (upstream).
5. **Early stopping** on `val/loss` + min 15 epochs (local) — makes effective training length a variable.
6. **Anomaly score is a configurable superset** (local) — default identical (pure prior NLL), but a reconstruction/interaction term *can* be switched on.
7. **Evaluation metrics:** VUS‑PR / PATE / paper top‑K (local) vs UCR top‑k localization only (upstream) → cross‑repo numbers are not directly comparable.
8. **Framework & scope:** flat‑code, univariate‑only, plain torch, `config.py` (local) vs Lightning + Hydra + wandb, `config.yaml`, general (upstream).

---

## Component‑by‑component

| Component | Original TimeVQVAE‑AD (upstream) | This repo's `local` | Practical consequence |
|---|---|---|---|
| **Transform / STFT** | `torch.stft`, `n_fft=4`, `normalized=True`, Hann window; real+imag stacked → 2 channels; `H = n_fft//2+1 = 3` freq bins. No band split. | **Identical**: STFT `n_fft=4`, normalized, Hann, real+imag → 2 channels, 3 freq bins. | None — faithfully ported. The time‑freq "image" is only 3 bins tall in both. |
| **Stage‑1 encoder / decoder** | ResNet blocks, **Snake** activation, temporal‑only downsampling `stride=(1,2)`, frequency‑independent `ResBlock` kernel `(1,3)`, initial width `d=4` doubled per stage up to `dim=64`, `n_resnet_blocks=4`, `downsampled_width=32`, dropout 0.3. Decoder mirrors with `ConvTranspose2d`. | Same block family (Snake, height‑1 kernels `(1,3)/(1,4)`, `stride=(1,2)`, 4 blocks, `downsampled_width=32`), grouped `Conv2d` (`groups=C=1`), **`BatchNorm2d`**, encoder dropout **0.2** / decoder 0.3, lazy‑built, outputs **`token_embedding_dim=4`**. | Same conv topology and same frequency‑independence (from the **height‑1 kernel**, not from grouping — `groups=1` is an ordinary conv; grouping only matters at C>1). The one big change is the **latent width 64 → 4** (see next row). Whether local's `BatchNorm2d` differs from upstream is **unverifiable** — upstream's ResBlock internal norm was not characterized. |
| **Vector quantizer + codebook** | Vendored lucidrains `VectorQuantize`; **`codebook_size=64`**; `codebook_dim` defaults to encoder `dim = 64`; `project_in/out = Identity` (dims equal); EMA **decay 0.8**; `commitment_weight=1.0`; **`kmeans_init=False`**; dead‑code threshold **0**. Single codebook, Euclidean. | `SharedCodebookPerChannelVQ → SharedVectorQuantizer`; single shared codebook; **`codebook_size=64`**; **token/codebook dim forced = 4** (no projection); EMA **decay 0.99**; `commitment_weight=1.0`; **k‑means init** on first batch; dead‑code expiry threshold **2**; channels folded into batch. | Codebook **count** (64) is identical, but each code lives in a **4‑D** space vs **64‑D** — a **16× smaller quantization/latent space**, the single biggest capacity divergence. Slower EMA (0.99 vs 0.8), active dead‑code revival (thr 2 vs 0) and k‑means init change utilization/collapse dynamics. |
| **LF/HF band split** | **None** — a single encoder/codebook/prior over the full STFT map. (This AD variant already dropped the *generative* TimeVQVAE's dual LF/HF branches.) | **None** — identical single‑branch design. | No difference; both differ from generative TimeVQVAE the same way. |
| **Stage‑2 prior + positional embeddings** | `BidirectionalTransformer` on x‑transformers `ContinuousTransformerWrapper`; **token/embed dim 64**, hidden 128, depth 4, heads 4, `attn_dim_head=64`, `ff_mult=4` (ff inner 512), **RMSNorm**, dropout 0.3; `T=20`, `choice_temperature=4`, cosine. **1‑D learned pos‑emb** over the flattened `(H'·W')` token sequence (`nn.Embedding(num_tokens+1)`); wrapper `use_abs_pos_emb=False`. Logits **weight‑tied** to `tok_emb` **+ learned bias**; `pred_head = Linear→GELU→LayerNorm`. | `MaskGITPrior3DPos` on **stock `nn.TransformerEncoder`** (`norm_first=True`, GELU, **LayerNorm**); **embed dim 128**, hidden 128, ff 512, depth 4, heads 4, dropout **0.2**; `T=20`, `choice_temperature=4.0`, cosine. **Factorised positions** `PE(c,f,t)=E_ch[c]+E_freq[f]+E_time[t]` (three `nn.Embedding`). Logits weight‑tied **+ learned per‑position `output_bias`**; `pred_head = Linear+GELU+LayerNorm`. `mask_token_id = codebook_size`. | Prior **size** is comparable (depth/heads/ff equal), but: token‑embed width **128 vs 64**, norm **LayerNorm vs RMSNorm**, library **torch vs x‑transformers**, and positions go from **1‑D flat** to **factorised freq+time** — a genuinely different positional inductive bias (an explicit freq‑axis embedding upstream lacks). *Note:* at C=1 the `E_ch` table is a single row (a constant bias), so the factorisation is effectively **2‑D (freq+time)**. |
| **Anomaly score composition** | **Structurally prior‑only** masked‑token NLL: `a_w = −log p.mean(...)`; stage‑1 reconstruction is computed only for plotting, never added. Multi‑scale rates `(0.1,0.3,0.5)` **summed**; per‑frequency reduction = **mean over H′ rows**; `a_final=(a+MA(a))/2`; rolling assembly = sum; threshold `q=0.99`. Metric = **UCR top‑k localization only**. | Default score = **pure prior NLL** via `weight_s_prior=1.0, weight_s_local=0.0, weight_s_interaction=0.0`; multi‑scale `τ=(0.1,0.3,0.5)`; per‑frequency reduction = **mean** (`scores.mean(dim=1)`, matches upstream); `a_final=(a_s+MovingAvg(a_s,T))/2`; rolling = sum; threshold `q=0.99`. Adds **VUS‑PR / PATE** + paper top‑K. | **Default score behaviour is the same** (incl. the mean‑over‑freq reduction). The real differences are: (1) local exposes **configurable** recon/interaction weights (a *superset* of upstream, off by default); (2) the reported **metric set differs** — upstream never computes PR/VUS/PATE. ⚠️ `cfg.scoring.aggregation="max"` is the **channel** aggregation and is a **no‑op at C=1** — it does *not* change the frequency‑axis reduction. |
| **Refinement head** | **Present**: after iSTFT + interp, `out = out + self.linear(out)` (residual Linear on the time‑domain waveform). | **Present**: lazily‑built `nn.Linear(window_len, window_len)`, additive residual `x → x + Linear(x)` on the time‑domain waveform after iSTFT. | **Faithfully ported** — present and load‑bearing in *both*. (Corrects the repo's mistaken "non‑upstream addition" label.) |
| **Training framework** | PyTorch Lightning + wandb; **AdamW** (decoupled wd; value not passed explicitly → torch default 0.01); linear‑warmup + cosine (warmup 0.1, η_min 1e‑6); step caps stage1 10k / stage2 50k + 12 h wall cap; batch **256 (s1) / 128 (s2)**; **fp32**; no augmentation. | Plain torch loop, single `config.py`; **Adam** (coupled wd = 0.01, explicit); `warmup_rate=0.1`; step caps ≤10k / ≤50k **plus early‑stopping on `val/loss` and min 15 epochs**; batch **64**; **AMP fp16** (Turing GPU). | Different optimiser regularization (Adam coupled‑wd vs AdamW decoupled‑wd), fp16 vs fp32 numerics, smaller batch (**4× at s1, 2× at s2**), and early stopping that can end training **before** the step caps — so effective training length and convergence differ even where the architecture matches. |
| **Key default hyperparams** | codebook 64 @ **dim 64**; EMA 0.8; k‑means off; dead‑code 0; prior embed **64**, dropout 0.3, RMSNorm; batch 256/128; AdamW; fp32; lr 1e‑3 both stages; warmup 0.1; steps 10k/50k. | codebook 64 @ **dim 4**; EMA 0.99; k‑means on; dead‑code 2; prior embed **128**, dropout 0.2, LayerNorm; batch 64; Adam; fp16; lr 1e‑3 both stages; warmup 0.1; steps ≤10k/≤50k + early stop. | Numeric deltas compound; **lr, warmup and step caps are the shared invariants**. |

---

## The important differences, explained

### 1. Tokenizer latent width: 4 vs 64 (the biggest one)

In **both** repos the encoder output dimension **equals** the codebook dimension — there is no
projection layer between them. Upstream sets that width to **64** (`encoder.dim`); this repo sets
it to **`token_embedding_dim = 4`**. So every discrete token, and the continuous vector it
quantizes, lives in a space that is **16× narrower** here.

**Why it matters.** This caps how much the tokenizer can represent, which flows straight through
to (a) reconstruction fidelity and (b) how *discriminative* the downstream prior‑NLL score can
be. It is the dominant capacity difference between the two models.

**Nuance (don't over‑attribute to the codebook).** The number of codewords (64) is *identical*;
what shrank is the **dimensionality**. And this repo's own reconstruction autopsy found the
quantizer essentially **innocent** (bypassing VQ changed recon by only +0.0025) and the
`RefinementHead` load‑bearing — i.e. the real bottleneck is the **narrow encoder/decoder latent
width + data scarcity**, not the quantization step per se. Frame this as a *latent‑width* cut,
not a *codebook* cut. See the `local` reconstruction autopsy, `scripts/local_recon_autopsy.py`.

### 2. Prior transformer stack + positional embeddings

Same **shape** (depth 4, heads 4, ff 512, T=20, choice_temperature 4, cosine schedule,
weight‑tied logits + learned bias, `pred_head = Linear→GELU→LayerNorm`), but four swaps:

- **Library:** stock `nn.TransformerEncoder` (local) vs x‑transformers `ContinuousTransformerWrapper` (upstream).
- **Norm:** LayerNorm (local) vs RMSNorm (upstream).
- **Token‑embed width:** 128 (local) vs 64 (upstream).
- **Positional scheme (the substantive one):** upstream indexes **one flat 1‑D position** over
  the flattened `(H'·W')` token grid; local **factorises** into separate `E_freq[f]` and
  `E_time[t]` tables (plus a degenerate `E_ch` that is a single constant row at C=1).

**Why it matters.** The factorised freq/time embedding gives the prior an **explicit
frequency‑axis signal** the upstream 1‑D scheme never had, i.e. a different positional inductive
bias — on top of a wider token embedding and a different normalization. This is a real (moderate)
change to prior capacity and to what structure the prior can exploit.

### 3. VQ codebook dynamics

Local uses **EMA decay 0.99** (slower, more stable running centroids), **k‑means initialization**
on the first batch (data‑driven codes from step 0), and **dead‑code revival** at usage
threshold 2 (unused codes get re‑seeded). Upstream uses **EMA 0.8**, **no k‑means init**, and
**no revival** (threshold 0).

**Why it matters.** These three knobs govern **codebook utilization and collapse** — directly
relevant to the codebook‑collapse behaviour noted elsewhere in this repo. They change *which*
tokens exist and how stably, independent of everything downstream.

### 4. Optimiser, precision, batch size

- **Adam (coupled weight decay)** here vs **AdamW (decoupled weight decay)** upstream — these are
  *different regularizers*, not a cosmetic rename; the numeric match (0.01) does **not** mean
  identical regularization.
- **fp16 (AMP)** here vs **fp32** upstream — different Turing numerics.
- **batch 64** here vs **256 (stage 1) / 128 (stage 2)** upstream — higher gradient noise and a
  different number of effective steps per epoch here.

**Why it matters.** All three shift the *optimization trajectory* and the final scores even where
the architecture is bit‑for‑bit the same. When comparing to published upstream numbers, these are
confounders.

### 5. Early stopping

Local adds **early stopping on `val/loss`** with a **min of 15 epochs**; upstream trains to the
**step cap** (or the 12‑hour wall clock). So here, *effective training length* is a **variable**,
not a constant — which is exactly what the repo's "undertraining" discussions hinge on.
*(Caveat: upstream's **absence** of early stopping is inferred from its step‑based setup, not
confirmed from its callbacks.)*

### 6. The score is a configurable superset

Both **default** to a **pure MaskGIT prior masked‑token NLL** score, and both reduce over the
frequency rows with a **mean**, apply the multi‑scale `τ=(0.1,0.3,0.5)` rates, the impulse
`a_final=(a+MA(a))/2`, and a `q=0.99` threshold. The difference is only that local **exposes**
`weight_s_local / weight_s_prior / weight_s_interaction`, so a reconstruction/interaction term
*can* be switched on. Upstream's score is **structurally** prior‑only. Default behaviour is
identical; local is a strict superset.

> **Retracted claim.** An earlier draft said local uses `max` over the frequency rows vs
> upstream's `mean`. **False.** `cfg.scoring.aggregation="max"` is the **channel** aggregation
> (`detect.py`), applied to a `(T, C)` tensor **after** rolling assembly — and with C=1 it is a
> **no‑op**. The frequency‑row reduction is a **mean** in both (`model/prior.py`, `scores.mean(dim=1)`).

### 7. Evaluation metrics

Local reports **VUS‑PR / PATE** (range‑aware) + paper top‑K; upstream reports **UCR top‑k
localization only** (no F1/PR/VUS/PATE/AUROC). Not a model change, but it means **"better" means
different things** in each repo, and raw cross‑repo numbers are **not comparable**.

---

## Same / faithfully ported (unchanged from upstream)

- **STFT front‑end** — `n_fft=4`, `normalized=True`, Hann, 3 freq bins, real+imag → 2 channels.
- **No LF/HF band split** — single encoder / single codebook / single prior in both (both already
  diverge from the *generative* TimeVQVAE this way).
- **Codebook size 64**, single shared codebook, EMA + straight‑through, `commitment_weight=1.0`.
- **Conv block family** — ResNet blocks, Snake activation, frequency‑independent height‑1 kernels,
  temporal‑only downsampling `stride=(1,2)`, `n_resnet_blocks=4`, `downsampled_width=32`, mirrored decoder.
- **MaskGIT prior *design*** — bidirectional transformer, random‑token masking with cosine
  schedule, `T=20` iterative decode, `choice_temperature=4.0`, depth 4 / heads 4 / ff 512,
  weight‑tied logits + learned bias, `pred_head = Linear→GELU→LayerNorm`, `mask_token_id = codebook_size`.
- **Default anomaly score** — pure prior masked‑token NLL; multi‑scale rates `(0.1,0.3,0.5)`
  summed; mean over frequency rows; `a_final=(a+MovingAvg(a))/2`; `q=0.99` quantile threshold.
- **Refinement head** — residual Linear on the post‑iSTFT time‑domain waveform, present and
  load‑bearing in **both**.
- **Core training invariants** — `lr=1e‑3` for both stages, `warmup_rate=0.1`, step caps 10k/50k,
  univariate C=1.

---

## Unverifiable / hedged upstream facts

These are stated as *inferences*, not asserted as verified:

- **Upstream ResBlock internal normalization** (BatchNorm / GroupNorm / none) was not
  characterized, so whether local's `BatchNorm2d` is a true divergence **cannot be confirmed**.
- **Upstream `weight_decay`** value is *inferred* to be torch's AdamW default (0.01) because it is
  never passed explicitly; upstream applies it AdamW‑style (decoupled), local uses Adam (coupled)
  — the numeric match should not be read as identical regularization.
- **Upstream's absence of early stopping** is inferred from its step‑based training, not confirmed
  from its Lightning callbacks.
- **`frequency_indepence` default** in the upstream encoder signature has no code default and its
  caller passes `True`; irrelevant to the C=1 comparison since the height‑1 kernel makes the
  encoder frequency‑independent anyway.

---

## Source provenance (upstream, branch `main`, verified 2026‑07‑15)

| Fact | Upstream file(s) |
|---|---|
| Single branch, no LF/HF; one encoder/decoder/codebook | `experiments/exp_stage1.py`, `models/stage1/vq_vae_encdec.py` |
| Encoder/decoder blocks (Snake, `(1,3)` kernel, `stride=(1,2)`, `d=4→64`) | `models/stage1/vq_vae_encdec.py` |
| VQ (vendored lucidrains, `codebook_size=64`, `dim=64`, EMA 0.8, `kmeans_init=False`, Identity projection) | `models/stage1/vq.py`, `configs/config.yaml` |
| STFT `n_fft=4`, normalized, Hann, real+imag→channels, `H=3` | `utils/__init__.py`, `configs/config.yaml` |
| Refinement head `out = out + self.linear(out)` after iSTFT+interp | `models/stage1/vq_vae_encdec.py` (`VQVAEDecoder.forward`) |
| Stage‑1 loss = time‑domain MSE + VQ (no spectral loss anywhere) | `experiments/exp_stage1.py` |
| Single joint prior; embed 64 / hidden 128 / depth 4 / heads 4 / ff_mult 4 / RMSNorm / dropout 0.3 | `models/stage2/maskgit.py`, `models/stage2/bidirectional_transformer.py`, `configs/config.yaml` |
| 1‑D flat `pos_emb = nn.Embedding(num_tokens+1, in_dim)`; `use_abs_pos_emb=False` | `models/stage2/bidirectional_transformer.py` |
| Weight‑tied logits + learned `self.bias`; `pred_head Linear→GELU→LayerNorm` | `models/stage2/bidirectional_transformer.py` |
| Random‑token masking, cosine `γ(r)=cos(rπ/2)`, `T=20`, `choice_temperature=4` | `models/stage2/maskgit.py`, `configs/config.yaml` |
| Prior‑only score `a_w = −log(p).mean(...)`; recon only for plotting; `a_final=(a+MA)/2` | `evaluation/__init__.py`, `evaluate.py` |
| Lightning + wandb; AdamW; fp32; batch 256/128; lr 1e‑3; steps 10k/50k | `stage1.py`, `stage2.py`, `configs/config.yaml` |

**Local side** read from: `config.py`, `model/transforms.py`, `model/encoder.py`,
`model/decoder.py`, `model/vector_quantizer.py`, `model/common.py`, `model/prior.py`,
`pipeline/detect.py`.
