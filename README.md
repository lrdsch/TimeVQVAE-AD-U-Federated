# Federated TimeVQVAE-AD (univariate)

A **federated-learning** study of the [TimeVQVAE-AD](documentation/2311.12550v5.pdf) anomaly
detector (Lee et al., 2023) on **univariate** time series. Instead of pooling raw data, we train
**one model per cluster of clients** (one client = one series) and ask what — if anything — can be
shared. Fixed architecture, flat code, one implementation per concept, all configuration in
[config.py](config.py).

> **Start here for the science:** [documentation/RESEARCH_LEDGER.md](documentation/RESEARCH_LEDGER.md)
> — every model with its *why*, its training conditions, its real (matched) result, and a
> trust verdict. File inventory: [documentation/DISPOSITION.md](documentation/DISPOSITION.md).
> Dead/open lines: [documentation/DEAD_AND_OPEN_LINES.md](documentation/DEAD_AND_OPEN_LINES.md).
>
> This repo descends from the multivariate base repo "TimeVQVAE-AD-M-2"; that lineage's docs are
> under [documentation/archive/](documentation/archive/) and the multivariate paper under
> `paper_M_backup_20260710/`. Multivariate capability is retained in the code but is **off-path** here.

## What this project is

- **Datasets (univariate, C=1):** `wsd_fed` (real — AnoTransfer KPIs, 31 series / 4 clusters) is the
  target; `toy_fed_uni` (synthetic, 47 series / 6 machine-type clusters) is the control.
  `toy_fed_uni_scarce` / `_scarcer` are data-scarcity probes.
- **Deployed score = pure MaskGIT prior token-NLL** (`weight_s_local = 0`), so every federation
  effect lands on the stage-2 prior.
- **Two named mechanisms under test:** **(A)** merge the VQ codebook by a *sufficient-statistic*
  k-means M-step (= k-FED) instead of naive FedAvg of codeword weights; **(B)** a
  *partially-personalized* prior (local `channel_embedding` + `output_bias`).
- **Headline result (honest):** on real KPIs, **no federated arm beats local** — but we can say
  *why* (the tokenizer dominates the gap; the centralized prize is training-time *diversity*, not
  quantity). The value that *is* real is **Federated Analytics** — federating additive *statistics*,
  not weights (DP-graceful, Byzantine-robust, interoperable, a cold-start lead). See the ledger.

## Architecture

A two-stage pipeline followed by detection (unchanged from the base detector; C=1 here):

1. **Stage 1** — VQ-VAE on an STFT representation. Conv2d encoder/decoder, quantizer with a
   **shared codebook** (single embedding table). It is Stage 1's codebook that mechanism (A) federates.
2. **Stage 2** — MaskGIT prior over Stage 1's discrete tokens. It is this prior that mechanism (B)
   personalizes and that all the distribution-space / Federated-Analytics experiments act on.
3. **Detection** — per-window score (here: prior masked-NLL), rolling-window assembly, per-entity
   quantile threshold. Scored through `pipeline/detect.py`; every federated arm keeps the scaler and
   threshold **local**.

Each concept lives in one file: `model/{common,transforms,encoder,vector_quantizer,decoder,prior}.py`.

## Federation arms

Trained and evaluated by [pipeline/federated_eval.py](pipeline/federated_eval.py); federation
mechanics in [pipeline/federated.py](pipeline/federated.py). Grouped (full list + results in the ledger):

| Arm | What it does |
|---|---|
| `local` | each client alone — the floor to beat. |
| `centralized` | one model on pooled data — the skyline. |
| `federated` | full recipe: suff-stat codebook (A) + partial-personal prior (B). |
| `federated_cb_only` | (A) alone: suff-stat codebook, entire prior local. Substrate the FA experiments reuse. |
| `federated_shared` | (B)-ablation: whole prior FedAvg'd (no personalization). |
| `federated_fedavg_cb` | (A)-ablation: naive weight-FedAvg of the codebook. |
| `federated_fedsgd[_pooltok]` | τ-step FedSGD prior + FedAdam server, with per-client vs pooled tokenizer. |
| `federated_protoprior` | FedProto at the prior level (KL to an ensemble consensus; no weight averaging). |
| `federated_enc_{fedavg,fedprox,fedproto}` | the STAGE-1 ENCODER federated three ways — weight-space, weight-space + prox anchor, function-space prototypes. Everything else held fixed. See [FED_ENCODER_ALGOS.md](documentation/FED_ENCODER_ALGOS.md). |
| `federated_enc_commoninit` | their NULL: common encoder init, then nothing federated. Run it or the trio is uninterpretable. |
| `centralized_cap` | diversity-vs-quantity probe (central model on a capped, diverse window budget). |

## Repository structure

```
config.py                # @dataclass Config — single source of truth (default dataset = toy_fed_uni)
data.py                  # loaders, PerEntityScaler, sliding-window dataset, token cache
utils.py / metrics_core.py  # infra + the unified metric suite (threshold-free, PATE, VUS, affiliation)
run.py / run_matrix.py   # per-entity base launchers (multivariate lineage; off-path for the fed study)

pipeline/
  stage1.py stage2.py detect.py         # the base two-stage + detection
  federated.py                          # federation mechanics (codebook suff-stat merge, prior FL, FedProto)
  federated_eval.py                     # THE federated driver: trains + evaluates every arm
  per_entity_eval.py quality_stage1.py quality_stage2.py compare_aggregations.py cf_eval.py cf_channel.py

model/                   # core components, one per file
metrics/                 # vendored TSAD metrics (PATE / affiliation / VUS) — numpy+sklearn only

scripts/                 # see documentation/DISPOSITION.md for the full keep/legacy classification
  # federated experiments + infra:
  run_fed_converged.sh   #   the CONVERGED sweep (the reliability bar: 35 epochs / 18 rounds, seeds 0-3)
  mixture_eval.py feddf_distill.py tail_starvation.py aggregate_all.py   # distribution-space infra
  fa_*.py launch_all_fa.sh                                               # 14 Federated-Analytics experiments
  exp_*.sh                                                               # derived-experiment drivers
  fed_aggregate.py fed_codebook_autopsy.py fed_*_unittest.py            # aggregation + analysis + tests
  build_toy_fed_uni.py build_wsd_fed.py _toy_common.py ensure_dataset.py list_entities.py  # live data
  # (legacy base-repo downloaders/generators for the 39-dataset MV era live here too — see DISPOSITION.md)

documentation/           # RESEARCH_LEDGER.md (canonical) + DISPOSITION.md + planning docs + archive/
artifacts/fed_eval/      # run outputs (converged/ = the reliability base; gitignored)
```

## Installation

Python 3.10, CUDA 12.8 (exact pins in `requirements.txt`). The TSAD metrics (PATE / affiliation /
VUS) are **vendored** in [metrics/](metrics/) — no extra pip dependency.

```bash
pip install -r requirements.txt
```

## Running

**Hardware note (this node):** GPU 0 is shared with another user — pin to GPU 1
(`CUDA_VISIBLE_DEVICES=1`). The node is Turing (sm_75): use fp16, **not** bf16. The drivers use the
project interpreter `/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10`.

**The converged federated sweep** (the reliability bar — 6 arms × seeds 0–3 at 35 epochs / 18 rounds):

```bash
bash scripts/run_fed_converged.sh        # writes artifacts/fed_eval/converged/<ds>/<cluster>.json
```

**A single arm / cluster** directly:

```bash
CUDA_VISIBLE_DEVICES=1 python pipeline/federated_eval.py \
    --dataset wsd_fed --cluster c3 \
    --arms local,centralized,federated,federated_cb_only \
    --s1-epochs 35 --s2-epochs 35 --s1-rounds 18 --s2-rounds 18 --local-epochs 2 \
    --batch 128 --seeds 0,1,2,3 --out-dir artifacts/fed_eval/converged/wsd_fed
```

**The Federated-Analytics suite** (mostly inference over the converged `federated_cb_only` base):

```bash
bash scripts/launch_all_fa.sh            # 14 experiments in staggered screens (GPU 1)
```

**Datasets** are materialized on disk under `data/raw/` and provisioned on demand by
`scripts/ensure_dataset.py` (`build_toy_fed_uni.py` for `toy_fed_uni`, `build_wsd_fed.py` for `wsd_fed`).

## Metrics & the decision axis

`detect.py` writes a `report.json` per entity with the full unified suite. **On `wsd_fed` the
decision metrics are VUS-PR / AUPRC / PATE-F1** (base rate ~0.45% → **AUROC is misleading**). The
minimum detectable effect at n=31 is **0.087 VUS-PR**: a matched delta below that is a statistical
tie. All comparisons in the ledger are *matched* over shared `(cluster, seed, entity)` keys.

## Philosophy

- **Flat, debuggable code:** one concept per file, no factory indirection, no `name:` switch.
- **Config as a dataclass:** you edit Python, not YAML; every field is used by the code.
- **Explicit pipeline:** `stage1 → stage2 → detect`; federation is an outer loop, not an orchestrator.
- **Vendored metrics** to avoid upstream pinning conflicts.
- **Honesty-first evidence:** every number carries its training budget and coverage (see the ledger);
  matched comparisons only; effects below MDE are reported as ties.
