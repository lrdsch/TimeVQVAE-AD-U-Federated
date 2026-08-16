# Federated TimeVQVAE-AD

Federated learning for **TimeVQVAE-AD**, a two-stage generative anomaly detector for time series:
a VQ-VAE first learns a discrete **tokenizer** of the signal, then a MaskGIT **prior** models the
tokens it emits.

The two stages cannot be federated independently. Sharing the body of the prior improves
detection **only after** the clients' encoders have been averaged; the identical change is a null
when they have not been, because the prior is indexed by token identity and unaligned clients use
those identities differently. This repository is the code, the experiment recipes and the
per-cell results behind that claim.

Benchmark: the [UCR Time Series Anomaly
Archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCR_TimeSeriesAnomalyDatasets2021.zip),
each series split chronologically across five clients (10/10/20/20/30 % of its training segment)
that never exchange a window.

---

## Contents

```
config.py            every knob, one dataclass; edit Python, not YAML
data.py              loaders, per-entity scaler, sliding windows, token cache
utils.py             paths, seeding, device
metrics_core.py      the metric suite (AUPRC, AUROC, VUS, PATE, affiliation)

model/               encoder · vector quantizer · decoder · MaskGIT prior — one file each
pipeline/
  stage1.py          VQ-VAE over the STFT of a window
  stage2.py          MaskGIT prior over stage 1's tokens
  detect.py          masked token NLL -> per-timestep score -> metrics
  federated.py       the federation itself: FedAvg, BN pooling, sufficient-statistic codebook
                     merge, FedProx, FedProto, partial prior sharing
  federated_eval.py  the driver: trains and evaluates one (series, arm, seed) cell
  cf_eval.py         counterfactual explanations
metrics/             vendored TSAD metrics (numpy + scikit-learn only)
lib/                 process titles, profiling, training-time visualisation

scripts/             build the data · launch · analyse · test        (detailed below)
cohorts/             the two data cohorts of the paper, pinned by fingerprint
artifacts/runs/      the per-cell results every published table is read back from
docs/                REPRODUCE.md · CONFIGURATIONS.md · PREREGISTRATION.md
```

---

## 1. Install

Python 3.10 and, for anything that trains, one CUDA GPU.

```bash
python3.10 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the versions that produced the published runs, torch included (CUDA 12.8
wheels). For a different CUDA version or CPU-only, install torch yourself first from
[pytorch.org](https://pytorch.org/get-started/locally/) and then run the file again — pip keeps
the torch you already have.

**Check the install** — static checks, the unit tests, and the published tables re-derived from
the shipped results. Three minutes, no GPU, no download:

```bash
bash scripts/run_tests.sh
```

```
== 1. static ==            every file byte-compiles · the pipeline imports · the drivers parse
== 2. unit ==              5 tests on the aggregation math
== 3. the paper's numbers  paper2_numbers.py · c50_table.py · fusion_probe.py
== 4. the launcher ==      launch.sh --dry over the development cohort
14 passed · 0 failed · 1 skipped
```

Add `--smoke` to also train every arm end to end on a GPU at collapsed budgets (~40 min). The
"skipped" ones tell you what they need: layer 4 needs the benchmark built (§3), the smoke layer
needs `--smoke`.

---

## 2. Check the published numbers — no GPU, no data

The result of every cell is in `artifacts/runs/`: the per-series summary, the per-client
`report.json`, and the per-timestep score profiles of the `local` arm. Checkpoints are not
shipped (54 GB); nothing below needs them.

| What it does | Command |
|---|---|
| Re-derives every number in the paper that is not read straight from a table, prints `[OK]`/`[FAIL]` per claim, exits non-zero on any disagreement | `python scripts/paper2_numbers.py` |
| Table IV — the confirmation sample, n=48: win/loss/tie, median, mean and a two-sided sign test on both endpoints, checked against the printed table | `python scripts/c50_table.py` |
| Score fusion: averages the five clients' standardized profiles and compares against the per-client mean, the best client and the federated model | `python scripts/fusion_probe.py --run artifacts/runs/zn_main --all-series` |

```bash
python scripts/paper2_numbers.py --json /tmp/numbers.json    # ... and dump the raw values
python scripts/c50_table.py      --json /tmp/table4.json
```

`paper2_numbers.py` gates itself before printing anything: its own top-1 rule must first
reproduce the `local` column of the development table exactly. If that fails, every fusion number
it computed is declared unvalidated instead of reported.

---

## 3. Get the data

The archive is not redistributed here. Download and unzip it so that this path exists:

```
preprocessing/dataset/AnomalyDatasets_2021/UCR_TimeSeriesAnomalyDatasets2021/FilesAreInHere/UCR_Anomaly_FullData/
```

```bash
# build the federated benchmark: one federation per series, five clients each
python scripts/build_ucr_split.py --name ucr_split_w2p --window-mode 2p \
       --shares 10,10,20,20,30 --val-pct 10

# verify what landed on disk against the raw archive, independently
python scripts/check_ucr_split.py --root data/raw/ucr_split_w2p
```

The build writes `data/raw/ucr_split_w2p/` and prints how many series survive: `--window-mode 2p`
sets each series' window to twice its measured period, and with that rule 180 of the 250 archive
series are usable — the rest have no measured period, or the smallest client would hold less than
one window. The verifier re-reads the source `.txt` and asserts that the five shards plus the
validation tail concatenate back to the original training segment exactly, that the test segment
is verbatim, and that both are identical across the cluster.

`--limit 12` builds a 12-series subset in seconds, which is enough to try the launcher out.

A synthetic dataset is also available and needs no download — it is what the unit tests and the
arm smoke run on:

```bash
python scripts/build_toy_fed_uni.py        # 47 series, 6 clusters, under a second
python scripts/list_entities.py --dataset toy_fed_uni
```

---

## 4. Train

### The unit of work

One cell = one (series, arm, seed). `pipeline/federated_eval.py` runs one; `scripts/launch.sh`
runs a whole cohort of them across the GPUs you give it.

A **cohort** file pins which series, which window, which tolerance and which seeds a run covers,
and stamps a fingerprint into every result. Two tags with the same fingerprint are matched by
construction; an ablation is the same cohort with different knobs. The launcher refuses an
`--extra` that touches a pinned axis, because that would produce a run claiming a comparability
it does not have.

```bash
python scripts/cohort.py datasets                     # what is built on disk
python scripts/cohort.py show ucr2p_10                # the 10 development series
python scripts/cohort.py verify c50                   # does every series still resolve?
python scripts/cohort.py new mine --datasets ucr_split_w2p --clusters ucr_001,ucr_011
```

### Launch a configuration

```bash
LAUNCH_GPUS="0 1" SLOTS_PER_GPU=3 bash scripts/launch.sh \
    --cohort ucr2p_10 --tag a2 --arms federated_enc_fedavg \
    --extra "--window-normalization zscore --fed-enc-cb suffstat --fed-enc-prior partial --fed-enc-bn shared"
```

That is configuration **(g)**, the main federated arm: encoder averaged, BN statistics pooled,
codebook merged from sufficient statistics, prior body shared with the channel embedding and the
output bias kept local.

- `--dry` prints the job list and writes nothing.
- Runs are resumable: a job whose output json already exists is skipped, so re-running after an
  interruption costs only what is missing.
- Results go to `artifacts/runs/<tag>/`, logs to `logs/runs/<tag>/`, and `RUN.json` records what
  the tag *is* — cohort, fingerprint, arms, protocol, every flag.
- The launcher ends with a **convergence audit** and names any cell whose best round was its
  last. Those cells are not reportable: the model was still improving when the budget ran out.
  Raise `--s1-rounds` / `--s2-rounds` (they are ceilings; patience stops each run where it
  actually flattens) and launch again.

Every configuration of the paper, as a command:

```bash
bash scripts/paper_runs.sh                 # print them all, launch nothing
bash scripts/paper_runs.sh --run g         # launch one row
bash scripts/paper_runs.sh --run dev       # the development set, in dependency order
bash scripts/paper_runs.sh --run c50       # the pre-registered confirmation sample
```

What each row means, and which knob it changes, is in
[docs/CONFIGURATIONS.md](docs/CONFIGURATIONS.md).

### One cell, directly

```bash
CUDA_VISIBLE_DEVICES=0 python pipeline/federated_eval.py \
    --dataset ucr_split_w2p --cluster ucr_011 --arms federated_enc_fedavg \
    --protocol converged --s1-rounds 300 --s2-rounds 300 --local-epochs 10 \
    --fed-patience-rounds 6 --batch 64 --seeds 0 \
    --window-length 182 --metrics-tolerance 64 \
    --window-normalization zscore --fed-enc-cb suffstat --fed-enc-prior partial --fed-enc-bn shared \
    --out-dir artifacts/runs/a2/ckpt/ucr_split_w2p \
    --out-json artifacts/runs/a2/ucr_split_w2p/ucr_011__federated_enc_fedavg.json
```

`python pipeline/federated_eval.py --help` lists every arm and every knob.

### The arms

| `--arms` | what is shared |
|---|---|
| `local` | nothing — each client trains alone |
| `centralized` | nothing — one model on the pooled windows (the upper reference) |
| `federated_cb_only` | the codebook, merged from per-code sufficient statistics |
| `federated_fedavg_cb_only` | the codebook, by averaging codeword weights instead |
| `federated` | the codebook plus the prior body, with local encoders |
| `federated_enc_fedavg` | the encoder too — with `--fed-enc-*` this is (f), (g) and (h) |
| `federated_enc_fedprox`, `federated_enc_fedproto`, `federated_enc_commoninit` | the encoder by another rule, or by none (the control) |

The decoder is never shared: it takes no part in the score.

### Cost

Training runs to convergence — patience on a held-out validation split, never a fixed round
budget — so one cell takes from twenty minutes to several hours on one 24 GB card. The
development set is 10 series × ~16 configurations and the confirmation sample is 48 × 3; the
whole campaign is on the order of a GPU-month. Section 2 exists so that checking the numbers does
not cost that.

---

## 5. Analyse a run you produced

These read the checkpoint tree that a training run leaves behind, so they need §4 first.

```bash
# kappa: do the five clients assign the same code to the same latent position?
python scripts/latent_probe.py --run-dir artifacts/runs/a2 --all-complete \
       --arm federated_enc_fedavg_bn-shared_prior-partial
```

Alignment is a measured property of the checkpoints, not a label: under (g) the clients tokenize
*identically* (κ = 1.00), with local encoders the median is κ = 0.07, and the codebook is
bit-identical in both — what separates them is whether the clients use it the same way.

```bash
# counterfactual quality: plausibility and specificity, paired on window and rewritten set.
# The labels A2 / local / centralized are resolved to (tag, checkpoint directory) pairs; point
# them at your own tags -- here `main` for the baselines and `a2` for the federated arm.
python scripts/cf_quality.py --series ucr_011 --arms A2,local,centralized \
       --tag-baselines main --tag-federated a2 --out cf_quality.json --dump cf_arrays/

# the counterfactual as deployed: the mask is what the client's OWN threshold declares.
# --starts is in TEST coordinates (on ucr_011 the labelled event is at 1800-2100).
python scripts/cf_explainable.py --tag a2 --series ucr_011 --client 1 --starts 1850 \
       --out cf_ucr_011.npz

# figures
python scripts/fig_framework.py                            # -> a2_framework.pdf/.svg (+ a PNG
                                                           #    proof), the arrangement of (g)
python scripts/fig_overview.py --series ucr_011 --tag a2 --cf-npz cf_ucr_011.npz --out overview.pdf
python scripts/cf_figure.py --npz cf_arrays/A2_ucr_011.npz --out counterfactual.pdf

# the parameter-free floor: squared residual of a centered moving average, same detection path
bash scripts/launch.sh --cohort ucr2p_10 --tag floor --engine floor --heads ma_c --modes local
```

### Where things land

```
artifacts/runs/<tag>/RUN.json                                  what this tag is
artifacts/runs/<tag>/ucr_split_w2p/<series>__<arm>.json        the summary the tables read
artifacts/runs/<tag>/ckpt/ucr_split_w2p/<series>/seed0/<arm+knobs>/<client>/
    report.json    every metric for that client        scores.npz   the per-timestep profile
    stage1.ckpt    the tokenizer                       stage2.ckpt  the prior
```

The summary json uses the **bare** arm name while the checkpoint directory carries the knobs
(`federated_enc_fedavg_bn-shared_prior-partial`). That is how two configurations of one arm avoid
overwriting each other, and it is worth remembering: `latent_probe.py` wants the directory name,
`paper2_numbers.py` reads the bare one.

---

## 6. Tests

```bash
bash scripts/run_tests.sh              # everything below except the smoke, ~3 min, no GPU
bash scripts/run_tests.sh --smoke      # + every arm trained end to end, ~40 min, one GPU
```

Individually:

```bash
python scripts/fed_codebook_unittest.py        # the merged codebook IS the pooled k-means
                                               # M-step, exactly; collecting statistics does not
                                               # mutate the codebook
python scripts/fed_regression_unittest.py      # count-weighted FedAvg, the seed schedule, and
                                               # BN running statistics never entering the mean
python scripts/fed_enc_algo_unittest.py        # FedAvg / FedProx / FedProto, including the
                                               # FedProto degeneracy: count-weighted prototypes
                                               # ARE the merged codebook, uniform ones are not
python scripts/fed_val_selection_unittest.py   # restoring a best round that is not the last
python scripts/fed_cb_server_ema_unittest.py   # the server-side codebook EMA arm

python scripts/build_toy_fed_uni.py && bash scripts/smoke_arms.sh
ARMS=federated_enc_fedavg bash scripts/smoke_arms.sh          # one arm only
```

The smoke collapses the training budgets but keeps the *converged* code path — validation
selection, patience, best-round restore — so what it exercises is what runs for real. Its metrics
are meaningless by construction and it says so on every line; what it proves is that no arm
raises, none fails to write its json, and none silently loses its stopping criterion.

---

## Protocol notes that change numbers

- **The unit of analysis is the series, not the client.** The five clients are shards of one
  signal with correlated scores; counting them as independent observations inflates *n* tenfold.
  Comparisons are paired by series.
- **The primary endpoint is the archive's own accuracy** — one predicted location per series,
  correct within 64 timesteps. It is coarse: on the confirmation sample the arms agree on 31 to
  37 of 48 series, so the effective sample size is 11 to 17. AUPRC is secondary and reported as
  such. AUROC is not reported: it saturates here.
- **Two settings are inherited from the original method, not tuned**: the window is twice the
  measured period, and each window is z-normalized before the transform. Both are pinned in the
  cohort so a run cannot silently omit them.
- **One seed per cell.** Per-series differences below a few hundredths are not effects.
- **Train to convergence.** A truncated cell is excluded, not reported with a caveat.

## Documentation

- [docs/REPRODUCE.md](docs/REPRODUCE.md) — the three levels of reproduction, end to end.
- [docs/CONFIGURATIONS.md](docs/CONFIGURATIONS.md) — every configuration of the paper as an exact
  command, with the tag its results are under.
- [docs/PREREGISTRATION.md](docs/PREREGISTRATION.md) — the analysis plan for the confirmation
  sample, fixed before any of its numbers existed, and what departed from it.

## Credits

The detector this work federates is TimeVQVAE-AD (Lee, Malacarne and Aune); the generative
backbone is TimeVQVAE (Lee et al.). This repository descends from their implementation and keeps
its MIT license — see [LICENSE](LICENSE).

Some in-code commentary is in Italian, the working language of the group that produced the runs;
the documentation and the entry points are in English.
