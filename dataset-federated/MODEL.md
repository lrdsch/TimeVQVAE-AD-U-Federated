# Federated anomaly-detection — model & experiment protocol

Guidance for building the model on the frozen WSD dataset. The goal is a study that is
**hard to attack**: every design choice below closes a specific reviewer objection.

---

## 0. Central claim (what the experiments must prove)

> Grouping **similar** clients and federating within groups beats federating over all clients,
> which beats training each client alone.
>
> **cluster-fed  >  global-fed  >  local-only**

Two ways this can fail, and both must be tested:

1. `cluster-fed ≤ global-fed` → the clustering adds nothing. Thesis dead.
2. `cluster-fed ≈ random-fed` → the gain comes from **smaller cohorts**, not from *similarity*.
   Thesis dead in a subtler way, and this is the objection a reviewer will raise first.

So the primary result is a **three-way** ordering, reported first:

> **cluster-fed  >  random-fed  ≈  global-fed  >  local-only**

`random-fed` is the control that isolates the one factor the paper is about. Without it,
`cluster-fed > global-fed` is uninterpretable — a cluster model sees ~1/4 of the clients, so it
differs from the global model in *two* ways (cohort size **and** cohort homogeneity) at once.

---

## 1. Dataset

Single version: `data/federated/WSD_frozen/` — **31 clients**, 4 clusters, NaN-free.

**The seasonal period is measured, not assumed: 1440 min (24 h).** ACF (peak chosen by *prominence* —
by height you get 30–60 min, which is smoothness, not seasonality) and an ×8 zero-padded periodogram
(un-padded cannot resolve 1440: for N=4176 the bins are 1392 and 2088) agree within ±15% on 30/31
clients, all of them within ±5%, median exactly 1440. See `scripts/detect_periods.py` / `periods.csv`.

**The model window is SHORTER than the period: W = 128–256 samples (2–4 h).** Making it a whole period
is not affordable and down-sampling cannot rescue it:
- `window = 2 × period` (the rule in the model repo's `config.py`) needs a 2-day window — `val` and
  `train` must each be ≥ 2880 — which shrinks selection from **31 clients to 18**. Dead end.
- Aggregating to coarser bins to shrink the period in *samples* is ruled out by the labels: the 63
  anomaly segments are **14 min median, 3 min minimum**. A 5-min bin destroys 16% of them, a 10-min
  bin 35%. And it does not even help — `2 × period` stays 2 days of wall-clock at any bin size.

Consequence for the model: a 256-sample window sees 4.3 h, i.e. 18% of a cycle. With
`scaling: per_entity_standard` and `window_normalization: none` the window keeps its absolute level,
which implicitly encodes time-of-day — a KPI at its nightly trough sits at a different level than at
its afternoon peak. What is lost is *phase-conditional* discrimination ("this level is normal at 03:00,
anomalous at 15:00"). To recover it, feed time-of-day as a feature (`t0_unix` + `dt_sec` are in the
manifest) — do **not** lengthen the window. Put this in Threats to Validity.

Two period-derived constraints survive anyway — `train ≥ 2 periods` and `val = 1 period` — but for reasons
independent of the window (they serve the clustering descriptor and the early-stopping split). The full
construction:

- **Imputed segments are masked as NaN first.** WSD ships runs of ≥ 32 samples with a perfectly
  constant first difference (an exactly-linear ramp, or an exactly-flat plateau) as real values with
  `label = 0`. They are upstream imputation / a dead sensor, and the NaN-free filter cannot see them.
  42 341 points over 36/210 series are masked before window search — otherwise `train` teaches the
  model that a perfect straight line is normal (KPI 143 alone had 23 h of it, 13.5% of its train).
- 1 client = 1 univariate WSD KPI (one of **210 separate** series, 1-min sampling), restricted to
  the longest contiguous NaN-free window `[a, b)` such that **no labelled anomaly falls in
  `[a − GUARD, a + split)`**, with `GUARD` = 1 day. That is: the train part is anomaly-free *and*
  the window starts at least one day after the last incident, so a KPI is not thrown away for
  misbehaving early, but training never begins inside a post-incident recovery. The look-back runs
  on the **original series**, not on the NaN-free run — a KPI often goes missing *because* it broke,
  so a NaN gap must not be allowed to hide a recent anomaly. Given the window, the split point is a
  **fixed 50%**.
- `train` = first 50% minus the last 1440 samples → anomaly- & NaN-free, **≥ 2 periods** (≥ 2880 pts).
  Not a window requirement: the clustering descriptor is the client's *average day*, and a train that
  never sees the diurnal pattern repeat has nothing to average.
- `val`   = last **1440 samples = exactly one period (1 day)** of the train part → anomaly- & NaN-free.
  **All model selection, early stopping and hyper-parameter choice uses `val`. Never `test`.**
  A fixed *duration*, not a fraction, and one whole cycle: a shorter `val` would score early stopping
  on a slice of the day (e.g. only the night). Gives 1313 / 1185 / 929 windows at W = 128 / 256 / 512.
- A client is admitted only if its train shows a daily cycle: `acf(1440) ≥ 0.20` (min 0.252). Again a
  *clustering* requirement, not a window one — a client with no stable day has a meaningless average-day
  descriptor. It was also the lone singleton cluster (KPI 207, `acf(1440) = −0.215`).
- `test`  = remaining 50% → contains ≥ 1 anomaly segment, but is **< 10% anomalous**. A test window
  that is mostly anomalous is a regime change, not an anomaly: "normal" stops being the majority
  class. This is a safety rail, not a tuner — observed rates are median 0.36%, max 3.44%, and only one
  KPI (128, a single segment covering 52% of its test) is excluded by it.
- **Clusters are computed on `train` only** (KMeans k=4, seed 0) — the grouping never sees val/test.
  The descriptor is the client's **average day**: values are `log1p`-scaled, averaged into 24 hourly
  bins by wall-clock time-of-day (`t0_unix` + `dt`), z-normalised, then the **corpus-mean average day
  is subtracted** and it is re-z-normalised. See `scripts/features.py`. Length-invariant and
  phase-correct; the subtraction matters because every WSD KPI shares one strong diurnal cycle
  (without it intra_r 0.494 vs inter_r 0.515 — one blob).
  `k=4` maximises silhouette (0.53) among partitions that are **singleton-free and seed-stable**. At the
  build seed (0), k=5 is also singleton-free but scores lower (0.52) and collapses to a singleton under
  4 of the 5 seeds; k=6 has a singleton even at the build seed. A cluster of one is local-only in disguise.

Selection funnel (see `plots/selection_funnel.png`): 210 → −30 no-anomaly → 180 → −136 no-clean-window
→ −1 anomaly-dominated → −2 no-period → 41 viable → −10 near-duplicate → **31 clients**.

Dedup is on the **time-aligned overlap**, not on resampled shape: two clients are merged when
`|corr| > 0.99` (same underlying series) **or** anomaly-label `Jaccard > 0.5` (same incident).
This is the step that matters — an earlier shape-based dedup let three copies of one incident
(clients 47/53/122, label-Jaccard 0.99) hold 93% of all anomalous points.

Shape of the resulting benchmark:

| property | value |
|---|---|
| clients / clusters | 31 / 4 (sizes 5, 6, 9, 11) |
| train days | median 3.52, min 2.08, max 11.66 (≥ 2 periods by construction) |
| val days | exactly 1.00 (one period) for every client → 1185 windows at W=256 |
| anomalous points | 1 543 total; largest client holds **10.2%**, top-3 hold 28.9% |
| per-client anomaly rate | median 0.36%, max 3.44% (ceiling 10%) |
| anomaly segments/client | 1: 10 · 2: 11 · 3: 9 · 4: 1 (63 segments; 14 min median, 3 min min) |
| anomaly mass per cluster | c0 17% · c1 22% · c2 **52%** · c3 10% (c2 has 9 clients — watch this) |
| train-only cohesion | silhouette **+0.53**, intra_r 0.93, inter_r −0.28 |

```python
import numpy as np, pandas as pd
Z = np.load("data/federated/WSD_frozen/wsd_federated.npz")
clients = {int(s): dict(train=Z[f"{s}_train"],           # (Ttr,) float32, anomaly- & NaN-free
                        val  =Z[f"{s}_val"],             # (Tva,) float32, anomaly- & NaN-free
                        test =Z[f"{s}_test"],            # (Tte,) float32
                        test_label=Z[f"{s}_test_label"]) # (Tte,) 0/1
           for s in Z["ids"]}
man = pd.read_csv("data/federated/WSD_frozen/manifest.csv")  # id, cluster, t0_unix, dt_sec, …
cluster_of = dict(zip(man.id, man.cluster))
# absolute time of point i of a client's window: t0_unix + i * dt_sec   (dt = 60 s)
```

Per-client normalization: fit mean/std on **that client's `train` only** (your `PerEntityScaler`),
apply to train+val+test. Never fit on val or test.

---

## 2. The five conditions (ablation)

Same model, same hyper-params, **same total number of gradient steps** — only the *data sharing*
changes.

| Condition | Training | Purpose |
|---|---|---|
| **local-only** | one model per client, on its own `train` | lower bound — federation must beat this |
| **federated-global** | one model, FedAvg over ALL 31 clients | standard FL, no grouping |
| **federated-random** ⭐ | one model per **random** group, group sizes matched to the real clusters | **control**: isolates cohort size from cohort similarity |
| **federated-per-cluster** ⭐ | one model per cluster, FedAvg within the cluster's clients | the proposed method |
| **centralized** | one model on the **pooled** train of all clients | data-sharing upper bound (no privacy) — context, not the goal |
| *centralized-per-cluster* (optional) | pooled train within each cluster | isolates "grouping" from "federated vs pooled" |

`federated-random` must be run over **≥ 10 random partitions** (different seeds), each reproducing
the real cluster-size distribution (5, 6, 9, 11). Report its mean ± std over **≥ 10 partitions** — at
these sizes the between-partition variance is large and 5 draws cannot separate it from signal. If the
cluster-fed advantage over random-fed is inside that std, the grouping is doing nothing.

Evaluation is always **per client, on its own held-out `test`**, then aggregated. A client is
scored by the model of its condition (its own / global / its random group's / its cluster's /
centralized).

FedAvg details to fix and report: # rounds, local epochs per round, client sampling (use all,
N is small), aggregation weights, and the same total gradient steps across conditions.
Train sizes span 2.1 → 11.7 days (≈ 5.6×), so **n-weighted FedAvg lets the few large clients dominate**.
Run both `weights ∝ n_train` and uniform weights; report both. If the ordering flips between them,
say so.

---

## 3. Evaluation

### Metrics
Per client on `test`, **threshold-free only**: AUROC, AUPRC, **VUS-PR**, **PATE**, **affiliation**
precision/recall. **No point-adjustment** — it inflates scores and is the first thing reviewers
reject. Reuse the metrics in `TimeVQVAE-AD-M/metrics/`.

### Aggregation — weight by event, not by point
Anomalous points are unevenly distributed (largest client 10.2% of the mass) and **10/31 clients
have a single anomaly segment**, so their per-client AUPRC is close to a coin flip. Therefore:

- Report the **unweighted mean ± 95% CI over clients** (each client counts once) as the headline.
  Do *not* pool all points and compute one global AUPRC — that reduces to "how well did we do on
  the 3 biggest clients".
- Report **segment/event-level** metrics (affiliation, VUS-PR) alongside point-level ones.
- Break results out by **anomaly-segment count** (1 segment vs ≥2) — a method that only wins on
  single-segment clients is winning on noise.
- Always give the **per-cluster** breakdown, so a good average isn't hiding a cluster that
  federation actively hurts.

### Significance — use the *effective* number of clients
The 31 clients are **not independent**. After dedup, `correlations.csv` still holds **8 pairs with
|corr| > 0.9** on their time-aligned overlap (correlated KPIs of the same service, but with
*different* anomalies — hence kept). Building the graph of `|corr| > 0.9` edges leaves **26
connected components**.

- Paired test across clients between conditions (Wilcoxon on per-client AUPRC / VUS-PR).
- Report the p-value **together with n_eff ≈ 26**, not n = 31. Better: run the paired test
  **once per correlation component** (one representative, or the component mean), and report that
  p-value as the conservative one. State both.
- ≥ 5 seeds (model init + FedAvg client order); mean ± std.
- Report the three-way ordering explicitly, per-metric table + per-cluster breakdown.

---

## 4. Robustness (the "with/without" checks that pre-empt attacks)

- **Random-grouping control** (§2). The single most important experiment in the paper.
- **Cluster count k**: sweep k ∈ {3,4,5,6,8}; show the cluster-fed > random-fed gap is stable.
  Silhouette / smallest cluster at the build seed (0) over k ∈ {2,3,4,5,6}:
  0.758/11 · 0.471/6 · **0.525/5** · 0.515/3 · 0.536/1. k=5's smallest cluster is a singleton under
  **4 of the 5 seeds** (the build seed happens to avoid it); k=6 is a singleton even at the build seed.
  **k=4 is the largest singleton-free, seed-stable k**, with the highest silhouette among those and 88%
  same-service purity (7/8, p<1e-4). Report the sweep — k=4 is a choice with a rule, not a discovery.
- **Clustering seed**: ARI = **0.956** across 5 seeds (k=4). Near-reproducible, not exact. Re-run
  cluster-fed under ≥ 3 clustering seeds and document it. (The old resampled-window feature: 0.51.)
- **Native groups**: repeat on **NAB** (`data/NAB/`), where groups are the *native* source folders
  (AWS, Tweets, Traffic…), not learned. If cluster-fed wins there too, the effect isn't a clustering
  artifact. (Optionally LEAD / KPI-AIOps as extra domains.)
- **The grouping is validated against something it never saw.** 19 pairs of clients are > 0.9
  correlated in *wall-clock* time — i.e. different metrics of the same service. The descriptor is
  computed per client, in isolation, and never sees cross-client alignment. **7/8 (88%) of those
  pairs land in the same cluster, vs 27% expected under label permutation (p < 1e-4).**
  Only 8 pairs survive at n=31, so this evidence is thinner than it was at n=49 (16/19). Say so.
  Cross-descriptor check: clustering on the average-day profile also produces a positive cohesion gap
  measured on the autocorrelation descriptor, and vice-versa; the old feature produced ≈ 0 on both.
  Cohesion is now strong (silhouette +0.53, intra_r 0.93, inter_r −0.28) — but silhouette is measured
  in the space KMeans optimised, so the 88% purity is the load-bearing number, not the silhouette.

---

## 5. Threats to validity (put this section IN the paper)

1. **Learned grouping, not ground truth.** Clusters are unsupervised (KMeans on the train-only
   average-day deviation profile), like AnoTransfer on WSD. They are stable (ARI 0.956 across seeds)
   and 88% of same-service pairs co-cluster (p < 1e-4, but only 8 pairs), and "same service" is a proxy.
   An earlier descriptor (whole train resampled to 256 points, z-normalised) produced clusters that
   were **noise**: silhouette +0.14, ARI 0.51, and a cohesion gap of ~0 when measured on any other
   descriptor. If you change the descriptor, re-run the purity test before trusting anything.
   Mitigation: the random-grouping control and the NAB native-group replication.
2. **Clients are not fully independent.** WSD KPIs can be correlated metrics of the same service.
   Exact duplicates (same series, or the same incident) are removed — 19 clients, by `|corr| > 0.99`
   or label-`Jaccard > 0.5` on the time-aligned overlap. **8 pairs with |corr| > 0.9 remain**
   (26 correlation components); they are kept because their *anomalies* differ. Mitigation:
   `correlations.csv` is published, and significance is reported at n_eff ≈ 26.
3. **The model window covers 18% of a period, and no weekly cycle is learnable.** The daily period
   (1440 min) is measured and every client shows it at least twice, but the window is 128–256 samples
   (2–4 h): the model has no phase-conditional view unless time-of-day is fed as a feature. Making the
   window a whole period costs 13 of 31 clients (keeps 18); down-sampling to shrink it destroys 16–35%
   of the anomaly segments (14 min median). Weekly seasonality is out of reach entirely — `train` is 3.5 days
   at the median. Models
   therefore see daily cycles only, and the ≈ equal-length `test` may contain weekend regimes never
   observed in `train`. This inflates false positives for *every* condition equally, so it does not
   bias the ordering — but it caps the absolute numbers. State it; do not claim weekly modelling.
4. **Selection / reduced sample.** 210 → 31 via an explicit rule (`plots/selection_funnel.png`).
   The rule is **label-dependent at the client level**: a KPI survives only if some window has an
   anomaly-free first half. Consequences: (a) clients whose anomalies are dense or start immediately
   are absent, so the "clean burn-in" assumption of unsupervised AD is satisfied *by construction*;
   (b) the 30 anomaly-free KPIs are excluded, i.e. the realistic FL case where a client never has an
   anomaly and only false positives matter is **not** measured. Absolute scores are therefore
   optimistic; the *relative* ordering is the defensible claim. Publish the funnel; no cherry-picking
   of *which* pass.
5. **The published data is partly imputed.** 42 341 points across 36/210 raw series are exactly-linear
   ramps or exactly-flat plateaus of ≥ 32 minutes, shipped as real values with `label = 0`. We mask them
   as NaN before window selection, so no client trains on them — but the masking rule is ours, and a
   shorter imputed run (< 32 samples) would slip through. `interp_masked_orig` in the manifest records
   how much was masked per series.
6. **Concept drift is not separated from anomaly.** `test` is as long as `train` and follows it in
   time. Level shifts in `test` that are not labelled will be scored as false positives. Report the
   per-client train→test mean shift (in σ_train) alongside the results.
7. **Simulated federation.** Clients are series from one public corpus: no systems heterogeneity, no
   client dropout, no communication constraints, no real privacy boundary. This is a *statistical* FL
   study. Do not claim FL-systems findings.
8. **Single primary dataset.** Mitigation: NAB (+ optionally LEAD/KPI) replication.
9. **No test leakage:** the window's split point is a fixed fraction; `train`/`val` are NaN- and
   anomaly-free; clustering uses `train` only; normalization fits on `train` only; model selection
   uses `val` only. State each explicitly.

---

## 6. Reproducibility

- Deterministic pipeline, seed 0; each client is a pure function of its own raw file (verified).
- Rebuild data, **in this order** (`build_frozen.py` rmtree's the frozen dir, so anything written into it
  earlier is lost):
  1. `python scripts/build_frozen.py` — funnel + client cohesion; writes `manifest.csv`, `selection.csv`,
     `correlations.csv`, `profile_mean.npy`
  2. `python scripts/detect_periods.py` — `periods.csv`, `plots/periods.png`
  3. `python scripts/build_overview.py`, `python scripts/build_funnel.py`, and
     `python scripts/plot_cluster_profiles.py`
  Caveat on an older number: a pre-selection analysis clustered **all 210 raw WSD series** (a different,
  whole-series descriptor) and reported silhouette ≈ 0.233. That is *not* the cohesion of the 31 frozen
  clients (0.53, on the average-day deviation profile). Don't conflate the two.
- Pin: model init seed, FedAvg client order seed, **clustering seed**, **random-grouping seeds**,
  k (4), profile bins (24), period (1440), window (128 or 256), split fraction (0.50),
  val length (1440 = 1 period), min train (2 periods), min acf@period (0.20), guard (1 day),
  dedup thresholds (corr 0.99 / Jaccard 0.50), straight-line mask length (32), max test anomaly rate (0.10).
- Browse the data: `python dashboard/server.py` → http://localhost:8765.

## 7. Pre-submission checklist
- [ ] **cluster-fed > random-fed** (≥5 partitions) — the control, reported before anything else
- [ ] cluster-fed > global-fed > local, per-metric table + per-cluster breakdown
- [ ] cluster-fed advantage holds across ≥3 clustering seeds (ARI 0.956 — near-stable, not exact)
- [ ] ≥5 model seeds, mean ± CI, paired significance test at **n_eff ≈ 26**, not n = 31
- [ ] event-level metrics reported next to point-level; split by 1-segment vs ≥2-segment clients
- [ ] FedAvg reported under both n-weighted and uniform aggregation
- [ ] threshold-free metrics only; no point-adjust; `val` used for all model selection
- [ ] k-sensitivity sweep (k=4: largest singleton-free k with significant purity — state the rule)
- [ ] NAB native-group replication
- [ ] funnel figure + Threats-to-Validity section included
