# Expanding the pool of REAL datasets — verified screening

Written 2026-07-25. **Rewritten the same day**: the first draft of this file was a literature-based
shortlist. Eight candidates were then verified against primary sources *and* against the repo's own
admission funnel run on real bytes, and six of the eight were rejected — including two the first draft
recommended. What follows is the verified version. Companion to [DATASETS.md](DATASETS.md) (inventory)
and [MODEL.md](MODEL.md) (protocol).

**On-disk right now:** only `data/WSD/real-world/` (210 CSVs). Everything else needs a download.

---

## 0. Read this before anything else: the question changed under us

The motivation for a bigger corpus was *"raise the honest n so the centralized prize clears the MDE"*.
**That prize no longer exists**, and this repo already says so:

> `documentation/PROJECT_converged_federation.md:5-7` — "lifted **wsd local 0.375 → 0.617** and
> **centralized 0.522 → 0.585**, flipping the ranking: on real data local now ties/beats centralized
> (Δ −0.032, p=0.23)."
>
> `documentation/FED_ENCODER_ALGOS.md:508` — "**There is no headroom.** On converged wsd, `local`
> 0.617 beats `centralized` 0.585 — the pooled skyline sits *below* the floor. No federated arm can
> exceed a skyline that is already losing."
>
> `documentation/RESEARCH_LEDGER.md:64` — `centralized` ⛔ **SUPERSEDED** … "**The prize inverted.**"

So the primary endpoint is now **mechanism + a TOST equivalence test**, not superiority. Equivalence at
a margin near 0.05 needs `n_eff` ≈ 150–279. **No candidate below reaches even 90.** Adding a corpus for
statistical power is chasing a target that moved.

That reframes the whole question: the only defensible reason left to add a real corpus is
**"does the federated-collapse negative replicate off WSD?"** — a replication argument, not a power one.

---

## 1. The admission rules, as numbers

Read off `scripts/build_frozen.py`. `P` = seasonal period **in samples**, `W` = model window (**128** in
the current config, not 256).

| # | Rule | Constant | What kills a corpus |
|---|---|---|---|
| R1 | N separate univariate streams, ≥4 non-singleton clusters after the funnel | `K=4` | WSD: 210 → 41 → 31 |
| R2 | per-point 0/1 labels | `label` col | window/series labels break VUS-PR + affiliation |
| R3 | wall-clock timestamps + regular Δt | `t0_unix`,`dt_sec` | the descriptor **is** the average day |
| R4 | **train part must contain ZERO labelled anomalies**, + GUARD=P look-back | `build_frozen.py:125` | ⚠️ *this* is the rule everyone misreads as "≥6·P clean samples" |
| R5 | `acf(P) ≥ 0.20` **on a detrended series** | `MIN_PERIOD_ACF` | drift makes it pass vacuously |
| R6 | test anomaly rate ≤ 10 %, ≥1 segment | `MAX_ANOM_RATE` | — |
| R7 | val must hold enough model windows | `VAL_LEN` vs `W` | `data.py:496` **silently copies train into val** |
| R8 | dedup on `|corr|>0.99` **or label-Jaccard>0.50** | `CORR/JACC` | label-Jaccard is the one that bites |

**R4 is the trap.** It is *not* "6·P contiguous clean samples" — it is **zero labelled anomalies in the
train half**. On a corpus where most streams have anomalies early, `best_window` must retreat to a
sub-slice and the admitted client becomes a fraction of its nominal length. This single misreading was
responsible for most of the first draft's over-optimism.

**R7 is the frontier nobody prices.** On WSD `VAL_LEN=P=1440` gives 1185 windows for free. Elsewhere
`VAL_LEN` and client count trade off directly — see the LEAD frontier in §2.

Arm-specific constraints extracted from the code (26 rules, 31 constants), the load-bearing ones:

- **D5** — every cluster needs ≥2 clients, realistically ≥5. Nothing in the code stops a singleton, and
  a singleton "federated" arm is arithmetically identical to `local` while still printing a full table.
  (`build_frozen.py:40`, `MODEL.md:94` "A cluster of one is local-only in disguise.")
- **D6** — `federated_cb_only`/`_ema`/`fedproto` require that **one frozen K=64 dictionary means the same
  thing on every client in a cluster**. Per-client z-scoring already removes level/scale, so it is
  *shape* that must match. If code *j* means different things per client, `M_j/N_j` averages unrelated
  regions of latent space and the merge is degenerate rather than a k-means M-step.
  (`federated.py:588-591`, `1290-1295`.) **Consequence: on a weakly-clustered corpus a cb_only collapse
  is uninterpretable** — you cannot separate a real Prop.1 failure from an incoherent cluster.
- **D8** — if all clients have equal train size, `--fedproto-agg uniform` **is** `count`, i.e. exactly the
  Prop.1 codebook (rel err 2e-8), and the arm becomes its own pre-registered degeneracy control.

---

## 2. Verdict table (scores = suitability *for this study*, 0–100)

| Score | Dataset | Outcome |
|---|---|---|
| **78 / 70** | **LEAD1.0** | ⚠️ **best of a weak field** — all 3 adversarial passes refuted its *rationale*, not its arithmetic |
| 54 | SMD decomposed | ❌ refuted on both lenses — **train contamination** |
| 34 | ESA-ADB | ❌ R3+R5+R8 — worth building only for one scoped factorial |
| 15 | IOPS / AIOps-2018 | ❌ **it IS WSD** — see below |
| 12 | NAB | ❌ admits **zero** clients, structurally |
| 9 | SMAP / MSL | ❌ "anonymized with regard to time" |
| 6 | Microsoft cloud-monitoring | ❌ admits **zero** clients |

### The single most consequential finding: IOPS ≡ WSD
**138 of the 210 WSD series (65.7 %) are bit-identical time-slices of IOPS series, and 23 of the 31
current `wsd_fed` clients (74 %) live inside the IOPS release.** Using IOPS to replicate a WSD result
would be testing the data on itself. The first draft listed IOPS as "near-zero-cost replication" —
it is near-zero-*value* replication. Note also that TSB-AD-U bundles **both** (WSD 111 + IOPS 17).

### NAB is arithmetically impossible, not merely awkward
NAB *defines* anomaly mass as exactly 10 % of every file (arXiv:1510.03336; reproduced numerically at
0.0966 ± 0.0150 across all 47 real series). R4 requires an anomaly-free train half. The two cannot both
hold: the funnel admits **0** clients.

### Microsoft cloud-monitoring — R1 terminal
60 CSVs, not 67 (`git log --diff-filter=A` confirms 67 never existed). Strict protocol → 0 clients;
maximum-generosity ceiling → 4, and 3 after dedup, across 3 domains at 2 sampling rates. 82 % hourly
with modal T=720; 20/60 have duplicate timestamps; R5 passes for 30/60.

### SMAP/MSL — R3 unfixable at the source
The README states the telemetry is **"anonymized with regard to time"**. Without wall-clock there is no
time-of-day descriptor, and that descriptor is the load-bearing construct of the per-cluster thesis.
Max train length 4308 < 6·P=8640 anyway. Plus the Wu & Keogh (ICDE 2022) label critique.

### SMD — the `interpretation_label` claim is true, and it does not save it
Verified: `interpretation_label` is a `start-end:dims` list, **327 lines against 327 `test_label`
segments** (99.4 % coverage), and **73.7 %** of (machine, channel) pairs carry ≥1 anomaly segment. So
per-channel labels really are derivable. But:
- **Train contamination is fatal.** "Labels exist only for the test half, so train is anomaly-free by
  construction" is a non-sequitur, and it was falsified on disk: **189/475 (39.8 %)** of admitted
  channels contain a train excursion at least as extreme as the p99 of their *own* labelled anomalies;
  84.0 % exceed the median labelled-anomaly deviation. Because there are no train labels,
  `build_frozen.py:110`'s clean-window predicate is **vacuously true everywhere** and the whole
  GUARD/clean-window machinery silently no-ops.
- The contamination is **asymmetric across arms**: `local` absorbs its own client's contamination,
  `centralized` dilutes it across 90 clients — a built-in mechanism that *manufactures*
  `centralized > local` for reasons unrelated to diversity. That is the confound this study exists to
  rule out.
- Power: measured ICC(machine)=0.268 → deff 1.56 → **n_eff 55.7**, MDE 0.110 → 0.082. Machine-level
  unit is **n=28 < WSD's 31**. 81.6 % of survivors share ≥1 anomaly event with another survivor.

### ESA-ADB — superb dataset, wrong shape
R3 fails at the source: the paper (§2.3) states the timeline is **"scaled by a non-disclosed factor
larger than 1 and shifted"** → any time-of-day descriptor is fabricated. R5: detrended acf(24 h) =
−0.031 / +0.000 / −0.106 on the three target channels measured (it passes *vacuously* off drift at
+0.97/+0.87/+0.73). R8: label-Jaccard>0.50 collapses Mission 1 from 58 to 7–11 **all singleton**.
Also: on the official common grid every M1 client has *exactly* 7,364,161 train samples → `uniform` ≡
`count` → **fedproto becomes its own degeneracy control** (D8).
Worth building for exactly one thing: per-client volume (0.5M–21M) and client count (6…58) varied
**completely independently**, using the shipped `Group` column and explicitly waiving R8. ~2–3 weeks.

---

## 3. LEAD1.0 — the only survivor, and what the adversaries did to it

Verified on the real file: **200** buildings with public labels (**not** 1413 — that release never
shipped; repo dead since 2022-06-28), Δt **3600 s exactly** on a dense 8784-point 2016 grid,
**P=24 measured** (periodogram peaks at exactly 24.0 h for 61/65; acf(24) median 0.699), per-point
`anomaly` column already int 0/1. Two independent ports of the funnel agree: **121→65** and **117→63**
clients, train min 534–538 / median ~830 / max 3346. Clusters: k=4 sil 0.326, **zero singletons to k=6**,
seed-ARI 0.80–0.99. Download: `raw.githubusercontent.com/samy101/lead-dataset/main/data/lead1.0-small.zip`
(10,027,589 B, md5 `139d7424f7f57b5f00c39c76dc2c6da6`) + BDG2 `metadata.csv` (LFS URL) for `site_id`/
`primary_use`.

**The arithmetic reproduces. The rationale does not.** All three adversarial passes refuted it:

1. **The power gain is ~3 %, not 3.3×.** Four estimators of `n_eff`: 59 (connected components, primary),
   38.1 (Kish, shared-time block bootstrap on arm *differences*), 10.5 (participation ratio), 5.8–12.1
   (spectral, conceded by the primary itself). The component-count metric only works on WSD because
   **90.8 % of WSD entity pairs share zero wall-clock overlap** — no shared driver is *possible*. On LEAD
   every pair is comparable and 47 % exceed |corr|>0.5. Honest move: **31 → ~33 effective units.**
2. **A calendar leak disqualifies the `centralized` arm.** A detector using **zero** information from a
   client's own signal (score = fraction of the *other* 199 buildings flagged at hour *t*) reaches median
   AP 0.079, 4.86× random, and **beats a moving-average signal baseline on 64.0 % of buildings**. 52.9 %
   of all anomalous points fall in hours with ≥10 simultaneous buildings; 28.7 % fall on 15 of 366 days;
   label correlation is **1619×** a circular-shift null. `centralized` can exploit that channel, `local`
   structurally cannot — it mimics exactly the effect under test.
3. **R4 bites hard.** Only **3 of 200** buildings have an anomaly-free first half (median 78 anomalous
   points there). The admitted window is a **median 1755 of 8784 samples = 20 % of the year**, with start
   indices spanning a=0…7721 — clients live in **different months**, so "same 2016 grid, hold volume
   fixed and sweep N" is false as stated.
4. **It lands in the data-scarcity regime already autopsied as broken.** 62 of 63 admitted clients
   (98 %) have **less training data than WSD's worst client**; median 6.4 non-overlapping W=128 windows
   vs WSD's 39.6. Below the kpi_015 threshold the repo's own recon autopsy diagnosed as degenerate
   (recon → low-pass, corr 0.95 with a moving average). Floor effects compress all arms together.
5. **The val/clients frontier**, measured at MIN_TRAIN=512, W=128:

   | VAL_LEN | 168 | 256 | 336 | 512 | 768 | 1024 | 1440 |
   |---|---|---|---|---|---|---|---|
   | clients | **63** | 54 | 45 | **26** | 9 | 9 | **7** |

   n=65 is purchased entirely by a val **8.6× thinner** than WSD's — which attacks the project's 🔴 hard
   rule (train to convergence on val), the rule that exists because truncation left 4/6 WSD entities
   near-random (0.16 → 0.69).
6. **It contaminates the highest-potential open line.** LEAD's readings *are* BDG2/GEPIII, and the LOTSA
   (Moirai) pretraining corpus verifiably contains `bdg-2_{bear,fox,panther,rat}`, `bull`, `cockatoo`,
   `hog`. The frozen-public-TS-FM-body line would be evaluating on data inside the body's training set.

**Which arms LEAD can and cannot serve.** It serves `local`, `federated_enc_commoninit`, and
`federated_enc_{fedavg,fedprox}` mechanically (C=1; 6.22× train-size span keeps D8 non-degenerate;
39k–309k tokens/round/client vs K=64). It does **not** serve `federated_cb_only`/`_ema`/`_norevive`/
`fedproto` **as evidence**, even though they will run: LEAD's clusters score silhouette 0.33–0.35, between
WSD's 0.53 and the descriptor `features.py` itself documents as *rejected noise* (0.14). Per **D6**, a
cb_only collapse there is uninterpretable. And `cb_only_ema`'s γ sweep rides on a **41-window val**.

**Pipeline adaptations:** `data.py:324` add `"lead_fed"` to the loader gate (1 line); `features.py` `DT`
60→3600 (module-level, 3 lines — **both the primary agent and one adversary got a wrong first run until
they patched this**); new `build_lead_fed.py` (~400 lines); `VAL_LEN` decoupled from `PERIOD`,
`MIN_TRAIN` 48→512, `STRAIGHT_MIN` 32→24 — the whole period-derived family, and the funnel is highly
sensitive (MIN_TRAIN 512→1024 takes viable 121→27). `metrics_tolerance` must be set explicitly to **~3–4**
(median test-segment length) vs WSD's 14 — which by the project's own buffer-trap note makes the two
corpora's VUS-PR/PATE **non-comparable**: no pooled analysis, no joint MDE.

---

## 4. The cheaper alternative the adversarial pass surfaced

The best argument for LEAD was *"it puts us in the W ≥ 2P regime and turns Threat #3 into a measured
ablation"*. That is **not an ablation** — LEAD changes W/P, Δt, domain, `metrics_tolerance` (~3 vs 14),
per-window anomaly mass (0.051 vs 0.133) and calendar alignment *simultaneously*.

The clean version costs 1–2 days with **zero downloads**: **sweep W on corpora already in hand.** `W` is a
free constant since the period override was removed (`config.py:447-448`), and the `toy_fed_uni_wsdlike`
/ `toy_fed_uni_ucrlike` twins give exact control of `P` with WSD's own length distribution. That isolates
the W/P ratio with everything else held fixed — which is what an ablation means.

---

## 5. Decision

- **If the goal is statistical power: build nothing.** Zero candidates raise the honest unit above WSD's
  n_eff≈26. Equivalence at a 0.05 margin needs n_eff 150–279; the best candidate offers ~33.
- **If the goal is "does the negative replicate off WSD": LEAD1.0, with a scoped claim.** Report `local`
  and the encoder trio; **do not** report `centralized` (calendar leak) and **do not** report cb_only
  family results as evidence (D6, silhouette 0.33). Price the val/clients frontier before committing.
- **First, spend 1–2 days on the W-sweep + toy twins** (§4). It answers the actual methodological question
  for free and tells you whether a second real corpus is even needed.

## 6. Complete inventory — everything considered, ranked

Spec that actually matters: **N clients, each with its OWN train and OWN test + per-point labels.**
(R3/R5 — wall-clock timestamps and `acf(P)≥0.20` — are needed *only* for the average-day clustering
descriptor. Drop the learned-clustering thesis on a given corpus and they stop applying.)

### Tier 0 — already on disk

| # | Dataset | Clients | Own train+test | Status |
|---|---|---|---|---|
| 1 | **`ucr_ad`** | **248** | ✅ `train/`+`test/`+`test_label/`, one file per entity, each a *different* recording | ⚠️ **no `val/`** → `data.py:496` silently copies train into val. ~20 lines to fix. 60 base recordings (not 248 independent), 1 segment/series, no timestamps, `clusters.json` = single `"pool"` |
| 2 | **`wsd_fed`** | 31 | ✅ + `val/` | the rigorous primary; n_eff≈26 |
| 3 | `ucr_pool` | 248 | train 16384 / test 4096 | pretraining corpus for the frozen-body line — **same 248 series as `ucr_ad`**, do not use both |
| 4 | `toy_fed_uni` (+`_wsdlike`,`_ucrlike`,`_scarce`,`_scarcer`) | 47 / 6 clusters | ✅ | synthetic; exact control of `P`, cluster count, scarcity |

### Tier 1 — worth building

| # | Dataset | Clients | Verdict |
|---|---|---|---|
| 5 | **LEAD1.0** | 63–65 | replication only. **Never report `centralized`** (calendar leak: signal-free detector beats a moving average on 64% of buildings). Not a power upgrade (n_eff ~33). License unresolved. Contaminates the frozen-FM line via LOTSA |
| 6 | **NEK** (Network Equipment KPI) | 48 raw | ⚠️ **unverified beyond the sweep** — WSD lineage, per-point labels + unix timestamps, released 2024 with TimeSeriesBench (ISSRE'24), Zenodo 8147768. 22/48 fail R6. Cheapest unexplored lead |

### Tier 2 — one specific experiment only

| # | Dataset | Verdict |
|---|---|---|
| 7 | **ESA-ADB** | 105 target channels. R3 fails (timeline "scaled by a non-disclosed factor and shifted"), R5 fails (detrended acf(24h)≈0), R8 collapses M1 58→7–11 all-singleton. Build **only** for the volume(0.5M–21M)×count(6…58) factorial with `Group` as the clustering and R8 explicitly waived. ~2–3 weeks |

### Tier 3 — usable only with a declared, load-bearing caveat

| # | Dataset | Clients | The caveat |
|---|---|---|---|
| 8 | **SMD decomposed** | ≤1064 → 87 after R8 | **train contamination**: 39.8% of admitted channels hold a train excursion ≥ p99 of their own labelled anomalies; no train labels ⇒ `build_frozen.py:110`'s clean-window predicate is vacuously true and GUARD no-ops. Contamination is *asymmetric* local-vs-centralized |
| 9 | **SMAP/MSL** | 81 unique (`P-2` duplicated) | **normalization leak published into the data**: values pre-scaled with **test-set** min/max (measured train ch0 at −1.477…4.163). Plus train max 4308, 31/81 over the 10% anomaly cap, 120 label-Jaccard>0.50 pairs, S3 download dead (403) |

### Tier 4 — dead, with the disqualifying fact

| Dataset | Why |
|---|---|
| **IOPS / AIOps-2018** | **IT IS WSD** — 138/210 WSD series bit-identical; 23/31 `wsd_fed` clients inside it |
| **NAB** | defines anomaly mass as exactly 10% of every file → an anomaly-free train half cannot exist → **0 clients**, arithmetically |
| **Microsoft cloud-monitoring** | 60 CSVs (not 67); strict funnel → **0 clients**; max-generosity ceiling 4, 3 after dedup |
| **TSB-AD-U / TSB-UAD** | released CSVs are `value…,Label` with **no timestamp column**; and it *contains* WSD (111) + IOPS (17) + NAB + SMD + SMAP + MSL |
| **CESNET-TimeSeries24** | 275k IPs / 283 institutions, but **no ground-truth labels**. Useful only as unlabelled pretraining fuel |
| **Yahoo S5 (A1)** | gated licence; ~1420 points/series |
| **CARE to Compare** (wind) | 36 turbines / 3 farms, native groups — but weather-driven, **no diurnal cycle** → R5 fails, descriptor undefined |
| **UCR archive raw** | superseded by `ucr_ad` on disk |
| **BDG2 / ASHRAE GEPIII** | no anomaly labels (it is LEAD's source) |
| **SWaT · WADI · PSM · Exathlon · MSDS · 3W · MetroPT-3 · SKAB · CATS** | multivariate industrial/server; univariate use is a reinterpretation with no per-channel labels |
| **SGCC electricity theft** | label is per-customer, not per-point |
| **BattLeDIM** | semi-synthetic water network |

## 7. Verify yourself before committing

- **LEAD license is UNRESOLVED.** No LICENSE file; Apache-2.0 was deliberately deleted (commit
  `3d9dc3c5`) three months *before* the data upload. Signal + site metadata are cleanly CC BY-SA 4.0 via
  BDG2; **the labels are not**. Email Pandarasamy Arjunan before any submission.
- **LEAD timestamps: site-local or UTC?** Inferred local from median peak-hour by timezone; no primary
  sentence found. If UTC, the average-day descriptor mixes phases across 14 sites.
- **63–65 is a port, not the repo's builder.** Run the real `build_frozen.py` retarget and record the frontier.
- **The calendar-leak measurement used surrogate detectors.** Cheap to reproduce; do it before writing
  any `centralized` claim on LEAD.
- **ESA claims rest on 6 of 105 channels, all Mission 1.** "No diurnal period" comes from 3 channels;
  the duplicate finding from one group. Zero Mission-2 signal inspected. Also: shipped
  `anomaly_types.csv` gives **149** anomalies not 148; there is **no** 4th "system-defined condition"
  class (Category ∈ {Anomaly, Rare Event, Communication Gap}); "84/21 months" are *training-half*
  lengths, not durations (14 yr / 3.5 yr anonymised); the Mission-2 ZOH target is **18 s**, not 10 s.
- **LEAD anomaly rate:** measured 2.13 % on the shipped CSV vs "~5 %" in the 1st-place solution README.
