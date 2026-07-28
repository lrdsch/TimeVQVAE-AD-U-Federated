# Federated Anomaly-Detection Datasets — Inventory

Downloaded for simulating **federated anomaly detection** where a *client* = one univariate
stream and a *group* = a family of streams with similar normal patterns. Data lives under [data/](data/).

Verified with `TimeEnvM` (Python 3.10, pandas 2.3.3) on 2026-07-07.

> **Local footprint (2026-07-09):** the per-dataset `.git` folders were stripped (~340 MB), and the
> **UCR** and **SMD** trees were removed to save disk — neither is in the federated shortlist (UCR is
> evaluation-only, SMD multivariate). Their rows below are kept as the candidate inventory; re-download
> via the source links in **Reproduce** if needed. The active study uses **WSD** (source) and **NAB**
> (native-group replication); **KPI-AIOps** and **LEAD** are retained as optional extra domains.

## Summary

| Dataset | Local path | Source | Series (clients) | Native group | Format | Anomaly labels |
|---|---|---|---|---|---|---|
| **WSD** | [data/WSD/real-world/](data/WSD/real-world/) | [anotransfer/AnoTransfer-data](https://github.com/anotransfer/AnoTransfer-data) | **210** univariate KPI CSVs | none native → **shape clustering** | `timestamp,value,label` | per-point (0/1) |
| **NAB** | [data/NAB/data/](data/NAB/data/) | [numenta/NAB](https://github.com/numenta/NAB) | **58** (7 real + 2 artificial folders) | **folder = domain** (AWS, AdExchange, Traffic, Tweets…) | `timestamp,value` | JSON windows in [data/NAB/labels/](data/NAB/labels/) |
| **KPI-AIOps** | [data/KPI-AIOps/Finals_dataset/](data/KPI-AIOps/Finals_dataset/) | [NetManAIOps/KPI-Anomaly-Detection](https://github.com/NetManAIOps/KPI-Anomaly-Detection) | **29** KPIs (long tables, 3.0 M pts) | coarse: service- vs machine-KPI → clustering | `timestamp,value,label,KPI ID` | per-point (0/1), 2.6% |
| **LEAD1.0** | [data/LEAD/data/lead1.0-small.csv](data/LEAD/data/) | [samy101/lead-dataset](https://github.com/samy101/lead-dataset) | **200** buildings (small ver.) | **building type / site** (see Kaggle for full) | `building_id,timestamp,meter_reading,anomaly` | per-hour (0/1), 2.1% |
| **SMD** | [data/SMD/ServerMachineDataset/](data/SMD/ServerMachineDataset/) | [NetManAIOps/OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly) | **28 machines × 38 dims** (multivariate) | 3 machine groups (`machine-{1,2,3}-*`) | space-sep `.txt`, one col/metric | test_label + interpretation_label |
| **UCR** | [data/UCR/.../UCR_Anomaly_FullData/](data/UCR/) | [UCR TSAD 2021](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/) | **250** univariate series | none (composite) — **not for grouping** | one value/line `.txt` | encoded in filename |
| **Yahoo S5** | [data/Yahoo-S5/](data/Yahoo-S5/) | gated (Webscope / HF) | A1 ≈ 67 real | single real family | — | **manual download required** |

## Per-dataset notes (federated mapping)

### WSD — best "learned-group" candidate ⭐ (the chosen dataset)
210 real KPIs from Baidu/Sogou/eBay, 1-min sampling. No native `family` column → the AnoTransfer
paper clusters KPIs by normal-shape (e.g. 10 clusters). **client = one `<n>.csv`; group = shape cluster.**
Defensible but grouping is *learned*, not read from a native field.

The frozen federated build lives in [data/federated/WSD_frozen/](data/federated/WSD_frozen/):
**31 clients**, 4 train-only clusters (KMeans on the average-day deviation profile, `scripts/features.py`),
`train`/`val`/`test` all NaN-free, 50/50 split. The seasonal period is **measured** at 1440 min (24 h,
`scripts/detect_periods.py`). The model window (128–256 samples) is *shorter* than the period — a whole-period
window shrinks selection from 31 to 18 clients, and down-sampling to shrink the period destroys 16–35% of
the anomaly segments (14 min median). Still, `train` spans ≥ 2 periods and `val` is exactly 1 period, so the average-day
descriptor has something to average and early stopping covers a whole diurnal cycle. Near-duplicates removed
on the time-aligned overlap; a 1-day guard so training never starts inside a post-incident recovery; imputed
straight-line segments masked as NaN (WSD ships them as real values with `label=0`). Rebuild with
`python scripts/build_frozen.py`; protocol in [MODEL.md](MODEL.md).

### NAB — best "natural-group" candidate ⭐
Sub-corpora are already domain families: `realAWSCloudwatch` (17), `realTweets` (10), `realKnownCause` (7),
`realTraffic` (7), `realAdExchange` (6), plus `artificial*` (11). **client = one series; group = folder.**
Coarse groups are turn-key; don't merge all folders into one group. Labels are anomaly *windows* in
`labels/combined_labels.json` / `combined_windows.json`, not an inline column.

### KPI-AIOps 2018 — second KPI domain
29 KPIs, one big long-format table (`phase2_train.csv`, 3.0 M rows; ground truth in `phase2_ground_truth.hdf`;
also a smaller `Preliminary_dataset/`). **client = one `KPI ID`; group = service-vs-machine then cluster.**
Native grouping is only coarse.

### LEAD1.0 — best non-IT (energy) domain ⭐
This repo ships only `lead1.0-small.csv` = **200 buildings**, hourly, one year (`meter_reading` has some
missing values). **client = one `building_id`; group = building type + site (reduce intra-group heterogeneity).**
Fuller training set (200 train + 206 test buildings, 1,413 meters total) is on the
[Kaggle "energy-anomaly-detection" competition](https://www.kaggle.com/competitions/energy-anomaly-detection) —
grab it with the Kaggle API if you want the larger version.

### SMD — decomposed, secondary
Natively **multivariate** (28 machines × 38 metrics, 3 groups). Univariate use is a *reinterpretation*:
**client = one metric on one machine; group = same metric across machines** (optionally ∩ the 3 machine groups).
Channels are unnamed in the public release. `train/`, `test/`, `test_label/`, `interpretation_label/`.

### UCR — evaluation only, not for grouping
250 diverse single-anomaly series across 9 domains. Filename encodes splits:
`<idx>_UCR_Anomaly_<name>_<trainEnd>_<anomStart>_<anomEnd>.txt`. Useful as a general univariate AD baseline;
"groups" across *different* series would be artificial — keep it out of the federated-grouping claim.

Three builds exist in `data/raw/`, all from the same archive:
| build | protocol | use |
|---|---|---|
| `ucr_pool` | train capped at 16 384, test a 4 096-sample stub | public **pretraining** corpus (`scripts/build_ucr_pool.py`) |
| `ucr_ad`   | native, uncapped: train=`[0:trainEnd]`, test=`[trainEnd:]` | non-federated **detection** reference |
| `ucr_split`| native, then each series' train is **partitioned** into 5 clients (10/10/20/20/30 %) + a 10 % val tail; test untouched and shared | **quantity-skew federation** (`scripts/build_ucr_split.py`) |

`ucr_split` is the one place where a UCR "group" is *not* artificial: a cluster is one series, its clients are
disjoint contiguous slices of that series' own train. They are IID by construction, so the arms isolate the term
every other benchmark confounds — `centralized − federated` is pure aggregation loss, with no heterogeneity
excuse, and `centralized` is exactly a `ucr_ad`-style model on the whole train. 226 of 250 series survive the
"every client gets ≥ 256 samples" filter → 226 clusters / 1 130 clients, 833 MB.
Verify a build with `python scripts/check_ucr_split.py` (re-reads the raw `.txt` and re-asserts the partition).

### Yahoo S5 — gated
See [data/Yahoo-S5/README_HOW_TO_GET.md](data/Yahoo-S5/README_HOW_TO_GET.md). Only A1 is real (single family) →
good for intra-group federation, not cross-group heterogeneity.

## Recommended shortlist for the federated protocol
1. **NAB** — natural, explicit groups (per-source folders).
2. **WSD** — many real KPI clients, strong *learned* shape-clusters.
3. **LEAD1.0** — natural `client = building`, non-IT domain (needs careful type+site grouping).
4. **KPI-AIOps** — extra KPI domain (smaller, coarser groups).
5. Yahoo A1 — single real family (intra-group study); SMD — only as an explicitly-decomposed extension.

## Reproduce / re-download
Git repos were shallow-cloned (`--depth 1`); UCR via `curl`. See per-dataset source links above.
To reclaim space you can delete the `.git` folders under `data/*/` (data files are already checked out).
