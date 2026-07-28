# File disposition manifest

> Companion to [`RESEARCH_LEDGER.md`](RESEARCH_LEDGER.md). Records what each file/dir is and where it now lives after the 2026-07-13 cleanup. Nothing was deleted — obsolete docs were **`git mv`-d into `documentation/archive/`** (fully reversible). **No `.py` files were moved** (cross-imports + live training screens); scripts are classified here, not relocated.

## documentation/

| File | Status | Note |
|---|---|---|
| **RESEARCH_LEDGER.md** | 🟢 CANONICAL (new) | Single source of truth: every model → why → training conditions → matched result → trust verdict. |
| **DISPOSITION.md** | 🟢 CANONICAL (new) | This manifest. |
| DEAD_AND_OPEN_LINES.md | 🟢 CANONICAL | Dead/open catalog. Bannered: D7 stale, tokenizer-dominates added; else valid. |
| SPEEDUP_PLAN.md | 🟢 CANONICAL | Live speed-port reference (Phases 0–2 done on `speedup-port`). |
| lessons-learned.md | 🟢 CANONICAL-narrow | Durable SMAP scoring lessons; pre-federated topic but still true. |
| resume-and-telemetry-design.md | 🟢 CANONICAL-infra | Resumable-training / telemetry design; still applies. |
| RESEARCH_PLAN.md | 🟡 STALE (bannered) | Kept for R1–R7 paper backlog; experimental sections historical → ledger. |
| WORKSHOP_PAPER_PLAN.md | 🟡 STALE (bannered) | Kept for paper structure/checklist; headline numbers are MV-historical → ledger. |
| 2311.12550v5.pdf, federated_aggregation.pdf | 🟢 REFERENCE | Papers. Keep. |
| archive/research-status.md | ⚪ ARCHIVED | Pre-federated MV per-entity debug log (2026-05-23). Different project. |
| archive/DATASETS.md, dataset_extended.md, datasets-candidates-analysis.md | ⚪ ARCHIVED | MV base-repo dataset catalogs (39-dataset era). |
| archive/deep-research-report.md | ⚪ ARCHIVED | MV stage-1 design memo. |
| archive/ciss2019-diagnosis.md | ⚪ ARCHIVED | Single MV run diagnosis. |
| archive/code-dump.md, code-walkthrough.md | ⚪ ARCHIVED | 2026-06-02 MV source snapshots; predate all federated code. |
| archive/lines.txt | ⚪ ARCHIVED | MV command scratch cheat-sheet. |

## scripts/ (classified, not moved)

| Group | Files | Status |
|---|---|---|
| **FA / eval infrastructure** | `mixture_eval.py`, `feddf_distill.py`, `tail_starvation.py`, `aggregate_all.py` | 🟢 CANONICAL — shared harness (see ledger Groups 2/4). |
| **FA experiments** | `fa_backoff/ngram_ho/calibration/onboard/dp/robust/transfer/cbusage/bnstats/inputnorm/pca/rarity/coldstart/mergevar.py`, `launch_all_fa.sh` | 🔴 **SHELVED 2026-07-27 — do NOT run `launch_all_fa.sh`.** The suite presupposes a tokenizer common to all clients, but `federated_cb_only` federates only the *codebook* while `mixture_eval._load_pool:174` tokenizes every client with `have[0]`'s **encoder** (cross-client token agreement measured 0.0000, occupancy JS at the ln2 ceiling = disjoint support). `mixture_eval` now RAISES instead of skipping; re-point deliberately via `TVQ_CONVERGED_ROOT=` only after fixing the arm. Contaminated/clean split in RESEARCH_LEDGER. `aggregate_all.py` does **not** glob `fa_*` records (known gap). |
| **Experiment drivers** | `exp_mixture/feddf/fedsgd/fedsgd_clean/protoprior/capN/gate/pooltok/complete.sh` | 🟢 CANONICAL — ledger Groups 2/3/6. |
| **Converged/cluster runners** | `run_fed_converged.sh` (the reliability bar), `run_fed_clusters.sh` (3/3 smoke — historical), `run_scarcity_sweep.py` | 🟢 `run_fed_converged.sh` canonical; `run_fed_clusters.sh` 🟡 smoke-only. |
| **Federated analysis tools** | `fed_aggregate.py`, `fed_codebook_autopsy.py`, `fed_codebook_unittest.py`, `fed_val_selection_unittest.py` | 🟢 CANONICAL — analysis + tests. |
| **Analytic-head experiments (Group 7)** | `analytic_common.py` (shared infra), `flare_eval.py`, `prism_eval.py`, `halo_eval.py`, `flare_via.py` | ⚫ **DELETED** at `b77d65c` (`cleanup_obsolete_code.sh`). The analytic-head-federation negative is CLOSED (ledger §3 Group 7) and its *artifacts* were already purged 2026-07-23, so keeping the code could not have answered a reviewer anyway. Recoverable from history if the frozen-FM-body line is ever reopened — that would be a new research direction, not a restore. |
| **Live dataset infra** | `build_toy_fed_uni.py`, `build_wsd_fed.py`, `_toy_common.py` (imported by the live builder), `ensure_dataset.py` (imported by `run.py`), `list_entities.py`, `check_dataset_integrity.py` | 🟢 CANONICAL — build/verify the two live datasets. |
| **Dormant-but-WIRED (do not delete)** | `download_smap_msl.py`, `download_asd.py`, the 9 `build_toy_*_channel_anomalies.py`, `build_toy_fed.py` (MV) | 🟡 Reachable at runtime via `ensure_dataset._resolve` dispatch (`_TOY_RE`=`^toy_.+_channel_anomalies$`, `_FED_RE`=`^toy_fed…$`, + explicit smap/msl/asd). Unused by the univariate study, but removing them breaks `ensure_dataset` for those names and makes `config.py`'s legacy list lie. Keep unless you also prune both. |
| **True orphans → ARCHIVED to `scripts/legacy/` (2026-07-13, 51 files)** | 29 `download_*.py` (all except smap_msl/asd) incl. `download_datasets.py`; the non-channel-anomaly MV toy builders `build_toy_{analog2,ecg_synth,lorenz,mini_ics,var,periodic32,multivariate_small}.py`; `create_toy_dataset.py`, `create_synthetic_datasets.py`, `build_all_toy_channel_anomalies_32k.py`; all 6 `plot_*.py`; `audit_all_datasets.py`, `test_smd_interpretation.py`, `verify_grouped.py`, `_check_mask_equivalence.py`, `stage2_ablation_T.py`, `rerun_detect_compare_impulse.py` | ⚪ **DONE** — `git mv`-d into `scripts/legacy/` (not imported, not in any `.sh`, not in `ensure_dataset` dispatch → nothing breaks). Reversible. See `scripts/legacy/README.md`. |

## artifacts/fed_eval/ (result dirs)

| Dir | Status |
|---|---|
| `converged/toy_fed_uni/` | 🟢 FINAL (6 clusters × seeds 0–3 × 6 arms). |
| `converged/wsd_fed/` | 🟢 FINAL **except `c1/seed3`** (🔄 `fedconv` screen finishing 4 fed arms). |
| `mixture/`, `feddf/`, `fedsgd/`, `fedsgd_clean/`, `pooltok/`, `capN/`, `jitter/`, `enc_probe/`, `scarce*_probe/`, `fix1_fedavgm/`, `b0_anchor/`, `b1_align*/` | 🟢 FINAL (some NARROW scope — see ledger). `mixture` seed3 has only 3/6 arms. |
| `protoprior/`, `complete/`, `fa_rarity/`, `fa_coldstart/` | 🔄 IN-FLIGHT (screens `proto`, `complete`, `fa_rarity`, `fa_coldstart`). Provisional. |
| `fa_backoff/ngram_ho/calibration/onboard/dp/robust/transfer/cbusage/bnstats/inputnorm/pca/mergevar/` | 🟢 FINAL (inference; `fa_cbusage` c1/seed2 missing). |
| bare `toy_fed_uni/`, `wsd_fed/` (not under `converged/`) | 🟡 SMOKE (3/3/3/3 pilot) — historical, superseded by `converged/`. |
