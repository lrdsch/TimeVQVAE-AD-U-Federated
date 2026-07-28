# `ucr_split` — experiment plan (6 arms: local, centralized, FedAvg, FedProx, FedProto, common-init)

Status: **PLAN ONLY — nothing has been launched.** Written 2026-07-25.
Dataset: `data/raw/ucr_split` (built + verified 2026-07-25, see `scripts/check_ucr_split.py`).

---

## 1. Why this run is different from every previous one

Every federated result in `documentation/RESEARCH_LEDGER.md` is confounded: clients differ
in **distribution** *and* in **sample count** at once, so `federated < local` never says which
term did it. `ucr_split` removes the first term by construction — a cluster is one UCR series,
its 5 clients are disjoint contiguous slices of that series' own train. Same sensor, same
regime, same units. What is left is quantity skew, 10/10/20/20/30 %.

That buys three things no other dataset here gives:

| quantity | meaning on `ucr_split` |
|---|---|
| `centralized − federated` | **pure aggregation loss.** No heterogeneity excuse is available. |
| `centralized` | *is* a `ucr_ad`-style model on the whole train (the 5 shards reassemble it exactly). |
| `local(p0) … local(p4)` | a 5-point **dose-response curve**: detection quality vs sample count, inside one series. |

The last one is the real prize and it is free: every cluster yields a within-series
data-scarcity curve, and the federated arms can be read as "how far up that curve does
federation lift the small client?" — a question wsd_fed cannot even pose.

**Primary hypothesis (pre-registered).** If federation works at all, it works here. A
federated arm that does not beat `local(p0)` on IID clients has no path to beating it under
real heterogeneity, and that is a publishable negative with a clean mechanism.

---

## 2. The six arms

Names are the exact `--arms` ids in `pipeline/federated_eval.py`. "fedinit" = `commoninit`.

| # | arm id | what it is | models deployed / cluster |
|---|---|---|---|
| 1 | `local` | one model per client, trained on its shard alone. **The arm to beat.** | 5 |
| 2 | `centralized` | one model on the pooled 5 shards = the original train. **Skyline.** | 1 |
| 3 | `federated_enc_fedavg` | encoder weight-average `w̄ = Σ n_k/n · w_k` | 1 encoder + 5 local dec/prior |
| 4 | `federated_enc_fedprox` | FedAvg + local prox anchor `(μ/2)‖w−wᵗ‖²` | idem |
| 5 | `federated_enc_fedproto` | function-space: per-code prototype consensus, **no weight averaging** | 5 encoders |
| 6 | `federated_enc_commoninit` | shared random init, then never synced. **The null.** | 5 encoders |

Arm 6 is not optional. Without it, "FedAvg helps" cannot be separated from "a common starting
point helps" and arms 3–5 are uninterpretable (`documentation/FED_ENCODER_ALGOS.md`).

The codebook is federated in all of 3–6 (`--fed-enc-cb suffstat`, the default) and the
stage-2 prior is fully local (`--fed-enc-prior local`), so **stage 1 is the only federated
surface** and any gap attributes to the encoder-federation algorithm.

### Hyperparameters — FIXED, not swept

Copied verbatim from `scripts/run_converge60.sh` **so the two tables can sit side by side**.
Verified against `logs/converge60/_orchestrator.log` (the 18:30:17 boot, not the aborted
18:26:50 one):

```
--protocol converged --fed-enc-prior local
--s1-rounds 300 --fed-patience-rounds 6 --local-epochs 10
--fedprox-mu 0.1  --fedprox-form loss     # OFFICIAL FedProx (Li 2020)
--fedproto-weight 0.1 --fedproto-agg count # OFFICIAL FedProto (Tan 2022, Eq. 6+8)
--seeds 0
--batch 64                                 # ← the ONE deviation, see below
```

`--batch 64` instead of wsd's 128: the median 10 % client holds 873 stride-1 windows, so
batch 128 gives 7 steps/epoch and the smallest gives 2. 64 doubles the step count without
touching the data. It is applied to **every arm including `local` and `centralized`**, so
step parity between baseline and skyline is preserved (this is the trap the `--batch` help
text warns about).

**Two kill rules are already armed and must be reported whatever they say** — they fired on
smoke runs before:
- `prox_grad_ratio` — persistent AdamW divides the prox gradient by √v̂; at μ=0.1 FedProx may
  collapse onto FedAvg. Run it as published and let the log show it.
- `proto_agg_gap < 5 %` → FedProto is reported as a reweighted commitment term, not a distinct
  mechanism. At `--fedproto-agg count` the aggregation provably *is* the Prop.1 codebook
  (measured rel. err 2e-8), so this is expected to be tight.

---

## 3. The statistical unit — read this before sizing anything

**1130 clients are not 1130 observations, and 226 clusters are not 226 either.**

1. *Within a cluster, all 5 clients share the same test series and the same labels.* Their 5
   records are 5 models scored on **one** anomaly event. They are not independent. The unit is
   the **cluster**, via a macro-mean over its 5 clients — exactly the two-level aggregation
   `scripts/fed_aggregate.py` already implements.
2. *Across clusters, the UCR archive is heavily duplicated.* Stripping the `DISTORTED`/`NOISE`
   prefixes collapses the 226 series into **90 base-name families** (89 of them have >1 member;
   the largest has 9). `DISTORTED1sddb40` and `1sddb40` are the same signal. Counting them
   twice is the same pseudo-replication the ledger's `n_eff` work already had to correct for.
3. For `centralized` the 5 per-client records are the **same model** on the **same test**,
   differing only through each client's `PerEntityScaler`. Near-duplicates by construction.

⇒ **Confirmatory n = 90** (one representative per family), paired Wilcoxon across clusters.
The remaining 136 clusters are a *replication* set, not extra power.

At n=90 paired, this is by a wide margin the best-powered comparison in the project — wsd_fed
gives n=31 with n_eff≈26, and the audit put the honest MDE there at 0.11–0.15 VUS-PR. n=90
should reach roughly half that.

---

## 4. Scope: three tiers

| tier | clusters | jobs | est. slot-hours | est. wall @12 slots |
|---|---|---|---|---|
| **A — probe** | 3 (small / median / large) | 18 | ~25 | **2–3 h** |
| **B — confirmatory** | 90 family representatives | 540 | ~1 180 | **~4 days** |
| **C — replication** | remaining 136 | 816 | ~2 090 | ~7 days |

Tier A is not optional: it is what turns the estimates below into measurements, and it is the
first time arms 2–6 will have run on this dataset at all (only the default `federated` path
was smoke-tested during the build).

**Representative selection** (90 ids, already computed): within each base-name family prefer a
series with **no** `DISTORTED`/`NOISE` prefix — the un-corrupted original — breaking ties by
cost. Note the side effect: the clean versions live in the archive's upper half, so the
representative list skews to ids ≥109. That is correct (we want the real signal, not the
archive's synthetic corruptions) but it must be stated in the paper.

Probe picks: `ucr_162` (cheapest, 0.24 h/arm), `ucr_205` (median, 0.99 h/arm), and one
~3–5 h cluster. **Do not** probe with `ucr_239/240/241` (taichidb, 13–20 h/arm) — they defeat
the purpose of a quick calibration.

---

## 5. Cost model and where it is soft

Anchored on the measured converge60 run (60 jobs, 288.9 slot-hours, median 4.58 h/job,
min 2.59 h, max 9.44 h) via a mechanistic proxy:

```
cost_h ≈ [ rounds · local_epochs · Σ_k ceil(windows_k / batch)  +  stage2_steps ] / steps_per_s
       + Σ_k detection_windows_k / det_windows_per_s
```

with `steps_per_s ≈ 15` (the wsd jobs measured 7–18 depending on contention) and
`rounds ≈ 38` (the converged knee observed on wsd c3).

Result: **median 0.99 h per (arm × cluster)**, min 0.24 h, p90 5.3 h, max 20.4 h.

**Treat these as ±2×.** The soft spots, honestly:
- `steps_per_s` swings 2.6× across the converge60 jobs from contention alone.
- The convergence round count is dataset-dependent; 38 is the wsd knee, not a law. Patience 6
  will find each cluster's own knee, but the *ceiling* is 300 rounds and a cluster that walks
  toward it costs ~8× the estimate.
- Stage-2 is modelled as a per-client floor (~6 000 steps, the patience), not measured here.

**The cost profile inverts relative to wsd.** ucr_split clusters have ~3.5× *less* training
data but ~4× *more* detection work, because the shared test is scored once per client — 5×
the same series. Detection, not training, is the tail.

**The tail is extreme and must be scheduled around it:** the 10 most expensive clusters carry
28 % of the total cost, the 30 most expensive carry 55 %. `ucr_239/240/241` (taichidb, test
190k–250k × 5 clients) alone are ~50 h/arm between them.

---

## 6. Parallelization

Reuse the `run_converge60.sh` slot scheduler verbatim — it already does resumable
GPU×slot packing, and a job whose out-json exists is skipped.

**Hardware.** 2× Quadro RTX 8000 (48 GB each, both currently idle; the "never GPU0" rule is
dead). 16 CPU cores — **the CPU is the wall**, not the GPUs.

**Settings:**
```bash
# 1. enable MPS FIRST — measured 2–3.5× throughput on this box
nvidia-cuda-mps-control -d          # currently NOT running
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=...   # per mps-2gpu-throughput-ceiling notes

SLOTS_PER_GPU=6                     # 12 concurrent; ~7/GPU is the measured ceiling
export DEBUG_NUM_WORKERS=1          # cfg default is 4 → 12 jobs × 4 = 48 procs on 16 cores
```

**Three scheduling rules specific to this run:**

1. **Longest-processing-time-first.** Sort the job list by predicted cost *descending*. The
   cost spread is 85× (0.24 h → 20.4 h); FIFO order leaves a 20-hour single-job tail with 11
   idle slots. LPT is the difference between ~4 days and ~6.
2. **One job = one (cluster × arm).** 540 jobs at median 1 h is the right granularity for a
   12-slot scheduler, and it keeps resumability fine-grained. Batching all 6 arms into one
   job per cluster would give 90 jobs of 6 h — worse packing, and one crash loses 6 arms.
3. **Cap the giants separately.** Run `ucr_239/240/241` (and the ~7 other >6 h clusters) as
   their own low-priority queue, or exclude them from Tier B and put them in Tier C. They are
   3 of 90 families and buy ~1.3 % of the statistical power for ~15 % of the cost.

**Layout** (mirrors converge60):
```
artifacts/ucrsplit/<cluster>__<arm>.json      # one per job, presence ⇒ skip
artifacts/ucrsplit/ckpt/ucr_split/...         # fed_history.json, checkpoints
logs/ucrsplit/<cluster>_<arm>.log
logs/ucrsplit/_orchestrator.log
```

---

## 7. Command template

```bash
PY=/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10
COMMON="--protocol converged --fed-enc-prior local --s1-rounds 300 \
        --local-epochs 10 --fed-patience-rounds 6 --seeds 0 --batch 64"

# arms 1-2 (baselines; no federated flags apply)
$PY -u pipeline/federated_eval.py --dataset ucr_split --cluster ucr_205 \
    --arms local $COMMON --out-json artifacts/ucrsplit/ucr_205__local.json
$PY -u pipeline/federated_eval.py --dataset ucr_split --cluster ucr_205 \
    --arms centralized $COMMON --out-json artifacts/ucrsplit/ucr_205__centralized.json

# arms 3-6
$PY -u pipeline/federated_eval.py --dataset ucr_split --cluster ucr_205 \
    --arms federated_enc_fedavg $COMMON --out-json .../ucr_205__fedavg.json
$PY -u pipeline/federated_eval.py --dataset ucr_split --cluster ucr_205 \
    --arms federated_enc_fedprox --fedprox-mu 0.1 --fedprox-form loss $COMMON --out-json .../ucr_205__fedprox.json
$PY -u pipeline/federated_eval.py --dataset ucr_split --cluster ucr_205 \
    --arms federated_enc_fedproto --fedproto-weight 0.1 --fedproto-agg count $COMMON --out-json .../ucr_205__fedproto.json
$PY -u pipeline/federated_eval.py --dataset ucr_split --cluster ucr_205 \
    --arms federated_enc_commoninit $COMMON --out-json .../ucr_205__commoninit.json
```

Optional saving: for `centralized` add `--eval-clients <cluster>_p0` — it is one model on one
test set, so the other 4 detection passes are near-duplicates. Costs the scaler-induced
spread; saves ~80 % of that arm's detection time. **Default: don't** — keep the record shape
identical across arms so `fed_aggregate.py` needs no special case.

---

## 8. Analysis

Primary metric **VUS-PR** (the ledger established AUROC is the wrong axis on this family).
`metrics_tolerance` is deliberately absent from `ucr_split/metadata.json`, so the VUS/PATE
buffer is `window_length//2 = 64` — identical to `ucr_ad`, which keeps the two comparable.
UCR's measured median anomaly segment is 101 samples, so 64 is defensible; it must be stated.

```bash
python scripts/fed_aggregate.py artifacts/ucrsplit/*.json --primary vus_pr \
       --csv artifacts/ucrsplit/pairs.csv
```

Tables to produce:

- **T1 — headline.** Per arm: cluster-level macro-mean VUS-PR ± CI, paired Wilcoxon vs `local`,
  n=90 families. Sign-consistency alongside p (the honest headline at this n).
- **T2 — the dose-response curve.** VUS-PR vs share ∈ {10,10,20,20,30} %, per arm. This is the
  figure the dataset exists for: does federation flatten the curve (small clients lifted) or
  shift it (everyone lifted equally)?
- **T3 — aggregation loss.** `centralized − arm` per cluster. On IID clients this is the pure
  cost of not pooling.
- **T4 — the null check.** every federated arm vs `commoninit`. Anything not beating the null
  is a shared-init effect, not federation.
- **T5 — kill rules.** `prox_grad_ratio` and `proto_agg_gap` at the reported round.

Stratify T1–T3 by client size (S/M/L, cuts at p0 = 500 / 1880 samples → 26/33/31 families):
the ~60 clients with <256 stride-1 windows are a genuinely different regime and pooling them
with the large ones will hide whatever is there.

---

## 9. Known risks, stated up front

1. **Temporal confound.** Blocks are contiguous and chronological, so the 30 % client sits
   adjacent to val and its early-stopping selection is systematically favoured. Mitigation if
   it bites: rebuild with permuted `--shares` and re-run a subset.
2. **Tiny clients may be pathological, not merely small.** 26 of the 90 representatives have a
   10 % client under 500 samples (≈3 independent windows). If those produce near-random
   detectors they add variance, not signal. T1 stratified by size is the diagnostic; the
   fallback is to re-scope to `--min-client-train 512` (196/250 series survive).
3. **Convergence ceiling.** `--fed-patience-rounds 6` + 300-round ceiling; runs print
   **TRUNCATED, NOT CONVERGED** if the best round is the last. Any cluster with that banner is
   excluded from T1 and re-run with a higher ceiling — the 2026-07-24 hard rule.
4. **Single seed.** converge60 was also single-seed and the memory note flags it as the reason
   its headline is not yet claimable. Here the cluster-level n=90 carries the inference, but a
   3-seed re-run of the *winning* arm is the honest follow-up before any claim ships.
5. **Cost estimates are ±2×.** Tier A exists to replace them with measurements before Tier B
   is committed.

---

## 10. Decision points (need sign-off before launch)

- **Scope:** Tier A → Tier B (90 families, ~4 d), or Tier A → Tier B+C (all 226, ~11 d)?
- **Seeds:** 1 (as converge60) or 3 (×3 cost, proper error bars)?
- **The giants:** exclude `ucr_239/240/241` + the ~7 other >6 h clusters from Tier B, or keep?
