# Weight-space vs Function-space — Matched Fusion Findings (2026-07-15)

> Companion to [`RESEARCH_LEDGER.md`](RESEARCH_LEDGER.md). Records the P0-B **matched
> fusion matrix** + **FedDF capacity probe** run on 2026-07-15, and what they
> establish about *why* federating the deep prior fails. Every number here was
> recomputed from raw `report.json` on disk (not the ledger text).

**Provenance.** Scripts: [`scripts/naiveavg_eval.py`](../scripts/naiveavg_eval.py),
[`scripts/feddf_probe.py`](../scripts/feddf_probe.py), launcher
[`scripts/exp_p0.sh`](../scripts/exp_p0.sh). Results:
`artifacts/fed_eval/{naiveavg,mixture,feddf,feddf_probe}/`. All arms scored through
the exact `detect.py` machinery (pure MaskGIT prior token-NLL, `weight_s_local=0`),
so numbers sit on the same axis as `local`/`cb_only`/`centralized`. MDE on wsd ≈ **0.087** VUS-PR.

---

## 0. TL;DR

On **identical** `(cluster,seed,entity)` keys and the **same converged `federated_cb_only`
priors**, four ways of combining them:

- **Weight-space average (NaiveAvg) COLLAPSES** (0.048), while
- **Function-space fusion (Ensemble/mixture) RECOVERS local parity** (0.326 ≈ local 0.390, within MDE).

→ The local priors **do** carry mutually-usable structure; it is **specifically the
weight-averaging operator** that destroys it. But no post-hoc combination — weight
*or* function — reaches the **centralized** prize (+0.135): that requires joint
training on diverse data (capN), it is not latent in separately-trained priors.

The **FedDF capacity probe** rules out the "single model can't represent the mixture"
hypothesis: the distilled student **reaches** the ensemble mean p̄ (KL(p̄‖q)=0.064 on
H(p̄)≈2.12, ~3%). So FedDF < Ensemble is a **train→test transfer gap (proxy-shift)**,
not a capacity limit.

**One open fork remains:** is the weight-space collapse *repairable by alignment*
(a — permutation/mode-barrier) or *fundamental* (b — the priors are genuinely
different functions)? This test **motivates** the aligner; it does not answer it.

---

## 1. Audit (V1/V2) — canonical numbers refreshed

The converged baseline is now **complete**: `converged/wsd_fed` has all 6 arms for
c1/seed3 (the ledger's one gap), so the canonical matched set is **n=124** (not 113).

**Matched Δ vs local, multi-metric (converged, n=124):**

| arm | VUS-PR | AUPRC | PATE-F1 |
|---|---|---|---|
| centralized | **+0.135** | +0.141 | +0.154 |
| federated | −0.128 | −0.122 | −0.133 |
| federated_cb_only | −0.047 *(tie)* | −0.043 | −0.047 |
| federated_fedavg_cb | −0.112 | −0.105 | −0.119 |
| federated_shared | −0.145 *(worst)* | −0.150 | −0.163 |

Two red flags the ledger's pooled means hid (all confirmed on VUS-PR/AUPRC/PATE):

- **F1 — `federated < local` is cluster-heterogeneous.** Per-cluster VUS-PR Δ:
  c0 **−0.271**, c1 −0.103, c2 −0.119, c3 **−0.070 (tie, <MDE)**. The >MDE headline
  is carried by c0/c2; on c3 federation ties local.
- **F2 — contribution-(A) is NOT a defensible positive.** `cb_only − fedavg_cb`
  **sign-flips on c0** (VUS-PR −0.041, AUPRC −0.033, PATE −0.021), is positive only
  on c1/c2, and pooled **+0.066 < MDE**. Reframe (A) as *"merge policy is
  second-order"* (consistent with `fa_mergevar`), not a win.

**AUROC is saturated / misleading** (e.g. c2 shows `federated +0.035` on AUROC while
every real metric is negative) — drop it.

**Coverage (07-15):** `complete/centralprior_5400` present but **c3-only, 3 seeds**;
`protoprior` wsd has c0/c2/c3, **missing c1**; `fedsgd_clean`/`pooltok`/`fedenc`/`align` present.

---

## 2. The matched fusion matrix

Five arms, **identical 115 keys** (present in all arms), native readout, VUS-PR
(AUPRC / PATE-F1 give the **same** story — see §2.1):

| arm | ALL | c0 | c1 | c2 | c3 |
|---|---|---|---|---|---|
| local | 0.390 | 0.399 | 0.469 | 0.435 | 0.187 |
| cb_only | 0.337 (−0.053) | 0.295 | 0.472 | 0.325 | 0.140 |
| **NaiveAvg** (weight-space) | **0.048 (−0.342)** | 0.015 | 0.069 | 0.062 | 0.020 |
| **Ensemble** (function-space) | **0.326 (−0.064)** | 0.239 | 0.447 | 0.403 | 0.091 |
| FedDF (distilled) | 0.264 (−0.126) | 0.163 | 0.395 | 0.326 | 0.037 |

- **NaiveAvg** = n_k-weighted post-hoc FedAvg of the full prior state-dicts, scored as
  a single prior. Distinct from `federated_shared` (iterative FedAvg that re-syncs
  every round → bodies never diverge); NaiveAvg averages priors that converged
  **independently from their own init** — the regime where mode-barrier can bite.
- **Ensemble** = the `mixture` soft-min `−logsumexp_k(−nll_k)+log K` (= `−log p̄`).

### 2.1 Multi-metric (ALL, Δ vs local)

| arm | VUS-PR | AUPRC | PATE-F1 |
|---|---|---|---|
| cb_only | −0.053 | −0.050 | −0.056 |
| NaiveAvg | **−0.342** | −0.372 | −0.407 |
| Ensemble | −0.064 | −0.053 | −0.054 |
| FedDF | −0.126 | −0.125 | −0.138 |

The weight-space collapse and the function-space recovery are **robust across all
three real metrics and all four clusters**.

---

## 3. The FedDF capacity probe

**Question:** FedDF (0.264) < Ensemble (0.326) even on matched keys. The mixture score
`−log p̄` is exactly the NLL of the arithmetic mean p̄ = mean_k softmax(teacher_k)
that FedDF distills, so a *perfect* student (q=p̄) would reproduce the mixture. Why doesn't it?
Two candidates: **(i) capacity** — a single prior can't represent p̄ (residual KL>0);
**(ii) proxy≠test** — q≈p̄ on the train proxy but not on test windows.

**Measurement** (masked positions, held-out proxy sample; distillation CE was flat
from ~step 250, i.e. converged — not undertrained):

| cluster·seed | teachers | H(p̄) | KL(p̄‖q) |
|---|---|---|---|
| c0·0 | 5 | 1.988 | 0.102 |
| c0·1 | 5 | 1.842 | 0.097 |
| c1·0 | 11 | 1.996 | 0.043 |
| c1·1 | 11 | 1.922 | 0.046 |
| c2·0 | 9 | 2.336 | 0.057 |
| c2·1 | 9 | 2.472 | 0.051 |
| c3·0 | 6 | 2.267 | 0.059 |
| c3·1 | 6 | 2.175 | 0.058 |
| **mean** | | **2.125** | **0.064** |

**Verdict: (i) capacity is RULED OUT.** KL(p̄‖q)=0.064 on H≈2.12 (~3%) → the student
**reaches** p̄. So the FedDF<Ensemble gap is **(ii) proxy/scoring**: the student matches
p̄ on train contexts, that match doesn't transfer to test-window VUS-PR. Practical
consequence: compressing the ensemble into **one deployable model is feasible** (the
student hits the target); the fix is a **test-matched proxy**, not a bigger student or
more steps.

---

## 4. What this test explained

1. **The failure is the *operator*, not federation.** Same priors, same keys:
   weight-mean → 0.048, function-fusion → 0.326. The local priors are **not mute** —
   they speak a combinable language; weight-averaging turns it to mush. The negative
   result is now specific: **weight-mean is the wrong aggregation operator**, not
   "collaboration is pointless." This resolves the earlier apples-to-oranges (the
   clean matched collapse is 0.048, not the analytic-head 0.006).

2. **There is a ceiling at local.** Ensemble ties local, does **not** reach
   centralized (+0.135). Post-hoc combination recovers what each client already had,
   it does not synthesize the pooling/diversity prize (capN). This collaboration buys
   **parity + privacy, not accuracy**.

3. **The deployable-model route is viable.** The distilled student reaches p̄
   (capacity ruled out); the residual is a train→test transfer gap.

4. **A per-cluster coherence signal.** Function-space recovers local fully in c1/c2
   but only partially in c0 (−0.160) and c3 (−0.096). c0 is the least-coherent cluster
   on **every** axis (worst federated−local, (A) sign-flip, Ensemble under-recovery)
   → prediction for P0-C: the LMC weight-barrier should be **largest in c0**.

### Mapping to the driving question

| Question | Answer from this test |
|---|---|
| Do local priors contain complementary info? | **Yes, but only up to local.** Combinable (Ensemble≈local); the complementarity that would *beat* local (centralized diversity) is **absent**. |
| Why doesn't weight-averaging conserve it? | **Demonstrated that it destroys it** (0.048 vs 0.326); **mechanism not yet explained** (permutation vs genuinely-different function). |
| Recoverable by alignment or only function-space? | **Function-space recovers it** (to local). Whether *alignment* does is **still open** → P0-D. |

**Honest boundary:** NaiveAvg collapse is *consistent with* (a) parameter
misalignment / mode-barrier but does **not prove** it — averaging independent nets can
also collapse from rotation symmetries or from the priors being different functions
(b), which a permutation-aligner won't fix. We learned "weight-mean is wrong," not
"alignment is right."

---

## 5. What is now open / next steps

- **P0-C — LMC barrier, per-cluster** (needs a mini-retrain of the "same-data,
  different-seed" cell): interpolate two converged priors, measure the val-NLL barrier;
  test whether it tracks the per-cluster damage ranking (c0 ≫ c3).
- **P0-D — aligner**: synthetic-permutation round-trip **unit test** first
  (f_θ = f_{P(θ)}; aligner must recover P and P⁻¹P(θ)≈θ), *then* apply to independent
  checkpoints. Adjudicates (a) vs (b). Note: the readout/`token_embedding` are
  index-canonical (shared frozen codebook), so the permutation freedom lives only in
  the Transformer body — vanilla neuron-matching may be insufficient (attention/
  LayerNorm/rotation); expect to need OT-style Transformer fusion.
- **Test-KL follow-up**: measure KL(p̄‖q) on tokenised **test** windows to directly
  confirm (ii) (test-KL ≫ proxy-KL).
- **P1 FedDF full**: test-matched proxy (not more steps — CE already converged).
- **P0-A trivial baseline**: moving-average / Gaussian-NLL detector through detect.py
  to anchor what the 0.05–0.52 VUS-PR range means (raw-signal scorer, different path
  from token scoring — not yet written).

---

## 6. Reproduce

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/exp_p0.sh          # NaiveAvg + FedDF probe + c1 fills
# Matrix aggregation reads report.json globs under artifacts/fed_eval/{converged,naiveavg,mixture,feddf}/
# Probe verdict: artifacts/fed_eval/feddf_probe/records_wsd_fed.jsonl
```
