# RESEARCH LEDGER — Federated TimeVQVAE Anomaly Detection (univariate)

> **This is the single source of truth for the research path.** Written 2026-07-13, consolidating a full audit of every experiment (config, coverage, and *matched* metrics). It supersedes the planning docs `RESEARCH_PLAN.md` / `WORKSHOP_PAPER_PLAN.md` and the pre-federated docs `research-status.md` / old `README.md` (see `documentation/DISPOSITION.md`).
>
> Every model answers a **why**, is tagged with its **training conditions**, carries its **real matched result**, and ends with a **reliability verdict** (trust as-is vs retrain).

---

## 0. The thesis (current, honest)

We federate the two-stage **TimeVQVAE anomaly detector on univariate time series**, training **one model per cluster** (one client = one series) instead of pooling raw data. Two named mechanisms: **(A)** merge the VQ codebook by a **sufficient-statistic** k-means M-step (= k-FED, Dennis ICML'21) instead of naively FedAvg-ing codeword weights; **(B)** a **partially-personalized** MaskGIT prior (local `channel_embedding`+`output_bias`).

Real target: **`wsd_fed`** (AnoTransfer KPI, 31 series / 4 clusters, C=1). Synthetic control: **`toy_fed_uni`** (**47** series / 6 clusters since the 2026-07-16 rebuild). Deployed anomaly score = **pure MaskGIT prior token-NLL** (`weight_s_local=0`), so *all* federation effects land on the stage-2 prior.

> ⚠️ **Toy topology changed 2026-07-16 — every `toy_fed_uni*` number below is stale.** All toy results in this ledger were measured on the previous **14**-series build (cluster sizes 3/2/2/3/2/2); the benchmark now ships **47** series (6/7/9/11/8/6) because 2 clients/cluster is too few to federate. Per-client series lengths are unchanged, so this is a topology change, not a scarcity change. Entity ids were **reused**: `uni_04` was `M2_valve`, it is now `M1_rotary` — **stale checkpoints load without error and give wrong results**. Pre-rebuild data + checkpoints are archived under `data/_archive_pre_rebuild_20260716/` and `artifacts/fed_eval/_archive_pre_rebuild_20260716/`. **`wsd_fed` (the real target, and every headline claim) is unaffected.** Toy arms must be rerun before citing.

> 🕗 **EVERY number in §3 is OLD (arch + protocol), as of 2026-07-21.** The converged protocol (val early-stopping, warmup+cosine, best-on-val restore) plus the new arch (`width_base 16`, `token_embedding_dim 64`, `window 128`) lifted **wsd local 0.375 → 0.617** and **centralized 0.522 → 0.585** — which *flips the ranking*: local now ties/beats centralized (Δ −0.032, p=0.23). Every arm below was measured against a baseline that has since moved by +0.24, with priors that were undertrained by today's standard. Each row is therefore tagged 🕗 **OLD**; see `PROJECT_converged_federation.md` for the retrain plan. **Only three things on disk are NEW-arch:** `local` + `centralized` (`artifacts/converged_all/`, 2026-07-21), `federated_cb_only` (`artifacts/cbonly_converged/`, 2026-07-22), and the in-flight `federated_cb_only_ema` γ-sweep (`artifacts/cbema_sweep/`). Nothing else has been re-measured.

**The honest framing** (what the evidence actually supports): a convergence-controlled benchmark whose headline is **negative-with-a-mechanism** — federating this detector does *not* beat local on real KPIs, and we can say *why*.

> 🔴 **The second half of that sentence was withdrawn 2026-07-27.** It used to read: *"The value that is real lives in Federated Analytics: interoperability, DP-graceful counts, Byzantine robustness, and a cold-start lead."* All four legs failed audit:
> - **interoperability** — `fa_transfer` is withdrawn on BOTH datasets (see the contamination split at Group 4); the "shared codebook = shared language" bridge currently has **no valid support anywhere in the repo**, and the measured cross-client token agreement of 0.0000 is direct evidence *against* it.
> - **DP-graceful** — `fa_dp`'s σ is not an ε: no per-client clipping, unbounded L2 sensitivity, and the noise scale is a function of the private data itself. Worse, the codebook statistic is released **every round** (37–93 of them), so its budget composes exactly like DP-SGD — the "ε paid once" advantage exists only for a genuinely one-shot release such as the token histogram. Downgrade to "robustness to noise".
> - **Byzantine robustness** — coordinate-wise median is FL canon (Yin et al., ICML 2018), **not** an FA primitive: it is non-additive, so the server must see per-client vectors and the arm forfeits secure aggregation. Relabel; do not sell it as FA.
> - **cold-start lead** — `fa_coldstart` is confounded independently of the already-known cherry-picked slice: the `fa_assisted` arm is handed a tokenizer whose codebook was merged over **all** clients' 100% data, i.e. the very windows the `local_scarce` control is denied.
>
> Also: **calling `cb_only`/`cb_ema` "Federated Analytics" is not defensible** and must not enter the paper. The demarcation in the literature is *"queries that would not require optimization when solved in a centralized scenario"* (Elkordy et al., APSIPA TSIP 2023, §2.1) — centrally this task is SGD on a VQ-VAE, and the merge alone is Lloyd, an optimizer with a convergence loop. The "only additive statistics cross the wire" defence is refuted **by name** in that same section: *"each round of federated learning invokes the simplest question in federated analytics after local training: what is the sum of vectors (gradient updates)…"* — if an additive uplink sufficed, FedAvg would be FA. What IS defensible: *the tokenizer is federated by a non-gradient, exactly-additive aggregation primitive rather than by weight averaging* (the same split the survey itself applies to Orchestra: weights "by FL", centroids "by FA"). The **pooled token-count histogram** is canonical FA without qualification — but only conditional on a tokenizer common to all clients, which this system does not currently provide.

---

## 1. Reliability legend (how to read the "verdict" column)

| Tag | Meaning | Publication-grade? |
|---|---|---|
| ✅ **CONVERGED** | Full budget = **35 epochs / 18 rounds×2 local-ep, seeds 0–3** (`run_fed_converged.sh`). Coverage complete. | Yes — cite directly. |
| ✅ **INFERENCE** | Trains nothing new; re-scores the CONVERGED checkpoints. As reliable as its base. | Yes — no retrain needed. |
| ⚠️ **SMOKE / REDUCED** | Deliberately shortened budget (e.g. `s2-epochs 18`, `s1-rounds 10`, 600–900 prior steps). Number is **directional only**. | **No — retrain at 35/18 before citing.** |
| 🔄 **IN-FLIGHT** | A screen is still writing this dir right now. Provisional. | No — wait for completion. |
| 🔬 **NARROW** | Full budget but single-cluster / small-n scope. Locally valid, not generalizable. | Only as a probe; expand scope to generalize. |
| 🕗 **OLD** | Measured at the **OLD arch** (`width_base 4` / `token_embedding_dim 4` / `window 256`) under the **OLD fixed-budget protocol** (no val, no early-stop, no best-restore, `select_on_val=False`) — i.e. before the 2026-07-21 switch. Its baseline has since moved (wsd local 0.375 → **0.617**). The ✅/⚠️ tag next to it describes convergence *within that dead regime* and does **not** carry over. | **No — the number and its verdict are both uninformative. Retrain before citing.** |

**The MDE gate:** on `wsd_fed` (n=31) the minimum detectable effect is **0.087 VUS-PR**. A |Δ| below 0.087 is a statistical **tie**, not a win/loss, regardless of sign. `toy_fed_uni` has no such gate here (reported deltas are descriptive).

**Matched comparison:** every Δ below is a micro-mean over the exact `(cluster,seed,entity)` keys present in *both* the arm and its baseline (`matched_n` given). This is the corrected methodology — unmatched means were the source of an earlier false "beats local" claim.

---

## 2. The narrative arc (why the path went where it did)

1. **Phase 0 (pre-federated, ≤Jun 2026).** Multivariate per-entity detector (SMAP/MSL + synthetic MV toys). No federation. → archived; see `research-status.md`.
2. **Phase 1 (late Jun).** First federated evidence on MV `toy_fed`: the **suff-stat codebook merge (A) beats naive FedAvg-of-codebook** (+0.13–0.28 VUS-PR). *This is the only positive contribution-(A) result — but on C=8 multivariate data.*
3. **Phase 2–3 (Jul 09–10).** Discovery that `-U` = **univariate**, so the Phase-1 win (MV) doesn't back the univariate paper. Repo cleaned to univariate-only; MV paper closed. wsd audit: **federation loses to local** on every metric; heterogeneity driver is **null on real data**; Prop.1 = k-FED (no math novelty).
4. **Phase 4 (Jul 12).** On *converged* wsd the **centralized prize is large and real** (+0.147 VUS-PR, >MDE) yet every federated arm sits below local. New hypothesis: collaborate in **distribution/prediction space**, not weight space → mixture / FedDF / small-τ FedSGD / FedProto-prior.
5. **Phase 5 (Jul 12–13).** Distribution-space arms mostly tie cb_only; **FedProto-prior is the one clear win on toy**. FedSGD collapses. The **pooltok** decomposition shows the **tokenizer dominates** the wsd gap; a pooled tokenizer + a long central prior (`complete/`, in-flight) is the only thing that reaches the centralized skyline.
6. **Phase 6 (Jul 13, Federated Analytics).** Systematic sweep of *statistic*-federation (not weights): DP, Byzantine, onboarding, transfer, cold-start, codebook-usage, BN/input-norm/whitening. Confirms the FA value proposition at local-parity and one above-MDE cold-start lead.

**Two one-line takeaways the whole ledger supports:**
- **Federating the deep prior (weights) fails** on real KPIs — and it's *not* an undertraining artifact (`fedsgd_clean` with a converged tokenizer still collapses to 0.011).
- **What's shareable is the tokenizer + additive statistics.** Pool those and you reach local (pooltok) or, with a central prior, the skyline (`complete/`).

---

## 3. THE MODEL LEDGER

Baselines (matched): **wsd local = 0.375** (n=113, full scope) / **wsd centralized = 0.522**; on the **c3-only** slice used by the τ-experiments **local = 0.158, centralized = 0.483** (c3 is a weak cluster — do not confuse the two locals). **toy local = 0.816 / centralized = 0.869.**

### Group 0 — Frame (the two poles)

| Arm | WHY | Training | wsd Δvs local | toy Δvs local | Verdict |
|---|---|---|---|---|---|
| `local` 🕗 OLD | The floor: each client alone. The bar every federated arm must beat. | ✅ CONVERGED | — (0.375) | — (0.816) | ⛔ **SUPERSEDED** — new-arch value is **0.617** (`artifacts/converged_all/`). Every Δ in this ledger is against the dead 0.375. |
| `centralized` 🕗 OLD | The skyline: one model on pooled data. Upper bound of any collaboration. | ✅ CONVERGED | **+0.147 >MDE** (0.522) | +0.053 (0.869) | ⛔ **SUPERSEDED** — new-arch value is **0.585**, i.e. **below** the new local (Δ −0.032, p=0.23). **The prize inverted.** |

### Group 1 — Codebook federation — the original thesis (A)+(B)

**WHY:** the core contribution. Can we recover the centralized prize by federating only the codebook (suff-stat) and personalizing the prior, without sharing raw data?

| Arm | WHY (specific) | Training | wsd Δvs local (n=113) | toy Δvs local (n=56) | Verdict |
|---|---|---|---|---|---|
| `federated` 🕗 OLD | Full A+B recipe (suff-stat codebook + partial-personal prior). | ✅ CONVERGED | −0.134 >MDE (0.241) | −0.087 (0.729) | ✅ **TRUST** — reliably **< local**. The headline negative. |
| `federated_cb_only` 🕗 OLD | (A) alone: suff-stat codebook, **entire prior local**. Substrate all FA reuses. | ✅ CONVERGED | −0.051 (0.324) *tie* | −0.061 (0.755) | ⚠️ **RE-RUN EXISTS** — the only federated arm redone at the new arch/protocol (`artifacts/cbonly_converged/`, 2026-07-22). Use that, not 0.324. |
| `federated_shared` 🕗 OLD | (B)-ablation: whole prior FedAvg'd (no personalization). | ✅ CONVERGED | −0.150 >MDE (0.225) | −0.101 (0.715) | ✅ **TRUST** — worst arm → personalization *helps but not enough*. |
| `federated_fedavg_cb` 🕗 OLD | (A)-ablation: **naive** FedAvg codebook vs suff-stat. | ✅ CONVERGED | −0.115 >MDE (0.260) | −0.064 (0.752) | ✅ **TRUST**. On wsd, suff-stat (cb_only 0.324) **> naive (0.260)** → contribution (A) reproduces on real data, but both lose to local. |

> **Reading:** the (A) suff-stat-vs-naive contrast *survives* on univariate wsd (0.324 vs 0.260), the one defensible positive. But the whole family is below local — the deep prior is what's starved, and federation can't fix it.

### Group 2 — Distribution-space collaboration (don't average weights; combine predictions)

**WHY:** if the prize is in the deep conditional and weights don't merge (mode barrier), combine the per-client priors in *distribution* space instead.

| Arm | WHY (specific) | Training | wsd result | toy result | Verdict |
|---|---|---|---|---|---|
| `mixture` (mixture_eval) 🕗 OLD | Fuse per-token NLL of all cb_only priors (soft-min/logsumexp). | ✅ INFERENCE | 0.264 ≈ cb_only (−0.002); −0.080 vs local | 0.770 ≈ cb_only (+0.009) | ✅ **TRUST** (as inference). Verdict: **fusion has no juice** — ties cb_only. |
| `best` (hard-min oracle) 🕗 OLD | Ceiling of prior-selection (cheating: per-window min). | ✅ INFERENCE | +0.005 vs cb_only | +0.017 vs cb_only | ✅ **TRUST**. Even the *ceiling* is flat → selection can't help. |
| `feddf_student` 🕗 OLD | Distill the prior-ensemble into one deployable model (soft-KL). | ⚠️ SMOKE (1500 distill steps) | **0.182, −0.159 vs local >MDE** (hurts) | 0.746 ≈ cb_only (−0.010) | ⚠️ **RETRAIN to confirm**, but direction is clear: distillation **adds nothing / hurts on wsd**. |
| `federated_protoprior` (`proto_weight`) 🕗 OLD | FedProto at prior level: per-client prior + λ·KL to ensemble consensus on a shared probe. No weight averaging. | ✅ CONVERGED (toy) / 🔄 IN-FLIGHT (wsd) | 🔄 n=23, inconclusive (0.182, c2 absent) | **+0.044 vs local, +0.100 vs cb_only** | toy ✅ **TRUST — the one clear distribution-space win**. wsd 🔄 **must finish** (proto screen, c0/c2 pending) before any claim. |

### Group 3 — Prior communication + tokenizer pooling (the decomposition that cracked wsd)

**WHY:** does communicating the prior every few steps (FedSGD) recover centralized? And *what part* of the pipeline actually carries the wsd gap — the prior, or the tokenizer? All on **wsd_fed c3** (matched local=0.158, centralized=0.483, n=18).

| Arm | WHY (specific) | Training | wsd c3 result | Verdict |
|---|---|---|---|---|
| `federated_fedsgd` (τ=1…16) 🕗 OLD | Fully-shared prior, τ steps/round + FedAdam, **per-client tokenizer (10-round, REDUCED)**. | ⚠️ SMOKE tok | **collapse 0.012–0.065**, all ≪ local | ⚠️ Directional. τ doesn't matter; prior FL collapses. |
| `federated_fedsgd` **clean** (τ=1/4/16) 🕗 OLD | Same, but **per-client tokenizer at 18-round (CONVERGED)** — removes the undertraining confound. | ✅ tok CONVERGED / prior 600 steps | **still collapse 0.011–0.041** | ✅ **TRUST the conclusion**: collapse is **NOT** a weak-tokenizer artifact. Per-client tokenizer + shared prior is fundamentally broken. |
| `federated_fedsgd_pooltok` (τ=1…16) 🕗 OLD | Freeze the **POOLED (centralized) tokenizer**, then τ-FedSGD the prior. Isolates the tokenizer term. | ✅ tok CONVERGED (35ep) / prior 900 steps | **0.244–0.290** — recovers ≈ full local, ≈16× the collapse | ✅ **TRUST**: **the tokenizer dominates** (+0.23 at every τ). Comms freq (τ) barely matters. Reaches local, **not** skyline. |
| `complete/centralprior_steps{900,5400}` 🕗 OLD | Pooled tokenizer + **centralized** prior, step-based — closes the prior-budget confound. | 🔄 IN-FLIGHT (tok cached converged; prior 900/5400 steps) | 900→0.329; **5400→0.443 ≈ centralized 0.483 (Δ−0.041 <MDE)** | 🔄 **PROVISIONAL but pivotal**: pooled tok + long central prior **reaches the skyline**. Confirms the residual after tokenizer is prior *budget*, not federation. Let it finish. |
| `complete/federated_fedsgd_fedenc` (τ=1/16) 🕗 OLD | Deployable: FedAvg the whole encoder + τ-FedSGD prior (no data pooling). | 🔄 IN-FLIGHT (enc 10-round REDUCED) | τ1 collapse 0.058; τ16 0.224 (≈pooltok) | 🔄 Directional; the deployable federated-tokenizer route lands at pooltok level, not skyline. |
| `federated_fedsgd_align` 🕗 OLD | Deployable: local encoders + probe-MSE alignment + τ prior. | 🔄 IN-FLIGHT (not yet written) | pending | 🔄 Wait. |

### Group 4 — Federated Analytics — inference probes (federate *statistics*, not weights)

> 🔴 **CONTAMINATION SPLIT — read before citing ANY row below (2026-07-27).**
>
> `scripts/mixture_eval.py:174` loads **one** stage-1 — client `have[0]`'s — and tokenizes every client's data with it, while `federated_cb_only` federates **only the codebook** (verified: codeword tables bit-identical across clients, max|diff| 0.00e+00; encoders local and divergent). So any script that feeds a client's own deep prior with `have[0]`'s tokens is scoring a transformer on symbols it never saw. Measured on wsd c0/seed0: cross-client token agreement **0.0000**, codeword-occupancy JS **0.686–0.693** against a ln2 = 0.6931 ceiling (disjoint support); each client's prior scores that stream at **62–93 nats** vs 0.4–1.4 on its own tokens (uniform-prior max is 12.48). ⚠️ these magnitudes are pending re-measurement — a sibling number from the same probe did not reproduce — but the *structural* defect stands regardless. Use `scripts/tau_identity_check.py` and re-run the divergence probe before quoting figures.
>
> - **WITHDRAW** (hard-contaminated, unrecoverable without encoder federation): `fa_transfer` (all 3 arms, **both** datasets — the "report toy instead" mitigation fails by the identical mechanism), `fa_backoff`, `fa_inputnorm`, `fa_onboard`'s `onboard_mixture` arm.
> - **CLEAN** (already load per-client stage1, no change needed): `fa_calibration`, `fa_bnstats`, `fa_mergevar`.
> - **SELF-CONSISTENT BUT FIAT-TOKENIZER** (internally valid, externally mis-specified — cite only with the conditional caveat): `fa_robust`, `fa_cbusage`, `fa_ngram_ho`, `fa_pca`, `fa_rarity`, `fa_dp` (at the shipped `--lam 0`; contaminated for any `--lam>0`), `fa_onboard`'s `onboard_count`, `fa_coldstart`'s `fa_assisted`.
>
> **Consequence:** of the "FA holds at local-parity" claim, only `fa_bnstats` survives unqualified. The "same axis as the converged local/cb_only/centralized arms" sentence repeated in ~11 `fa_*` docstrings is **false** — a deployed cb_only client tokenizes with its OWN encoder, so this layer measures an upper bound no deployment can reach.
>
> Two further defects fixed in code 2026-07-27, both of which had made results look plausible while being degenerate: `fa_backoff` mixed a τ-summed deep NLL against a single-τ count NLL (3× inflation → every λ variant collapsed to the pure count prior, and every threshold-dependent metric it wrote was identically zero); and `mixture_eval` skipped silently and wrote a 0-byte ledger when checkpoints were absent, indistinguishable downstream from "not yet run". **No `fa_*`/mixture result was ever produced through the post-purge path** (`find artifacts -iname 'records_*.jsonl'` returns only the floor's), so nothing needs retracting on that count.

**WHY (umbrella):** the mergeable part of the system (token counts, norms, covariances) *is* exactly federable and those ARE federated-analytics primitives (no optimization when solved centrally — Elkordy §2.1). The **codebook is not one of them**: its merge is additive but sits inside an SGD loop over locally-updated encoders, so it is federated *learning* with a non-gradient aggregation step. All ✅ **INFERENCE** (reuse converged cb_only; reliable as base). Scope wsd c0,c2,c3 + toy, seeds 0–2.

| Arm | WHY (specific) | Headline result | Verdict |
|---|---|---|---|
| `fa_backoff` 🕗 OLD | Regularize the prior's starved tails via a pooled unigram count model (λ interpolation). | toy lam0.1 0.825 (+0.064 vs cb_only, ≈local); **wsd flat ~0.233, −0.110 vs local >MDE** | ✅ Helps toy tails; **doesn't rescue wsd**. |
| `fa_ngram_ho` 🕗 OLD | How high can an *exactly-mergeable* count prior climb (bigram/trigram)? + measures smoothing breaks mergeability. | **toy bigram_trigram 0.923 (+0.107 vs local, +0.163 vs cb_only)**; wsd 0.310 (+0.044 vs cb_only, ties local <MDE) | ✅ **Best FA-A result on toy**; on wsd recovers most of the cb_only→local gap (tie). Strong interpretability story. |
| `fa_calibration` 🕗 OLD | Federate the *threshold* (pooled quantile) for cross-silo comparability. | **Exact tie on all threshold-free metrics** (by construction); slightly worse hard-F1. | ✅ Confirms: threshold-free metrics can't be moved; calibration is a comparability tool, not an accuracy lever. |
| `fa_onboard` 🕗 OLD | Zero-shot new client (LOO) from shared vocab + neighbours' stats. | **toy count 0.802 ≈ local (−0.015)** — nearly free onboarding; wsd mixture 0.240 (−0.104 >MDE) | ✅ **Onboarding ≈ free on toy**; harder on wsd (heterogeneous KPIs). |
| `fa_transfer` 🕗 OLD | Is the shared codebook a shared *language*? Do foreign priors transfer? | toy ties own (+0.002); wsd cross_best +0.167 but **'own' baseline degenerate** | ✅ toy: interoperable. ⚠️ wsd number misleading (weak own-baseline) — report toy. |
| `fa_dp` 🕗 OLD | Gaussian-mechanism DP on the additive histogram vs DP-SGD. | Graceful: σ=1 −0.045(toy)/−0.061(wsd,<MDE); σ=4 −0.096 | ✅ **DP-graceful** — a genuine selling point vs per-round DP-SGD. |
| `fa_robust` 🕗 OLD | Byzantine robustness of *statistic* aggregation vs the mean. | **poison under mean −0.146 (toy); under robust −0.011** (13× less damage); wsd robust neutralizes (+0.002) | ✅ **Clear win** — robust median/trimmed aggregation defends the detector. |
| `fa_cbusage` 🕗 OLD | Fix dead-local-but-alive-pooled codewords via federated usage. | wsd pooled/usage_aware +0.021/+0.023 vs local (**<MDE**); toy tie | ✅ INFERENCE. Directionally right, **effect below MDE** — not a standalone win. |
| `fa_bnstats` 🕗 OLD | Counterfactual: federate encoder BN stats (should we?). | **federated_bn −0.222 >MDE (catastrophic)** | ✅ **Confirms FedBN**: keep BN local. Intended negative. |
| `fa_inputnorm` 🕗 OLD | Federate a single pooled input (mean,std)? | federated_norm −0.079 (favours per-entity) | ✅ Keep per-entity normalization. |
| `fa_pca` 🕗 OLD | Federated ZCA whitening of the latent (fix the metric, not weights). | Ties (wsd +0.008, toy −0.010) | ✅ INFERENCE. **No effect** — whitening doesn't buy a better token language here. |

### Group 5 — Federated Analytics — training-heavy (⚠️ REDUCED budget; several 🔄 IN-FLIGHT)

**WHY:** the three FA ideas that need a retrained prior/codebook rather than pure inference.

| Arm | WHY (specific) | Training | Result | Verdict |
|---|---|---|---|---|
| `fa_rarity` 🕗 OLD | Federate the inverse-freq rarity loss-weight (global rarity = anomaly-adjacent). | ⚠️ SMOKE `s2-epochs 18` · 🔄 c3 partial, toy pending | **ties local_weights (−0.001)** | ⚠️ **RETRAIN at 35ep** to confirm; current signal = **no effect**. |
| `fa_coldstart` 🕗 OLD | **Flagship FA claim:** data-poor silo + shared tokenizer + neighbour counts beats training alone. | ⚠️ SMOKE `epochs 20` · 🔄 c0 done, c2 partial, no c3/toy | **fa_assisted > local_scarce: +0.034 / +0.093 >MDE / +0.118 >MDE** (as frac 10→50%); **does NOT reach oracle** (−0.087/−0.053/−0.039) | ✅ **The one above-MDE FA lead.** ⚠️ but REDUCED + IN-FLIGHT → **finish c3+toy and retrain at full budget** before it's the headline. |
| `fa_mergevar` 🕗 OLD | Does codebook-merge weighting matter (n_k vs uniform vs iterated Lloyd)? | ⚠️ SMOKE `s1-rounds 15` · c0,c3 seed0,1 (n=22) | uniform +0.003, iterated_lloyd +0.008 vs suffstat_nk | ✅ Answers the question: **n_k weighting is inert, iterating the codebook doesn't help** → the simple suff-stat merge is fine. |

### Group 6 — Mechanism / ablation probes (why, not whether)

**WHY:** isolate the *cause* of the centralized prize and test individual knobs.

| Arm | WHY (specific) | Training | Result | Verdict |
|---|---|---|---|---|
| `centralized_cap` (capN) 🕗 OLD | **Decisive probe:** is the centralized win data *quantity* or *diversity*? Train central on 3200 *diverse* windows (= one client's budget). | ✅ CONVERGED (35ep) · 🔬 c3 only, n=18 | **0.568: +0.410 vs local(c3) >MDE; +0.085 vs full centralized (≈tie)** | ✅ **TRUST (narrow).** **Diversity, not quantity** — same budget, drawn diverse, ≈ full centralized. The mechanism of the prize. |
| `jitter` federated 🕗 OLD | Heterogeneity probe: does injected jitter hurt federation? | ✅ CONVERGED · toy | federated −0.113 vs local | ✅ Heterogeneity hurts federation (on synthetic). |
| `federated_enc` / `_enc_partial` 🕗 OLD | Does sharing the encoder close the gap? | ✅ CONVERGED · 🔬 M1,M4 | −0.050 / −0.051 vs local | ✅ **No** — encoder sharing doesn't close it. |
| `federated_fedavgm` 🕗 OLD | Server momentum help? | ✅ CONVERGED · 🔬 M1,M4 | +0.011 vs federated | ✅ **No effect.** |
| `scarce_probe` / `scarcer_probe` 🕗 OLD | Does scarcity *help* federation (variance-reduction hypothesis)? | ✅ CONVERGED · 🔬 M1(,M4) | federated −0.031 / **−0.114** vs local | ✅ **Refutes** scarcity-helps: gap *widens* as data shrinks. |
| `federated_anchor` (b0) 🕗 OLD | FedProto encoder anchor. | ✅ CONVERGED · 🔬 M1 | +0.072 vs federated (M1); vanishes under scarcity / doesn't transfer to M4 | 🔬 Local-only effect; not general. |
| `federated_align` (b1) 🕗 OLD | Probe-MSE encoder alignment. | ✅ CONVERGED · 🔬 M1/M4 | +0.050 (M1); −0.020/−0.002 (M4) | 🔬 Local-only; doesn't transfer. |

### Group 7 — Analytic-head federation (FLARE / PRISM / HALO / via) — CLOSED negative (2026-07-13)

**WHY:** ported the Analytic Federated Learning idea (AFL / FedOPAL, and the FMMVCC fall-detection variant): the prior's readout is linear (`logits = x @ Eᵀ + bias`, [prior.py:309]), so replace it with a ridge head fit in closed form from **additive** Gram stats `R=Σxᵀx, C=Σxᵀ·onehot(token)` → `W=(ΣR_k+λI)⁻¹ΣC_k`, which federates **exactly in one shot**. Target Y = the masked token (self-supervised). Code: `scripts/analytic_common.py`, `flare_eval.py`, `prism_eval.py`, `halo_eval.py`, `flare_via.py`. Scope wsd c3 (n=18).

| Arm | Body source | VUS-PR | Legal? | Note |
|---|---|---|---|---|
| `flare_centralbody` 🕗 OLD | centralized (pooled) | **0.426–0.428** | ❌ | analytic head over a good body beats cb_only (+0.10 >MDE) — but pools data |
| `via_distill_central` 🕗 OLD | distilled central teacher | 0.395 (own 0.469) | ❌ | distillation **preserves** a coherent body's signal |
| `flare_fedsharedbody` / `prism_*` / `halo_*` 🕗 OLD | federated_shared (FedAvg) | 0.032 | ✅ | collapse; **personalization (PRISM) + fusion (HALO) change nothing** |
| `via_fedshared_bias` 🕗 OLD | federated_shared + output_bias | 0.031 | ✅ | bias-fix does not rescue it → collapse is real, not a head artifact |
| `flare_oneclientbody` 🕗 OLD | one client's cb_only | 0.019 | ✅ | collapse |
| **`via_distill_ensemble`** 🕗 OLD | distill per-client cb_only ensemble | **0.011** (own 0.018) | ✅ | **the only fully-legal body → collapses** (not undertraining: own-readout twin dies too) |
| `flare_fedavgbody` / `flare_naivehead` 🕗 OLD | post-hoc FedAvg of cb_only | 0.006–0.007 | ✅ | collapse |

**Verdict — CLOSED negative with complete mechanism.** The analytic head faithfully *reads* a body; it needs a **coherent** one. Every federation-legal way to build a shared body produces an **incoherent** one and the detector collapses: averaging **weights** (FedAvg → 0.03) and distilling the ensemble's **predictions** (→ 0.01) both fail — the per-client priors disagree, and their fusion (in weight- or prediction-space) is mush. Only a **jointly-trained (pooled)** body is coherent (0.43–0.48) → illegal. Distillation itself works (central teacher 0.395/0.469); the blocker is that no legal *teacher* is coherent. The one remaining theoretical door (as FedOPAL uses frozen CLIP): a body **pretrained on genuinely public data** (a time-series foundation model) — out of scope, a separate project.

---

## 4. What is ESTABLISHED (cite these)

> 🕗 **All eight claims below rest on OLD-arch/OLD-protocol runs.** They were "established" within a regime that no longer exists. Claim 2 in particular is **already known to have inverted** (new centralized 0.585 < new local 0.617). Treat this list as *hypotheses to re-establish*, not as citable results, until the retrain in `PROJECT_converged_federation.md` lands.

1. **Federating the detector loses to local on real KPIs.** wsd `federated − local = −0.134` VUS-PR (>MDE, n=113), all 4 federated arms below local; `cb_only` is the least-bad (ties within MDE). ✅ CONVERGED.
2. **The centralized prize is large and real, and it's DIVERSITY not quantity.** `centralized +0.147` (>MDE); `capN` reproduces it with one client's *diverse* budget (+0.085 vs full centralized). ✅ CONVERGED / 🔬.
3. **Prior FL collapse is not an undertraining artifact.** `fedsgd_clean` (converged tokenizer) still collapses to 0.011. ✅.
4. **The tokenizer dominates the wsd gap.** Pooling it lifts collapse→local (`pooltok` 0.24–0.29); τ (comms frequency) is second-order. ✅.
5. **Only a pooled tokenizer + a central prior reaches the skyline** (`complete/centralprior_5400` ≈ 0.443, 🔄 provisional).
6. **Contribution (A) reproduces on univariate wsd** (suff-stat cb_only 0.324 > naive 0.260), the one defensible positive of the original thesis — though both lose to local. ✅.
7. **FA value proposition holds at local-parity:** DP-graceful (fa_dp), Byzantine-robust (fa_robust, 13× less damage), free-ish onboarding + interoperability on toy (fa_onboard/transfer/ngram_ho), FedBN-is-right (fa_bnstats). ✅ INFERENCE.
8. **Suff-stat merge weighting is inert** (fa_mergevar: uniform ≈ n_k ≈ iterated). ✅.

## 5. The single positive LEAD (needs one clean retrain to become a headline)

- **Cold-start FA** (`fa_coldstart`): `fa_assisted` beats `local_scarce` by **+0.093 / +0.118 VUS-PR (>MDE)** at 25–50% data — federation genuinely rescues the data-poor silo. **But** it's ⚠️ REDUCED (epochs 20) and 🔄 IN-FLIGHT (no c3/toy) and doesn't yet reach the full-data oracle. **Action:** finish c3+toy, then **retrain at 35/18** on a fixed cold-start split. This is the most promising thing to harden.
- **FedProto-prior on wsd** (`protoprior`): the only distribution-space win on toy (+0.100 vs cb_only). **Action:** let the `proto` screen finish c0/c2 on wsd before any claim.

## 6. DEAD lines (do not re-open)

**Analytic-head federation (FLARE / PRISM / HALO / via)** — CLOSED (Group 7): the analytic ridge readout works over a *coherent* body (centralized 0.43) but no federation-legal shared body is coherent — averaging weights (0.03) and distilling the per-client ensemble (0.01) both collapse; personalization/fusion don't help. The head reads a body, it can't fix an incoherent one.


Federation-beats-local (D1); scarcity drives the suff-stat advantage (D2, refuted by scarcer_probe); FedProto anchor as a distinct mechanism (D3, = commitment rescale); Prop.1 as a new theorem (D4, = k-FED); heterogeneity drives the benefit on real data (D5, null ρ=+0.095); personalization recovers the loss (D6, the loser *is* personalized); mixture/FedDF/best-oracle beat cb_only (Group 2, all tie/hurt); server momentum / encoder sharing / whitening / BN-federation help (all null-or-negative).

## 7. What must be RETRAINED before it appears in the paper

| Item | Current state | Required |
|---|---|---|
| `fa_coldstart` (the lead) | ⚠️ epochs 20, 🔄 c0/c2 only | Full 35/18, all clusters, fixed split. |
| `fa_rarity` | ⚠️ s2-epochs 18, 🔄 c3 partial | Full 35 ep — or drop (currently null). |
| `feddf_student` | ⚠️ 1500 steps | Longer distill *only if* re-motivated (currently hurts). |
| `protoprior` wsd | 🔄 in-flight | Finish c0/c2 (already CONVERGED-budget). |
| `complete/centralprior` | 🔄 in-flight | Finish 5400-step + toy (already converged tokenizer). |
| `converged/wsd_fed/c1/seed3` | 🔄 4 fed arms missing | Finish `fedconv` screen → closes the only baseline gap. |
| **Everything else** 🕗 OLD | ✅ CONVERGED / ✅ INFERENCE **at the dead arch+protocol** | **Retrain at wb16/td64/win128 + converged protocol + `select_on_val=True`.** The ✅ tag certified convergence within the old regime only. |
| `federated_enc` / `_enc_partial` / `_enc_neck` 🕗 OLD | old-arch, toy M1/M4 only; `_enc_neck` **never run at all** | Re-run all three: `td 4→64` changes the very `ProjectBlock` these arms share, so "encoder sharing doesn't help" was never tested where it could work. Sweep `--enc-split-at`. |

> ⛔ **This section's old closing line — "everything tagged ✅ CONVERGED or ✅ INFERENCE in §3 is publication-grade as-is" — is no longer true and has been removed.** Since 2026-07-21 the correct statement is the inverse: **every row in §3 carries 🕗 OLD and none of it is publication-grade.** ✅ CONVERGED meant "trained to convergence *under the old fixed-budget protocol*", which the new protocol showed was not convergence at all (local +0.24 on the same data). The retrain queue is `PROJECT_converged_federation.md`; the three arms already redone are listed in the §0 banner.
