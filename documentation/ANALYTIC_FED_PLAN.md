# Implementation plan — analytic-head federated anomaly detection

> Two independent methods, ported to our univariate federated TimeVQVAE-AD, plus optional fusions.
> Companion to [RESEARCH_LEDGER.md](RESEARCH_LEDGER.md). **Plan only — nothing built yet.**

## Codenames (our names, distinct from the sources)

| Ours | Source (do not use their names in code) | One line |
|---|---|---|
| **FLARE** | Lynn0925/FLICS ("FedOPAL/AFL") | Federated Linear Analytic Readout: a shared frozen body + one **global** ridge token-head, federated **exactly** in one round via additive Gram matrices. |
| **PRISM** | DonatoCerciello ("FMMVCC + FedOPAL") | Personalized Ridge over an Interpolated Shared-body with Multi-view gating: body built by **one-shot FedAvg** of self-supervised encoders + a **personalized** (per-client, zero-comm) proximal, cluster-gated ridge head. |
| **HALO** (optional) | fusion | PRISM's federated body + FLARE's exact global head + optional personalized residual. |

The bet (both): our prior's readout is **linear** — [prior.py:309-310](model/prior.py#L309-L310) `logits = x @ Eᵀ + output_bias`. Fit that head in closed form instead of by SGD. Target `Y` = the true masked token (self-supervised; no anomaly labels). The deep body producing `x` must be **shared & frozen** (Gram sums are only valid in a common feature space).

---

## Phase 0 — shared infrastructure (build once, both methods use it)

1. **Expose the pre-logit hidden state.** In [model/prior.py](model/prior.py) refactor `_logits` → add `_hidden(tokens) -> x` (the tensor fed to `x @ Eᵀ`); `_logits` calls `_hidden`. No behavior change.
2. **`scripts/analytic_common.py`** (new), reusing `mixture_eval._build_cfg/_load_pool/_score_entity`:
   - `collect_feats(frozen_body, loader, mask_scheme) -> (R, C, n)`: run the frozen body on masked windows, gather `x` at masked positions and one-hot targets; accumulate `R = Σ xᵀx` (D×D), `C = Σ xᵀ·onehot(token)` (D×K). D = embed_dim, K = 64.
   - `ridge_solve(R, C, lam, anchor=None) -> W`: `W = (R + lam·I)⁻¹ (C + lam·anchor)` (anchor=0 ⇒ plain ridge).
   - `AnalyticReadoutScorer`: fakes a `Stage2System` (`.stage1`, `.prior`, `score_batch`) exactly like the count-prior scorers in `mixture_eval`, so `detect._compute_entities_raw` runs unmodified. Score modes: `nll = −log softmax(Wx)[tok]`, `oneminusp`, `residual = ‖onehot − Wx‖`.
3. **Score-mode micro-ablation** picked once on wsd c3 seed0 (nll vs oneminusp vs residual), then fixed.

Baselines to compare against (from the ledger, matched): wsd **local 0.375** (full) / **0.158** (c3); **centralized 0.522 / 0.483**; **cb_only 0.324**. MDE gate 0.087. toy local 0.816 / centralized 0.869 / cb_only 0.755.

---

## FLARE — global exact analytic head (implement first, standalone)

**Question:** given a shared frozen body, does the exact one-shot federated ridge head match centralized and beat/tie cb_only?

- **Stage F0 — falsifier (cheapest, ~½ day).** Body = the converged **`centralized`** prior (frozen; upper-bound shared body). Fit ONE global head on pooled features. Score wsd c3, compare vs local/centralized.
  - **Kill-switch:** if FLARE(centralized-body) ≪ local → the linear readout discards the signal; **stop the whole line**. This is the decisive test.
- **Stage F1 — exactness + federation-legal body.** Body = a single **shared** frozen body from **one-shot FedAvg** of the per-client cb_only bodies (`θ = Σ (nₖ/n)θₖ`), frozen. Two heads:
  - `centralized_analytic`: pool `R,C`.
  - `federated_analytic`: `ΣR_k`, `ΣC_k` summed across clients.
  - **Assert numerical equality** `centralized_analytic == federated_analytic` (the AFL exactness result — a headline sanity check). Compare vs cb_only / local.
- **Stage F2 — scope.** Full wsd (c0–c3) + toy, seeds 0–2.
- **Arms:** `flare_global`. **Baselines:** local, cb_only, centralized.
- **Success:** `flare_global ≥ cb_only` and, ideally, `flare_global` closes part of the cb_only→local gap; the exactness assertion holds.

---

## PRISM — personalized proximal cluster-gated head (implement second, standalone)

**Question:** does a personalized (zero-comm) analytic head over a federated-built body beat the shared head and local?

- **Stage P0 — body.** Frozen body = **one-shot FedAvg** of per-client prior bodies (federation-legal; same as F1's body, so reuse it). (Stretch: a dedicated multi-view SSL body — deferred; start with what we have.)
- **Stage P1 — the three PRISM upgrades** (each an arm, additive):
  - `prism_personal`: per-client head `Wₖ = (Rₖ + λI)⁻¹ Cₖ` (no aggregation, zero inference comm).
  - `prism_proximal`: shared head with **anchor** `W = (Σw·Rₖ + μI)⁻¹ (Σw·Cₖ + μ·M0)`, `M0` = the cb_only readout (warm-start).
  - `prism_gated`: **cluster-gated** feature `X = [x ; g]`, `g` = cluster id one-hot (or federated codebook-usage vector) → head routes per cluster.
- **Stage P2 — scope.** Full wsd + toy.
- **Arms:** `prism_personal`, `prism_proximal`, `prism_gated`. **Baselines:** local, cb_only, `flare_global` (the shared-head reference).
- **Success:** `prism_personal > flare_global` (personalization pays) and/or any PRISM arm ties/beats local within MDE.

---

## HALO — fusions (optional, only if FLARE **or** PRISM clears F0/P1)

- `halo_body_flarehead`: PRISM's federated-SSL body + FLARE's exact global head.
- `halo_personal_residual`: FLARE global head + a small per-client analytic residual (personalization on top of the exact global).
- `halo_gated_proximal`: everything (gated + proximal + personalized). One arm to see the ceiling of the combined recipe.

---

## Cross-cutting

- **Scope order:** always wsd **c3** first (n=18, fast, weak cluster = clearest signal), then wsd full + toy.
- **Engineering hooks:** `pipeline/federated.py` (one-shot FedAvg of bodies — reuse `_fedavg_shared`), `scripts/analytic_common.py` (new), `pipeline/federated_eval.py` (register arms) or a standalone `scripts/exp_analytic.sh` driver mirroring the `fa_*` pattern. Prefer the standalone-script route (like `fa_*`) so it reuses converged checkpoints and stays inference-light.
- **Compute:** F0 = inference + one matrix solve (cheap). F1/P1 = one-shot FedAvg body (one pass) + per-client solves (cheap). No multi-round training. GPU1, fp16, `/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10`, screens.
- **Reliability tags** (ledger vocab): F0/F1/P1 over converged bodies = ✅ INFERENCE-grade if the FedAvg body pass is one-shot; a dedicated SSL body (PRISM stretch) would be ⚠️ new-training.
- **Novelty note:** analytic-head federation is published for **supervised classification**; applying it to **self-supervised masked-token anomaly detection over a VQ codebook** is new — publishable as either a positive (analytic federation works) or a clean negative-with-mechanism.

## Decision gates (summary)
1. **F0 kill-switch** — analytic head over the best shared body must approach local. If not → line dead.
2. **F1 exactness** — federated == centralized head numerically. If not → bug in the Gram accumulation.
3. **P1 personalization** — personalized ≥ shared. If not → personalization adds nothing (a finding).
