# Implementation plan — analytic-head federated anomaly detection

> Two independent methods, ported to our univariate federated TimeVQVAE-AD, plus optional fusions.
> Companion to [RESEARCH_LEDGER.md](RESEARCH_LEDGER.md).

## 🔴 STATUS — BUILT, RUN, REFUTED, DELETED. This is a post-mortem, not a to-do list.

**The header of this file said "Plan only — nothing built yet" until 2026-07-30. That was
false and it is the kind of false that costs weeks: a reader takes it as an unexplored line
and re-implements a closed negative.** What actually happened:

* **Built and executed** (2026-07-13). `scripts/analytic_common.py`, `flare_eval.py`,
  `prism_eval.py`, `halo_eval.py`, `flare_via.py` all existed and produced numbers on wsd c3.
* **Refuted.** [`RESEARCH_LEDGER.md`](RESEARCH_LEDGER.md) **Group 7 — CLOSED negative with
  complete mechanism.** Every gate below was run and the answer came back the same way:

  | body | arm | VUS-PR | federation-legal? |
  |---|---|---|---|
  | centralized (pooled) | `flare_centralbody` | **0.426–0.428** | ❌ pools data |
  | distilled central teacher | `via_distill_central` | 0.395 | ❌ |
  | `federated_shared` (FedAvg) | `flare_fedsharedbody` / `prism_*` / `halo_*` | **0.032** | ✅ |
  | one client's `cb_only` | `flare_oneclientbody` | 0.019 | ✅ |
  | distilled per-client ensemble | `via_distill_ensemble` | **0.011** | ✅ |
  | post-hoc FedAvg of `cb_only` | `flare_fedavgbody` / `flare_naivehead` | 0.006–0.007 | ✅ |

  **Verdict:** the analytic head faithfully *reads* a body and cannot fix an incoherent one.
  The F0 kill-switch (gate 1 below) passed on a **pooled** body and failed on **every**
  federation-legal body. PRISM's personalization and HALO's fusions changed nothing —
  gate 3 answered "personalization adds nothing", which the plan itself pre-registered as a
  finding. Gate 2 (exactness) is not what killed the line.
* **Deleted** at commit `b77d65c`. None of the `*_eval.py` drivers or `analytic_common.py`
  survive in the tree — `ls scripts/analytic_common.py` returns nothing.
* **One residue is still live:** Phase 0 step 1 shipped and was never reverted.
  [`model/prior.py:750`](../model/prior.py#L750) still defines `_hidden(tokens) -> x`, called
  from `_logits` at `:772`. That refactor is behaviour-neutral and can stay; do not mistake its
  presence for the line being half-built.

**Do not restart from this plan.** The one door the ledger leaves open is different in kind: a
body **pretrained on genuinely public data** (a time-series foundation model), which is what
FedOPAL gets from a frozen CLIP and what no arm here ever had. That is a separate project —
see the `sota-frozen-fm-federated-head` note, not this file.

Everything below is preserved **as written in 2026-07-13** — except the decision gates at the
end, which now carry their outcomes — because the gates and the reasoning are what make the
negative interpretable. Read it as the record of a refuted hypothesis; the numbers it quotes
as "baselines to compare against" are pre-purge and are not trustworthy on their own.

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

## Decision gates (summary) — **all three were run; outcomes recorded 2026-07-30**

1. **F0 kill-switch** — analytic head over the best shared body must approach local. If not →
   line dead.
   🔴 **FAILED where it counts.** Over the *pooled* body the head works (0.426–0.428); over
   every *federation-legal* body it collapses (0.032 / 0.019 / 0.011 / 0.006). The gate was
   written to be decisive and it was: **line dead.**
2. **F1 exactness** — federated == centralized head numerically. If not → bug in the Gram
   accumulation. ⚪ **Not the failure mode.** The exactness property was never what broke; the
   body was.
3. **P1 personalization** — personalized ≥ shared. If not → personalization adds nothing (a
   finding). 🔴 **Personalization adds nothing** — `prism_*` and `halo_*` sit at the same
   0.03 as the shared head. Recorded as the pre-registered finding, not as a surprise.
