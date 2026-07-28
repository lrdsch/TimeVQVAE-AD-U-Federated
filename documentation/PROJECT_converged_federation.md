# PROJECT — The federated version of the *new* local

**Premise (2026-07-21).** The converged protocol (val early-stopping, warmup+cosine,
best-on-val restore) + new arch (width_base 16, token_embedding_dim 64, window 128)
lifted **wsd local 0.375 → 0.617** and **centralized 0.522 → 0.585**, flipping the
ranking: on real data local now ties/beats centralized (Δ −0.032, p=0.23). **Every
federated arm in the ledger was measured at the OLD arch under the OLD fixed-budget
protocol.** Those verdicts are therefore uninformative — the priors were undertrained.
Before concluding anything about federation we must retrain the federated arms *with the
same protocol that lifted local*. This project defines how, starting from the cleanest
arm.

---

## 1. The starting arm: `federated_cb_only`

| component | status | how it is trained today |
|---|---|---|
| encoder | **LOCAL** (per client) | `federated_stage1` rounds: local SGD + suff-stat codebook merge |
| VQ codebook | **SHARED** | exact sufficient-statistic (k-FED) merge across clients — additive, no SGD |
| prior (stage 2) | **LOCAL** (`local_prefixes=("",)` → 0 shared keys) | `federated_stage2`: `rounds × local_epochs` of `_local_train_prior`, **no val, no early-stop, no best-restore** |

So cb_only is *local everything except the codebook*. That makes the scientific question
razor-sharp and NEW:

> **Does sharing ONLY the token vocabulary (a codebook pooled from sufficient statistics),
> with encoder and prior fully local and both trained to the new convergence, help / tie /
> hurt vs pure local (0.617)?**

Three outcomes, all publishable:
- **TIE** → "federating the codebook is free": you gain a shared, DP-able, exact-merge,
  onboarding-enabling vocabulary at zero accuracy cost. A systems result.
- **HELP (esp. the data-poor tail)** → the one genuine positive: pooled codebook stats
  revive dead codes for scarce clients → richer tokenization → better local prior →
  tail-rescue, WITHOUT sharing raw data or model weights.
- **HURT** → the negative holds even at the codebook level (encoder–codebook misalignment
  dominates).

---

## 2. What must change so cb_only is the federated version of the *new* local

The new local's lift came almost entirely from **stage-2 convergence**. cb_only keeps the
prior local, so its stage 2 can be made **byte-identical in protocol** to the local
baseline. Three concrete changes:

### Change A — Stage-2 prior → converged protocol  *(the dominant lever)*
`pipeline/federated.py::_local_train_prior` runs `for _ in range(n_epochs)` with no
validation, no early-stopping, no best-restore. For cb_only the prior is fully local (no
cross-client aggregation), so its training should simply BE the converged local training.

- **Cleanest:** in `train_federated`, when `local_prefixes==("",)` (prior fully local),
  bypass the round-based `federated_stage2` and train each client's prior with
  `federated_eval._build_train_stage2_converged` (the exact converged loop already written
  and validated). Zero aggregation, identical quality to `local`.
- Result: cb_only's prior = local's prior, on the shared-codebook tokenizer. The prior is
  then held at new-local quality and the ONLY moving part is the codebook — a clean
  isolation.

### Change B — Stage-1 encoder+codebook → converged, suff-stat merge kept
`federated_stage1` runs `rounds × local_epochs` of fixed local encoder SGD + a codebook
merge each round. To match the new local's stage 1 (encoder+codebook trained jointly to
val-convergence):

- **Recommended (round-based co-adaptation, faithful FL):** keep the round loop so the
  local encoders and the merged codebook keep co-adapting (this is what keeps the shared
  codebook coherent with local encoders — avoids the misalignment that would confound the
  codebook-sharing question). Give each round enough local depth, and add a **val-based
  stop across rounds** with `select_on_val=True` so the best round is restored. Target a
  total stage-1 optimizer budget matched to local's stage-1 (~10k steps).
- **Alternative (k-FED-pure, one-shot):** each client trains encoder+own codebook to
  val-convergence (= local stage1), then merge codebooks once via suff-stat. Simpler and
  matches k-FED theory exactly, but the encoder was optimized for its own codebook →
  re-assignment under the merged codebook may misalign. Under td=64 (expressive codes) this
  may be tolerable. Run as an ablation, not the primary.

### Change C — `select_on_val = True`  *(remove the now-inverted asymmetry)*
Both `federated_stage1` and `federated_stage2` default `select_on_val=False`, with a
comment: *"the baselines train fixed epochs with no validation-based restore, so a
federated arm that restores its best round would be reported at best while baselines at
last."* **That asymmetry has reversed** — the converged baselines NOW restore best-on-val.
So the federated arm must too, or it is reported at a disadvantage. Turn it on for every
converged federated arm.

> Arch (wb16/td64/win128) is already the config default → free for every arm.

---

## 3. Experiment plan

**Phase 0 — the arm itself (the gate).**
Converged `federated_cb_only` on **all wsd clusters** (real data, the one that matters) +
all toy clusters, seed 0. Matched per-client vs the `local` and `centralized` numbers
already on disk (`artifacts/converged_all/`). ~1–2 GPU-days.

**Phase 1 — the tail cut (where collaboration might still pay).**
Split each cluster's clients by train-window count. Report the paired delta
`cb_only − local` separately on the **data-poor bottom quartile** vs the rest. The toy
average hides that pooling only rescued ~5 scarce clients; test whether the *shared
codebook alone* reproduces that rescue on the tail.

**Phase 2 — decision → escalate or stop.**
Decision matrix on the tail + full paired delta:

| cb_only vs local (tail) | reading | next |
|---|---|---|
| ≥ +MDE on tail | shared codebook rescues scarce clients | escalate: add shared prior body only on tail; write positive+negative |
| within ±MDE | codebook federation is free | systems paper (DP/robust/onboard on this base) |
| < −MDE | codebook sharing hurts even converged | negative confirmed at the codebook level; stop federating |

**Phase 3 — only if Phase 2 says "help":** re-run the *next* arm up the sharing ladder
(`federated_enc` = also share encoder; then `federated` = also FedAvg prior body) under
the SAME converged protocol, to find how far up the ladder the benefit survives.

**Controls throughout:** seed {0,1,2} for CIs (single seed p=0.23 cannot carry a null);
metric = VUS-PR primary (threshold-free) + auprc/pate; report tail vs bulk separately.

---

## 4. Risks / confounds
1. **Encoder–codebook misalignment** (cb_only keeps encoder local): the shared codebook may
   not fit each local encoder's latent → this is the thing being measured, but if it
   dominates, Phase-3's `federated_enc` (shared encoder) is the fix, not a failure.
2. **Arch+protocol changed together** → cannot attribute cb_only's movement to a single
   cause vs the old 0.324; state it as "re-measured under the new regime," not "improved by X."
3. **Below-MDE ties**: with n≈31 the MDE is 0.11–0.15; a tie is "not distinguishable,"
   report as such, do not over-claim "free."
4. **cb_only prior is local → its stage-2 is literally local's** — so if cb_only ≠ local by
   more than noise, the difference is *provably* the codebook (clean attribution). This is
   the design's main strength.

---

## 5. Immediate next step
Implement Change A + C (Change B optional-recommended), then launch Phase 0 converged
`federated_cb_only` on wsd + toy, seed 0. GPUs are free (sweep finished). ~1–2 GPU-days.
