# Federating the Stage-1 encoder: FedAvg vs FedProx vs FedProto

Three ways to federate the **whole** stage-1 encoder, as three arms that differ in *nothing
else*. Everything around them is held fixed at whatever the flags say: the codebook regime
(`--fed-enc-cb`, federated by default), the local decoder, the stage-2 prior, the LR schedule,
the rounds, the seeds, the budget.

| arm | space | server does | client does | uplink/round | deployed |
|---|---|---|---|---|---|
| `federated_enc_commoninit` | — | nothing | nothing | 0 | K encoders |
| `federated_enc_fedavg` | weight | `w̄ = Σ n_k/n · w_k` | — | 48,608 floats | 1 encoder |
| `federated_enc_fedprox` | weight | same as FedAvg | `+ (μ/2)‖w − w^t‖²` | 48,608 floats | 1 encoder |
| `federated_enc_fedproto` | function | aggregate per-code prototypes | `+ λ·mean_k‖μ_k − p̄_k‖²` | **0 marginal** | K encoders |

The 2×2 with `--fed-enc-cb` is therefore: fedavg×{suffstat, local}, fedprox×{suffstat, local},
fedproto×suffstat **only** — the fourth cell is refused for a semantic reason, not skipped.

`commoninit` is not a federation — it is the **null**: same random init, then every client
trains alone. Without that row, "FedAvg helps" cannot be told apart from "a shared starting
point helps", and the trio is uninterpretable. Always run it.

FedProto's uplink is 0 *marginal* because its message — the per-code counts and centroid
sums `(n_j, m_j)` — is byte-for-byte the message the codebook merge already receives. The
prototypes are a re-reading of data the server holds, not a new channel.

## Running them

```bash
PY=/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10

# the four rows, one cluster, converged protocol
CUDA_VISIBLE_DEVICES=1 $PY -u pipeline/federated_eval.py \
  --dataset wsd_fed --cluster c0 --protocol converged --fed-enc-prior local \
  --arms federated_enc_commoninit,federated_enc_fedavg,federated_enc_fedprox,federated_enc_fedproto \
  --fedprox-mu 10 --fedprox-form decoupled --fedproto-weight 0.1 \
  --s1-rounds 30 --local-epochs 10 --batch 128 --seeds 0,1,2 --out-json out.json

# the full swept protocol (phase 0 = toy HP selection, phase 1 = wsd confirmatory)
DRYRUN=1 bash scripts/run_enc_algo_sweep.sh          # see the plan
setsid nohup bash scripts/run_enc_algo_sweep.sh > logs/encalgo_boot.log 2>&1 < /dev/null &

# unit tests (CPU, seconds) — the maths, not the training
$PY scripts/fed_enc_algo_unittest.py
```

Knobs: `--fed-enc-scope {full,partial,neck}` (which tensors), `--fed-enc-cb {suffstat,local}`
(is the DICTIONARY federated at all), `--fed-enc-sched {none,cosine}` (recipe-match `local`),
`--fed-enc-bn {buffers_local,fedbn,shared}`,
`--fed-enc-prior {local,partial,shared}`, `--fedprox-mu`, `--fedprox-form {loss,decoupled}`,
`--fedproto-weight`, `--fedproto-agg {uniform,count}`, `--fedproto-code-weight {uniform,count}`,
`--fedproto-fedavg`, `--fedproto-no-seed`, `--resume-from` (continue a finished run).

Defaults: whole encoder, BN *statistics* local but BN affine shared (NOT FedBN — see below),
**fully local stage-2 prior** — so
stage 1 is the only federated surface and the contrast against `local` / `federated_cb_only`
attributes cleanly to the encoder-federation algorithm.

## What is actually shared

`_encoder_shared_keys(model, "full")` = **52 tensors / 48,608 params = 44.9%** of the
108,334-param stage-1 model. The decoder and the `RefinementHead` stay local — worth
remembering when reading a μ→∞ result: it does **not** collapse to "no local training", it
collapses to "frozen encoder, freely-training decoder" (`federated_enc_frozen` would be the
proper upper bound; not implemented).

`--fed-enc-bn` has three settings, and the default is **not** FedBN — a distinction worth
getting right before anything cites it:

| mode | gamma / beta | running stats | what it is |
|---|---|---|---|
| `buffers_local` (default) | **averaged** | local | "BN buffers local". NOT FedBN |
| `fedbn` | local | local | FedBN (Li et al., ICLR 2021, arXiv:2102.07623) |
| `shared` | averaged | **pooled** | federate everything BN has |

FedBN's claim is that the *entire* BN layer must stay client-local under feature shift; the
default here shares 14 of this encoder's tensors (gamma/beta of 7 BN modules) that FedBN would
keep local, so calling the default FedBN would be a miscitation. `shared` pools the statistics
by the **law of total variance**, `var̄ = Σ w_j(var_j + mean_j²) − mean̄²`, not by a linear mean
of `running_var` — a linear mean understates the spread badly, which `scripts/fa_bnstats.py`
already measured in this repo. `num_batches_tracked` is never shared in any mode (int64
counter; a float average truncates on copy-back).

## The codebook: federated (default) or local

`--fed-enc-cb` decides whether the VQ **dictionary** crosses the network at all.

| | `suffstat` (default) | `local` |
|---|---|---|
| codebook | frozen locally, Prop.1 merge, broadcast every round | never aggregated, never broadcast |
| init | one common random broadcast | per-client k-means on the client's own first batch |
| update | server-side `M/smoothed(N)` | local EMA + dead-code expiry |
| what crosses the network | encoder **and** dictionary | **encoder only** |
| identical to the `local` baseline's quantizer? | no | **yes** (`collect_stats_only=False`, same k-means seeding) |

`local` is the variant that answers "what does federating *only the encoder* buy?", and it
keeps the quantizer the `local` baseline actually got — which matters here because `local` is
the arm to beat. First smoke signal, same config, 2 rounds, 2 clients: stage-1 loss
**1.36 -> 0.51** with a local codebook against **3.29 -> 2.03** with the federated one. Treat
that as a hint about where the cost sits, not a result: it is stage-1 reconstruction loss on a
toy pair, not VUS-PR, and lower recon loss is not automatically better detection (a tokenizer
that reconstructs anomalies faithfully is a *worse* detector).

⚠️ `federated_enc_fedproto` **refuses** `--fed-enc-cb local`, and that is a semantic obstacle
rather than a missing feature. FedProto aggregates one prototype per class; here the class is
the codebook index. That aggregation only means something while index *k* denotes the same
thing on every client — exactly what the shared broadcast codebook guarantees and what
per-client dictionaries destroy. Averaging client A's code-7 centroid with client B's code-7
centroid would average two unrelated regions of latent space: the usual FL permutation problem,
here at the dictionary level. A FedProto-with-local-codebook variant would need a *different*
class taxonomy (e.g. prototypes on the shared public probe indexed by latent grid cell) — that
would be **our invention, not FedProto**, so it is not implemented.

⚠️ `--fed-enc-cb local` **requires** `--fed-enc-prior local` (enforced, not advised): the
prior's `token_embedding` is an `nn.Embedding` indexed **by code id**, so FedAvg-ing a shared
prior body over clients with per-client dictionaries averages embeddings of unrelated symbols
— the same error the fedproto guard refuses, one level up.

**If you want a function-space arm with a local codebook, it already exists and is not
FedProto.** `--fed-align probe_mse` (arm `federated_align`) pulls each client's encoding of a
*shared public probe* toward the broadcast consensus: permutation-free, index-free, and
compatible with per-client dictionaries by construction. Combine it with `merge='local'` and
label it what it is — probe alignment, in the FedMD / FedDF public-data family — rather than
"FedProto with local codebooks". The citable route to a real codeword taxonomy without a shared
dictionary is server-side re-clustering of the union of client codewords (k-FED, Dennis et al.
ICML 2021), which this repo does not implement.

## Matching `local`'s recipe: `--fed-enc-sched`

The federated round loop has always trained stage 1 at a **constant** learning rate, while
`local` and `centralized` get warmup + cosine with val-driven early stopping and best-on-val
restore (`_converged_loop`: <=10k steps, warmup 10%, patience 2k). So a federated arm judged
against `local` carries a recipe handicap on top of federation:

| | `local` | trio, `sched=none` | trio, `sched=cosine` |
|---|---|---|---|
| LR | warmup + cosine | constant | warmup + cosine |
| stop | early stopping on val | fixed rounds x epochs | fixed rounds x epochs |
| selection | best-on-val, **step** granularity | best-on-**round** | best-on-**round** |
| optimizer | fresh AdamW | persistent across rounds | persistent across rounds |

`--fed-enc-sched cosine` closes the LR row, over the *full* budget rather than per round (a
per-round cosine would be a cyclic schedule — a different thing). The other rows stay open; see
"Known gaps".

**Which to use depends on the comparison, and the two cannot share a table.** Against `local`,
use `cosine`, so the gap measures federation and not the schedule. Against
`federated_cb_only` and the federated arms already on disk, use `none`. State which one a table
used.

## FedProto here is codeword-prototype, and why it is not the anchor bug

FedProto's classes are supervised labels. Stage 1 has none, so the class of a latent vector
is **the codebook index it was assigned** under the frozen broadcast codebook, and the
representation is the pre-quantization encoder output. That mapping is exact — the codebook
lookup *is* the classifier — but it walks straight toward a trap the repo has already fallen
into once. `model/vector_quantizer.py` documents `anchor_weight` (arm `federated_anchor`) as
degenerate: its target is the commitment target, so it merely rescales `commitment_weight`.
The prototype term escapes that on two independent axes.

**Form.** Commitment is per-sample. Expanding it around the per-code batch mean `μ_k`:

```
mean_i ‖z_i − e_k(i)‖²  =  mean_i ‖z_i − μ_k(i)‖²  +  Σ_k (n_k/n)‖μ_k − e_k‖²
                              └── WITHIN ──┘          └──── BETWEEN ────┘
```

The prototype term is the BETWEEN part alone. It moves the class centroid without shrinking
within-class spread; its gradient w.r.t. `z_i` is proportional to `(μ_k − p̄_k)` — the SAME
vector for every sample of class k, not each sample's own residual. No scalar multiple of the commitment loss does that
(`scripts/fed_enc_algo_unittest.py` asserts it: inflating within-class spread at fixed class
means leaves the prototype term unchanged to 1e-6 while commitment grows ~60×).

**Target.** FedProto's paper aggregation (Eq. 6) is count-weighted. Here that is
`p̄_k = Σ_j m_j^k / Σ_j n_j^k` — **exactly the Prop.1 merged codebook** (measured relative
error 2e-8). `--fedproto-agg count` is therefore *both* the paper-faithful aggregation *and*
the degeneracy control: in this setting those are the same row. See "Fidelity to the original
papers" below for the full statement — the canonical configuration (count aggregation + the
reference implementation's per-sample form) is provably the commitment loss itself.
The default `uniform` (mean over the clients that used code k, one vote each) differs from
the codebook whenever clients contribute unequal counts — i.e. under exactly the
heterogeneity this study is about — and it equalises client influence, which is the reason
prototype methods aggregate per class rather than per sample.

Because the two are so close in kind, every run logs `proto_agg_gap = ‖p̄_uniform −
p̄_count‖/‖p̄_count‖`. **Pre-registered kill rule: below 5% the arm is reported as a
reweighted commitment term, not as a distinct mechanism.** (On the toy smoke runs above: 12–46%, so at that scale it is genuinely
distinct — see the provenance note below before treating that as a result.)

**Read the gap as a heterogeneity measure, not only as a degeneracy alarm.** `p̄_uniform`
and `p̄_count` differ only because clients contribute unequal counts *and* unequal centroids,
so the gap shrinks precisely as the clients' code usage converges. The smoke runs show both
faces: pure FedProto (clients never re-synced) held 8.5% → 37.5% → 13.3% over three rounds,
while the **hybrid** (`--fedproto-fedavg`, weights averaged every round) collapsed to
8.5% → 10.7% → **2.7%** — under the kill threshold by round 2. That is not a bug in the
diagnostic; it is the hybrid being partly self-defeating: weight averaging removes the
cross-client spread the prototype target is made of. So (i) judge the rule at the round the
arm is actually reported at, (ii) a *falling* gap in an arm that federates well is evidence
of alignment, and only a gap that starts low is evidence of degeneracy, and (iii) expect the
hybrid ablation to look most like a commitment reweighting exactly where it works best.

Two more deliberate choices:

* **Stage 0 only.** A Residual-VQ quantizes `residual = residual − q` with a straight-through
  `q`, so `d(residual)/d(latent) = 0`: tokens from stage ≥1 carry **no gradient** to the
  encoder. A prototype term built on them would be a silent no-op. `_proto_term` asserts it.
* **Round-0 seeding, and its one honest caveat.** Prototypes only exist once a round of
  statistics has been collected, so without seeding the encoder would be unfederated for the
  whole of round 0 and the arm would federate in `rounds−1` of the rounds its siblings get.
  Round 0 is therefore seeded from the broadcast codebook. **Note what that codebook is at
  round 0: the random init, not an aggregate of anything** — so for exactly one round the
  prototype target coincides with the commitment target and the term degenerates to the
  `anchor_weight` case (a commitment-weight bump). From round 1 on the target is a real
  aggregate. One round in 30 is negligible either way, but it is a degeneracy, not a
  definition, and `proto_agg_gap` at round 0 will read low because of it.
  `--fedproto-no-seed` trades that round for an unfederated one.

`--fedproto-code-weight` decouples the **loss weighting** from the **target aggregation**.
Uniform code weighting gives a code seen once in a batch up to ~N/n_sel times the pull
commitment would give it, and rare codes are the anomaly signal (the detector scores prior
NLL over token streams). `count` restores token-frequency weighting without reintroducing
the target degeneracy. Sweep λ; never report a headline at an unswept λ=1.

## FedProx here goes through Adam, and that is measurable

The FedProx objective is `F_k(w) + (μ/2)‖w − w^t‖²`, and `--fedprox-form loss` implements
exactly that. But this repo keeps **one AdamW alive per client across rounds**, so the
proximal gradient `μ(w − w^t)` is divided by Adam's per-coordinate `√v̂`: μ only *rotates* an
update whose magnitude is ≈ lr. Worse, at the start of a round `w = w^t`, so the pull is
exactly zero precisely when drift begins, while the surviving β₁=0.9 momentum pushes the
client straight back off the aggregate for ~10 steps — and the smallest wsd clients run only
12 steps per round.

This is not a hypothesis. Every round logs
`prox_grad_ratio = ‖μ(w − w^t)‖ / ‖∇_w L_task‖`, and at **μ=0.1** it came out
**1.7e-4 … 5.5e-3** — two to three orders of magnitude below the data gradient. At that
setting FedProx is a no-op, and "FedProx ≡ FedAvg" would be a statement about the *solver*,
not about federation.

> **Provenance of every number quoted in this file.** They come from *smoke* runs, not from a
> study: `toy_fed_uni / M1_rotary`, 2–3 clients, 2–3 rounds, 1 seed, run while writing the
> code, into a scratch directory that no longer exists. They establish that the diagnostics
> *work* and that the two failure modes are *real at that scale* — nothing about wsd, and
> nothing that survives as evidence. Reproduce with:
> ```bash
> $PY -u pipeline/federated_eval.py --dataset toy_fed_uni --cluster M1_rotary \
>   --clients uni_00,uni_01,uni_02 --s1-rounds 3 --s2-rounds 2 --local-epochs 1 \
>   --batch 64 --seeds 0 --fedprox-mu 0.1 --fedproto-weight 0.1 --out-dir /tmp/smoke \
>   --arms federated_enc_commoninit,federated_enc_fedavg,federated_enc_fedprox,federated_enc_fedproto
> ```
> Anything that is going to be cited belongs in `artifacts/encalgo_sweep/`, produced by
> `scripts/run_enc_algo_sweep.sh`, with the seeds and the protocol below.

**Pre-registered kill rule, stated so it can actually fail: the μ carried into the
confirmatory run must show `prox_grad_ratio ≥ 1e-2` (loss form) or `prox_pull_frac ≥ 5%`
(decoupled form) in its own phase-0 rounds. A reported arm below that is a solver-level
no-op and must be labelled one — "FedProx ≡ FedAvg" is not available as a conclusion.**
The rule binds the *reported* μ, not the sweep: `prox_grad_ratio` is roughly linear in μ, so
"the ratio cleared 1e-2 somewhere in the sweep" proves nothing about the point you publish.

### RESOLVED 2026-07-27 — the kill rule was checked against the confirmatory run and PASSED

The smoke verdict above ("at μ=0.1 FedProx is a no-op") **does not reproduce at convergence
and must not be cited.** Measured on `artifacts/converge60/ckpt/wsd_fed/*/seed0/
federated_enc_fedprox_mu0.1/fed_history.json`, same μ=0.1, all four clusters, rounds ≥1:

| cluster | n rounds | min | median | max |
|---|---|---|---|---|
| c0 | 14 | 0.0707 | 0.0837 | 0.1712 |
| c1 | 42 | 0.0441 | 0.0520 | 0.0679 |
| c2 | 38 | 0.0446 | 0.0521 | 0.0719 |
| c3 | 38 | 0.0412 | 0.0554 | 0.0854 |
| **all** | 132 | **0.0412** | **0.0550** | 0.1712 |

The worst single round is `0.0412`, **4× above the `1e-2` threshold**. FedProx at μ=0.1 is
therefore *not* a solver-level no-op in the reported runs, and the arm is legitimate.

**Why the smoke said the opposite — it measured inside the initial loss transient.** The
ratio's denominator is the task gradient, which collapses by ~1000× within one round. On c0:

| round | `mean_loss` | `prox_grad_ratio` |
|---|---|---|
| 0 | 55.71 | 0.0099 |
| 1 | 0.577 | 0.1712 |
| 7 | 0.077 | 0.0886 |
| 14 | 0.058 | 0.0788 |

Round 0 alone sits at ~1e-2, i.e. the smoke's range; from round 1 the ratio is 8--17×
higher. `corr(log mean_loss, log prox_grad_ratio) = -0.683` across c0's rounds: the prox
anchor becomes *relatively* significant precisely as the model converges. A 3-round smoke
therefore cannot measure this quantity at all — the diagnostic is only meaningful once the
task gradient has settled. Generalise the lesson: **any diagnostic defined as a ratio
against the task gradient is untrustworthy on a short run.**

Consequence for reporting: `federated_enc_fedprox` (VUS-PR 0.524) vs `federated_enc_fedavg`
(0.444) is a real, mechanistically-supported separation, not a solver artifact — the prox
anchor also lets training keep improving for far longer (c1: 43 rounds vs 28; c2: 39 vs 23).
**But the 0.08 outcome gap is still below the n=31 MDE**, so the *mechanism* is measured
while the *outcome* difference is underpowered. Report them as two separate claims.

Two ways to get a μ that bites, both provided:

* sweep μ over **decades** (`0.001 … 10`), not around 0.01;
* `--fedprox-form decoupled`: apply the contraction after the optimizer step,
  `w ← w + lr·μ·(w^t − w)`, guarded by the GradScaler's "did this step apply?" test. It
  bypasses the preconditioner. **The knob that means "fraction of drift removed per step"
  is `lr·μ`, not μ** — at `lr=1e-3` even μ=10 removes 1% of the drift per step, which is why
  `prox_pull_frac = 1 − (1 − lr·μ)^steps` (the fraction removed over a whole round) is logged
  separately and is the number to read for this form. This is the AdamW-consistent analogue
  of the paper's objective, exactly as AdamW's decoupled weight decay is to L2.

The two diagnostics are **not comparable to each other** — one is a gradient ratio, the other
a displacement fraction. Never gate both with one threshold.

Picking μ for the decoupled form needs no run: the round removes
`1 − (1 − lr·μ)^steps` of the drift, so at `lr=1e-3` and τ steps per round

| μ | τ=15 | τ=66 (largest wsd client) | τ=12 (smallest) |
|---|---|---|---|
| 1 | 1.5% | 6.4% | 1.2% |
| 10 | 14% | 48% | 11% |
| 100 | 79% | 99.9% | 72% |

Two consequences worth stating before the sweep runs. μ=1 is below the 5% gate at every
realistic τ — the sweep's useful range for this form starts at μ≈10. And the pull is
**client-size dependent** (τ scales with the client's data), so the anchor is systematically
weaker on exactly the small silos the tail hypothesis is about: another face of the
epoch-vs-step budget gap in "Known gaps".

Either way, **μ here is not comparable to a μ in an SGD FedProx table.**

## Telemetry, and the proof that each arm did what it says

Written per round to `<out-dir>/<cluster>/seed<N>/<arm-tag>/fed_history.json`, and the arm
tag carries the knobs (`federated_enc_fedprox_mu0.1`), so a μ/λ sweep no longer overwrites
its own checkpoints:

| field | reads |
|---|---|
| `mean_loss` | **task loss only** — comparable to every arm already on disk |
| `mean_obj` | task + penalties (the objective actually minimised) |
| `enc_drift`, `enc_drift_rel` | `mean_j‖w_j − w̄‖` before aggregation, at FedAvg's **weighted** centroid |
| `enc_step` | `mean_j‖w_j^{r,pre} − w_j^{r−1,post}‖` — local movement only, hence the cross-arm-comparable channel¹ |
| `prox_grad_ratio` | 'loss' form: prox gradient vs data gradient (kill rule) |
| `prox_pull_frac` | 'decoupled' form: fraction of the round's drift the anchor removed |
| `proto_agg_gap` | is the prototype target distinct from the codebook (see the kill rule) |
| `proto_term`, `proto_coverage` | penalty size, fraction of codes carrying a prototype |
| `steps_per_client`, `skipped_steps` | budget match; fp16 steps the GradScaler threw away |

¹ `enc_drift` is *per-round* for fedavg/fedprox (clients are re-synced every round) but
*cumulative since init* for fedproto (they never are). Comparing raw drift across those two
compares two different quantities. `enc_step` is measured from the state each client
actually **started** the round in (i.e. after the previous round's aggregation), so it is
local movement for every arm alike — measuring it from the previous round's *pre*-aggregation
state would silently charge the server's own correction to the arms that aggregate.

`knobs.json` records `enc_max_cross_client_delta`, the machine-checkable proof of what
happened: **exactly 0.0** for any arm that weight-averages, **> 0** for one that does not.
A `fedproto` row reporting 0.0 would be a mislabelled FedAvg. Measured on the smoke run:
fedavg 0.0, fedprox 0.0, fedproto 0.057, commoninit 0.052.

## Fidelity to the original papers — what is canonical and what is not

Audited against the primary sources (ar5iv full text; for FedProto also the authors' own
implementation). Read this before writing any of it up: two of the three arms deviate, and one
of the deviations is *forced by a degeneracy* that has to be reported as a result.

### FedAvg (McMahan et al., AISTATS 2017, arXiv:1602.05629)

| item | verdict |
|---|---|
| `w_{t+1} = Σ_k (n_k/n)·w^k` with n_k = local sample count | **faithful** (exact, fp32, `_fedavg_encoder`) |
| shared initialization before round 0 | **faithful** (`_broadcast_encoder`, all algorithms) |
| E local epochs, B minibatch, one aggregation/round | **faithful** |
| C = 1 (full participation) | **scope choice** — inside the paper's family, but the paper's headline knob is never swept. Say "cross-silo, C=1", do not imply otherwise |
| local solver | **deviation** — the paper's `ClientUpdate` is *stateless SGD*; here one AdamW per client is kept alive, so momentum/second moments survive aggregation. Label it "ClientOpt = AdamW, state persisted" (cf. Reddi et al., ICLR 2021) |
| what is averaged | **deviation** — only `encoder.*` = 44.9% of stage 1 (≈5% of the two-stage system). The implied objective is `min Σ p_k F_k(w_E, θ_k)`, not the paper's single-w objective. This is the **FedPer** (Arivazhagan et al. 2019) / FedRep (Collins et al., ICML 2021) shared-representation family — cite it as such |
| BN affine averaged, statistics local | **deviation from the FedBN label**, see the BN table above |
| the codebook | **scope choice** — merged by the repo's k-FED/EM sufficient-statistic step, not by FedAvg, inside an arm named `..._fedavg`. Under `--fed-enc-cb local` this disappears and the arm becomes a clean FedPer split |

*Honest one-liner:* **canonical FedAvg on the aggregation rule, but applied to a
shared-representation split with a stateful Adam client — so "FedAvg" names the aggregator, not
the whole setup.**

### FedProx (Li et al., MLSys 2020, arXiv:1812.06127)

| item | verdict |
|---|---|
| `h_k(w; w^t) = F_k(w) + (μ/2)‖w − w^t‖²` (`--fedprox-form loss`) | **faithful**, including the ½ and the anchor being the round's broadcast |
| local solver free choice | **faithful** — the paper is explicitly solver-agnostic ("local solver of choice") |
| server aggregation | **scope choice** — Algorithm 2 writes `1/K` uniform; we use `n_k` weights, which is the paper's *own experimental* estimator with sampling variance removed |
| γ-inexactness (Definition 2) | **deviation** — never defined, enforced or measured. `prox_grad_ratio` measures penalty *size*, not solve inexactness. Theorem 4 therefore does not apply |
| prox restricted to the encoder | **deviation** — decoder, RefinementHead and the whole prior re-adapt unconstrained, so `μ̄ = μ − L₋ > 0` covers a subspace only |
| variable local work / stragglers | **not exercised** — uniform `local_epochs`, full participation. This tests only benefit (1) of the paper; benefit (2), its actual novelty over "FedAvg + penalty", is untested |
| AdamW `weight_decay=0.01` | **deviation** — a *second*, unmodelled proximal pull toward 0. At the old default μ=0.01 the two were numerically identical (`lr·μ = lr·wd = 1e-5`), i.e. the "anchor" was as strong as ordinary weight decay |
| `--fedprox-form decoupled` | **our variant, labelled** — not in the paper. It is the first-order proximal-*operator* split of the same penalty (`prox_{ηg}(v) = v + (ημ/(1+ημ))(w^t − v)`), exact to O((lr·μ)²) |

*Honest one-liner:* **canonical FedProx as an objective, but the theory does not transfer
(subspace anchor, no γ-inexactness, stateful Adam), and μ is not comparable to a published μ.**

### FedProto (Tan et al., AAAI 2022, arXiv:2105.00243) — the important one

The paper and the authors' code disagree on the *form* of the regularizer, so both were read:

* **Eq. 3** local prototype `C_i^(j) = (1/|D_{i,j}|) Σ f_i(φ_i;x)` — a per-class mean. **We match.**
* **Eq. 6** server aggregation `C̄^(j) = (1/|N_j|) Σ_{i∈N_j} (|D_{i,j}|/N_j) C_i^(j)` — **count-weighted**.
* **Eq. 8** `L_R = Σ_j d(C_i^(j), C̄_i^(j))` — reads as **per-class mean**.
* **Appendix + the authors' code** (`yuetan031/FedProto`, `update_weights_het`):
  `proto_new[i] = global_protos[label_i]; loss2 = MSELoss()(proto_new, protos)` —
  **per sample**, mean reduction.

**Now the algebra that decides everything** (asserted in `scripts/fed_enc_algo_unittest.py`,
test 8, measured relative error **6.7e-8**):

```
count-weighted prototype   p̄_k = Σ_j m_j^k / Σ_j n_j^k = e_k     (the Prop.1 merged codebook)
canonical per-sample term  mean_i ‖z_i − p̄_{k(i)}‖²  =  mean_i ‖z_i − e_{k(i)}‖²
VQ commitment loss         mean_i ‖z_i − e_{k(i)}‖²              (vector_quantizer.py)
                           ^^^^^^^^^ the same expression ^^^^^^^^^
```

**Canonical FedProto — per-sample regularizer + count-weighted aggregation — transplanted
literally into this codebase IS the VQ commitment loss.** Adding it with weight λ is exactly
equivalent to setting `commitment_weight ← commitment_weight + λ`. That is precisely the
degeneracy `model/vector_quantizer.py` already documents for `anchor_weight`, and it is not an
artefact of our coding: it follows from the class being the codebook index and the server's
M-step being a count-weighted mean, which are both properties of *this setting*, not choices.

So the two deviations are **forced, not stylistic**:

| item | our choice | paper | verdict |
|---|---|---|---|
| regularizer form | per-class batch **mean** (between-class only) | Eq. 8 says mean; the code says per-sample | **deviation from the reference implementation**, matches the main text's Eq. 8 |
| aggregation | **uniform** over the clients using code k (default) | count-weighted (Eq. 6) | **deviation** — the paper-faithful setting is the degenerate one |
| no weight averaging | none | none | **faithful** (Eq. 7 is prototype-only) |
| heterogeneous client models allowed | identical models here | allowed, not required | **scope choice** |
| classes = codebook indices | — | supervised labels | **adaptation**, and the source of the degeneracy |

*Honest one-liner:* **this is FedProto's mechanism (prototype-only communication, no weight
averaging) with two deliberate departures — between-class form and uniform aggregation —
because the canonical configuration provably reduces to the model's existing commitment loss.
Report `--fedproto-agg count --fedproto-code-weight count` as the canonical null, not as an
ablation.** Name the arm "codeword-prototype (FedProto-style)", not "FedProto".

### What to run because of this

`--fedproto-agg count --fedproto-code-weight count` is now the **canonical-FedProto null row**:
if the arm's headline is indistinguishable from it, the arm is a commitment-weight sweep. Pair
it with `federated_cb_only` at `commitment_weight ∈ {1, 1+λ*}`, which is the same null reached
from the other side.

## Continuing a run instead of restarting it (`--resume-from`)

**The federated arms stop while they are still improving.** All ten
`logs/cbonly_converged/*.log` end at round 29 with the stage-1 validation loss still falling
and `restored=0` — `select_on_val` never fires because the last round is always the best. That
is true of `federated_cb_only` itself, not only of the trio, so `--protocol converged` labels
the baselines and the per-client prior, **not** the federated stage-1 encoder.

```bash
# +30 rounds on a finished run. --s1-rounds means ADDITIONAL rounds.
$PY -u pipeline/federated_eval.py --dataset wsd_fed --cluster c0 \
  --arms federated_cb_only --protocol converged --s1-rounds 30 --local-epochs 10 \
  --seeds 0 --batch 128 \
  --resume-from artifacts/fed_eval/wsd_fed/c0/seed0/federated_cb_only \
  --out-dir artifacts/resume_probe/wsd_c0
```

`--resume-from` points at the **arm directory** of the run to continue (the one holding one
sub-directory per client). Two tiers, and the run says which one it took:

| tier | when | restores |
|---|---|---|
| **full** | the source run wrote `_fed_resume.pt` (every run from this version on) | model + VQ + **AdamW moments** + GradScaler + round counter + server codebook EMA |
| **weights only** | older runs | model + full VQ state; **the optimizer restarts** |

**Validated, not asserted.** Three runs on `toy_fed_uni/M1_rotary`: **A** = 2 rounds,
**B** = A resumed for 2 more, **C** = 4 rounds continuously. B reproduces C's rounds 2–3 —
identical dead-fraction and revival counts, perplexity 25.8/33.1 vs 25.9/33.6, loss within
0.36% / 0.15%. The reference for "within" is this setup's own nondeterminism: A and C share
their first two rounds by construction, yet differ by 0.04% / 0.69% (fp16 + `cudnn.benchmark`
autotuning). **The resume gap is inside the band two identical from-scratch runs already
differ by.** The per-round RNG seed uses the ABSOLUTE round index for this reason — a resumed
run must continue the batch-order stream, not replay round 0's; `rounds_done` is 0 when not
resuming, so nothing changes for any existing arm.

The weights-only tier is a real perturbation, not a formality: this repo keeps one AdamW alive
per client, so dropping its moments costs each client ~1/(1−β₁) ≈ 10 steps of unpreconditioned
updates. Such a run is a *warm-started continuation*, not "the same run trained longer" — the
code prints that warning itself. Guards: one (arm, seed) at a time (optimizer state is
per-client and positional), the `merge` regime must match, and the resumed clients must agree
on the codebook. Round indices continue (`round 30, 31, …`) so histories concatenate.

### Stopping where it actually converges (`--fed-patience-rounds`)

A round costs 8–18 h on wsd, so "just double the budget" is not a plan. `--fed-patience-rounds
N` stops the stage-1 loop once the cohort validation loss has not improved for N consecutive
rounds and logs `CONVERGED`, mirroring the baselines' own criterion at round granularity. That
makes a generous `--s1-rounds` safe: ask for 60, stop at the knee. Needs `--protocol converged`
(which is what turns per-round validation on); 0 = off, so no arm on disk changes.

Measured tail slope of the truncated runs, for sizing: −1 % to −5 % per round at round 29 on
both `wsd_fed/c0` and `toy_fed_uni/M1_rotary`. Nothing is flattening yet.

### …but check whether more training changes the METRIC

The tempting move — re-run everything longer — is 60+ GPU-jobs, and **this repo has already
run the equivalent experiment and got a null**: retraining `local` for 10k steps produced a
*dissociation*, wsd detection getting worse while toy got better
(`local-recon-autopsy-not-epochs`; conclusion on record: *"few epochs" refuted, wsd smoothing
is a DATA limit not undertraining*).

There is a mechanism, not just a coincidence. The anomaly score contains a reconstruction term,
`s_local = (x − recon)²` (`pipeline/detect.py`), so **a stage 1 that reconstructs anomalies more
faithfully is a worse detector**. A falling stage-1 validation loss is therefore not evidence
of better VUS-PR and can be evidence against it.

`scripts/resume_probe.sh` settles it on the CURRENT setting rather than by citing the older
experiment: it continues `federated_cb_only` on **six** clusters spanning cohort size
(wsd c0/c3/c2 = 5/6/9 clients, toy M1_rotary/M5_bearing/M4_cardiac = 6/8/11) for up to +60
rounds each, stopping at the knee via patience. `scripts/resume_probe_report.py` then prints,
per cluster, the val curve with its per-round slope and whether it stopped on patience, and
the **paired per-entity VUS-PR** before vs after.

* metric unchanged → the truncation does not matter for what is reported; run the sweep as is
  (the pooled |Δ| to beat is ~0.05 VUS-PR; the honest MDE at n=31 is 0.11–0.15)
* metric improves → raise the budget **for every arm in the table**, and redo the existing rows
* metric degrades → the dissociation is confirmed on the federated arm too, which is itself a
  ledger-worthy result

The probe **snapshots** its source checkpoints first: `artifacts/fed_eval/<ds>/<cl>/seed0/<arm>`
is the default scratch path, so a concurrently running sweep containing a `federated_cb_only`
job for that cluster (`run_cbonly_ema_sweep.sh` has one) would otherwise rewrite the very files
being resumed.

### What resume does *not* answer

Resume adds more rounds of **the same recipe**, and that recipe is constant-LR. `local` earns
its 0.617 in the tail of a warmup+cosine anneal. So "more rounds" and "recipe-matched to
`local`" are different interventions, and the second needs a fresh run with
`--fed-enc-sched cosine` over the full budget — a resumed cosine would anneal only over the
appended rounds, which is a restart-anneal, not the baseline's schedule.

## Reading a result honestly

Three things constrain what this comparison can conclude, and they are properties of the
benchmark, not of the code:

1. **There is no headroom.** On converged wsd, `local` 0.617 beats `centralized` 0.585 — the
   pooled skyline sits *below* the floor. No federated arm can exceed a skyline that is
   already losing. The primary endpoint should be **mechanism** (cross-client token
   agreement, usage-JS, drift), with detection reported as a **TOST equivalence** test, not
   as superiority.
2. **n = 31 entities on wsd.** The honest MDE is 0.11–0.15 VUS-PR (the ledger's 0.087 is
   refuted — see `audit-2026-07-15-publishability`). Three seeds cap a two-sided Wilcoxon at
   p = 0.25 regardless of effect size; test at the **entity** level (seed-average first),
   Holm-correct the contrast family.
3. **Select hyperparameters on toy, confirm on wsd.** `run_enc_algo_sweep.sh` enforces the
   split (`PHASE=0` toy, `PHASE=1` wsd). Choose μ\* and λ\* on stage-1 val loss and token
   agreement — **not** on test VUS-PR, and not on wsd. Sweeping on wsd and reporting the best
   point is the cherry-pick the ledger already caught once.

The most likely honest outcome, given the matched-fusion evidence already in the ledger
(weight-space fusion collapses to 0.048, function-space recovers to 0.326), is: the three
arms are *equivalent on detection*, and FedProto is the interesting row because it reaches
that equivalence **at zero marginal uplink, with no weight-space fusion, and with a
personalized encoder per client**. That is a publishable claim about cost and mechanism —
provided `proto_agg_gap` clears 5% and the equivalence is tested, not assumed.

## Known gaps (deliberately not implemented)

* **Budget matching.** Rounds are epoch-based, and wsd clients span 12–66 batches/epoch, so
  the number of optimizer steps is size-correlated — on exactly the tail axis where the one
  remaining headroom hypothesis lives. A step-based round budget (`τ` steps/round, as
  `_local_train_prior_steps` already does for the prior) is the fix.
* **Post-aggregation BN recalibration.** `_fedavg_encoder` is the last write to the encoder
  weights, so fedavg/fedprox ship BN statistics belonging to *pre-average* weights while
  fedproto does not — an arm-dependent asymmetry, and this repo measured BN-recal worth ±0.11
  VUS-PR in *opposite directions* on wsd vs toy.
* **AdamW moments survive aggregation** (`exp_avg` still points along the client's own
  pre-aggregation drift). An `enc_reset_opt_state` ablation would bound it.
* **Per-client model selection.** `select_on_val` restores one cohort-wide best round, which
  is the right thing for a cohort-wide encoder and arguably not for FedProto's K encoders.
* **`federated_enc_frozen`** (μ→∞ upper bound) and the `encoder-shared / codebook-local` cell
  of the 2×2, without which "encoder federation helps" is not separable from "encoder
  federation makes the shared codebook coherent".
