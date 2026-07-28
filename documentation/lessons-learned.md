# Lessons learned — anomaly detection scoring bugs

Record of two interrelated bugs that masked an excellent model behind random
detection performance (AUROC ≈ 0.5) on SMAP A-1, plus the diagnostic process
that uncovered them. Written so the same trap is not stepped on again.

The model was trained correctly all along. The bugs were entirely in
`cfg.scoring` (post-processing), not in training. The fix moved AUROC from
0.479 → 0.9954 with three config-line changes and zero retraining.

---

## Bug 1 — `normalization="zscore"` applied per-window

### What it did

In `detect.py`, the normalization is applied with `dim=-1` to a
`(B, C, T)` tensor. That means: for every batch sample's 64-point window,
for every channel, the score series is rescaled to mean 0 and std 1
**inside that window**.

### Why it looked reasonable

z-score is a "safe default" that prevents scale mismatches between
different score components (e.g. between `s_local` and `s_prior`).
It is standard practice in many pipelines.

### Why it broke detection

Anomaly detection requires comparing **across** windows: an anomalous
window should produce higher overall scores than a normal window.
Per-window z-score destroys exactly this comparison: every window is
forced to the same internal distribution. A window full of anomalous
tokens and a window of pure normal tokens become statistically
indistinguishable after normalization. The discriminating signal — the
absolute magnitude of the scores — is erased.

### How to spot it next time

- If AUROC is near 0.5, **the first thing to try is `normalization="none"`**.
  If AUROC jumps significantly, the normalization is the bug.
- Plot the histogram of raw scores for normal vs anomalous timestamps.
  If the two histograms overlap heavily despite the model converging
  well, the normalization step is the prime suspect.
- **General rule**: in anomaly detection, never normalize per-instance
  (per-window, per-batch). If normalization is needed, compute the
  statistics globally across the whole corpus, or per-channel across the
  whole training split, and apply the same constants everywhere.

---

## Bug 2 — wrong weight ratio between `s_local` and `s_prior`

### What it did

The default in `cfg.scoring` was `weight_s_local=1.0, weight_s_prior=0.5`.
This combined the Stage 1 reconstruction error at full strength with
half-strength Stage 2 prior surprise.

### Why it looked reasonable

`s_local` is the "primary" signal: pixel-level reconstruction error from
Stage 1, the more direct measurement. `s_prior` looks like an auxiliary
signal added by Stage 2. Defaulting to a heavier weight on the primary
component is intuitive.

### Why it was wrong for this data

Single-component AUROC reveals the truth:

| Component | AUROC alone |
|---|---:|
| `s_local` only | ~0.45–0.55 (random) |
| `s_prior` only | **0.9936** |

Stage 1 reconstructs both normal and anomalous patterns well, so the
residuals carry essentially no anomaly information on this dataset. The
prior's surprise (`-log p(token | masked context)`) carries almost all
the signal because anomalies produce token sequences that violate the
prior's training distribution.

Combining noise (weight 1.0) with clean signal (weight 0.5) gives a
score dominated by noise. Net AUROC: 0.48.

### How to spot it next time

- **Always compute single-component AUROC before tuning combination
  weights.** Thirty seconds of diagnostic before any retraining.
- If component A has AUROC ≈ 0.5 alone and component B has AUROC ≈ 0.95
  alone, the correct combination is to use B alone — never weight them
  to "average them out".
- Weight ratios should follow signal-to-noise ratios, not architectural
  intuition about which component is "primary".
- In time-series anomaly detection with a token-level prior,
  `weight_s_prior >= weight_s_local` is the safer default.

---

## Meta-lesson — debug downstream before retraining

When AUROC is unexpectedly bad and the training metrics look fine
(loss converging, val ≈ train, codebook utilized), suspect the scoring
pipeline **first**, the model **second**. Retraining is expensive and
will not fix a bug that lives in post-processing.

### Symptoms that point at the scoring pipeline

- Loss curves are healthy but AUROC ≈ 0.5
- AUROC stays flat across many aggregation choices (because the input
  to all of them is already destroyed)
- F1 = 0 because the threshold is calibrated on a degenerate score
  distribution

### Symptoms that point at the model

- val_loss diverges from train_loss (overfitting)
- Reconstruction quality is visibly bad on test windows
- Codebook collapsed (`global_codebook_usage_percent` < 30%, perplexity
  near 1)

These two failure modes look superficially identical from the final
report.json — both give bad AUROC — but the diagnostic path is
completely different.

---

## Standard diagnostic checklist

Run before changing any model hyperparameter when detection is bad:

1. **Single-component AUROC**: compute AUROC of `s_local` alone and
   `s_prior` alone. If one is near 0.5 and the other near 1.0, you
   have your answer — the bad weight ratio is the bug.
2. **Try `normalization="none"`**: if AUROC jumps significantly, the
   normalization step is destroying the signal.
3. **Try several aggregations** with the right normalization and
   weights. Use [`compare_aggregations.py`](../pipeline/compare_aggregations.py) —
   it caches per-window scores once and replays every (norm, agg)
   combination in seconds. Look at AUROC across the full grid; if all
   are stuck below 0.6, the problem is upstream (model or data).
4. **Sanity-check the labels**: read a window the script flags as
   anomalous and verify visually that the underlying signal really is
   anomalous in that range. An off-by-one in label loading produces
   exactly the same near-random AUROC pattern.
5. Only after steps 1–4 yield no improvement, look at the model.

---

## Concrete defaults that work for SMAP A-1 (post-fix)

```python
cfg.scoring.normalization     = "none"
cfg.scoring.weight_s_local    = 0.0
cfg.scoring.weight_s_prior    = 1.0
cfg.scoring.aggregation       = "max"      # logsumexp / softmax also valid
cfg.threshold.q               = 0.9999
```

Result: AUROC 0.9954, AUPRC 0.776, F1 0.51 (recall 91%, precision 35%).

The `0.0` weight on `s_local` is not a typo — on this dataset the Stage 1
residual carries no anomaly signal and including it actively hurts. Other
datasets may differ; rerun the single-component AUROC check whenever the
dataset or entity changes.
