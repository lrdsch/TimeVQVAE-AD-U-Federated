"""
=============================================================================
  quality_stage2.py — MaskGIT prior generation QA, step-by-step.
=============================================================================

Three sampling methods, all visualised step-by-step (token grid + partial
decoded waveform per channel):

  Method 1 — Unconditional sampling
    Start with all-masked tokens. Iteratively predict every position, sample
    at masked positions, keep the high-confidence picks, re-mask the rest.
    By step T everything is committed. Final waveform is decoded through the
    frozen stage-1 decoder (refinement head off — it is OOD on synthesised
    tokens).
    Quality is judged distributionally — value histograms, |FFT| spectra and
    token-id usage of generated vs. training windows. RMSE-to-closest-train
    is used ONLY as a triage scalar to pick best/median/worst plots: a small
    RMSE means "looks like SOMETHING in the training set", not "the prior is
    good"; the distribution plot is the actual quality signal.

  Method 3 — Autoregressive sequential decoding
    Same start (all-mask) but commit ONE token per iteration in flat sequence
    order (channel-major, freq-middle, time-inner). The bidirectional
    transformer still produces predictions for every position at each step,
    but only position k is kept at iteration k — all other predictions are
    discarded. Equivalent to running the prior in classic AR mode while
    keeping the bidirectional architecture. Memory: too large to keep every
    state for high-channel windows, so only `ar_snaps_per_channel` evenly-
    spaced snapshots within each channel's AR sweep are stored (union over
    channels). Plotting per (sample, channel) takes only the snapshots that
    fall in that channel's range.
    Same triage scalar (RMSE-to-closest-of-split-pool) and same per-split
    folder structure as Method 1.

  Method 2 — Conditional inpainting (30% contiguous time mask)
    Take a real window, encode it to tokens, blank a random contiguous
    window of W*0.3 latent-time columns across all (channel, frequency).
    Run the same iterative decode with the context positions FROZEN — they
    feed the transformer as evidence at every step but are never sampled
    or re-masked. RMSE / MAE / cosine / Pearson are computed on the masked
    REGION ONLY (context never enters the metric — that would just measure
    "did we remember the input we were given").

For each method we generate `samples_per_split * N_split` runs (default 1×):
- Methods 1 & 3 use only N_train samples (train-pool reference).
- Method 2 uses N_train + N_val + N_test samples (one per real window per split).
Plots are produced for the top-K best / median / worst by RMSE.
Conditional plots on test windows are split by whether the masked region
intersects a labelled anomaly (`test_clean` vs `test_anomaly`).

Output structure:
    output_dir/
      unconditional/                                    # train pool only
        summary.csv                                     # per-sample metrics (all)
        distributions.png                               # quality signal — value hist / FFT / token usage
        sample_NNNN/channel_CC.png                      # flat random subset (no ranking)
      autoregressive/                                   # train pool only
        summary.csv
        distributions.png
        sample_NNNN/channel_CC.png                      # flat random subset (no ranking)
      conditional/
        summary.csv
        {train|val|test_clean|test_anomaly}/{best|median|worst}/sample_NNNN/channel_CC.png
      config.json

Each `channel_CC.png` is a filmstrip with one row per iterative-decode step:
  • LEFT cell: F × W token grid for that channel, every cell coloured by its
    state — frozen / mask / newly-sampled / kept / newly-remasked — with the
    integer token id printed inside.
  • RIGHT cell: time-domain waveform decoded from the partial token state.
    For Method 2 the original signal is overlaid in grey, the masked region
    has a yellow shade, and on test windows the intersection with a labelled
    anomaly is hatched red.

Debug-friendly: every named function lives in this file (the iterative decode
loop is reimplemented here so you can drop a breakpoint at any step). Only
[quality_stage2.py] is touched — no edits to model/prior.py or stage2.py.
"""
from __future__ import annotations

# repo root on sys.path so this pipeline/ script can import the shared
# libs (config / data / utils / metrics_core) that live at the project root.
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
from lib.proctitle import set_process_title  # noqa: E402

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import Config, load_config
from data import (
    SlidingWindowDataset,
    load_records, load_scaled_records,
)
from stage1 import Stage1VQVAE, load_stage1
from stage2 import Stage2System, load_stage2
from utils import (
    best_checkpoint, configure_logging, resolve_path, run_dir_for,
    seed_everything, resolve_device,
)


# ════════════════════════════════════════════════════════════════════════════
#   Cell-state codes + colours for the token grid heatmap
# ════════════════════════════════════════════════════════════════════════════

STATE_FROZEN         = 0   # context, never changed (cond only)
STATE_MASK           = 1   # still mask_token_id, never sampled yet (init only)
STATE_NEWLY_SAMPLED  = 2   # transitioned mask → known THIS step
STATE_KEPT           = 3   # was known going INTO this step, still known
STATE_NEWLY_REMASKED = 4   # was sampled, then re-masked THIS step

CELL_COLORS = [
    "#1f4e79",   # frozen          — deep blue
    "#d9d9d9",   # mask            — light grey
    "#2ca02c",   # newly sampled   — green
    "#aec7e8",   # kept            — pale blue
    "#d62728",   # newly remasked  — red
]
CELL_CMAP = mcolors.ListedColormap(CELL_COLORS)
CELL_NAMES = ["frozen", "mask", "newly sampled", "kept", "newly remasked"]


# ════════════════════════════════════════════════════════════════════════════
#   Iterative parallel decoding with history (handles uncond + cond inpaint)
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def iterative_decode_with_history(
    prior,
    initial_tokens: torch.Tensor,
    frozen_mask: torch.Tensor | None,
    steps: int,
    mask_scheduling_fn,
    choice_temperature: float,
) -> list[torch.Tensor]:
    """MaskGIT iterative decoding that returns the token state at every step.

    initial_tokens : (B, seq_len) long.
        Use `mask_token_id` at every position to be generated. For
        unconditional sampling all positions are mask. For conditional
        inpainting, frozen positions hold the ground-truth tokens and the
        masked region holds `mask_token_id`.
    frozen_mask    : (B, seq_len) bool or None.
        True at positions that must NEVER be sampled or re-masked. (Implicit:
        their `initial_tokens` value is non-mask, so the existing
        `where(unknown, sampled, tokens)` logic already preserves them — the
        explicit mask is just a defensive guard.)

    Returns a list of (B, seq_len) long tensors of length `steps + 1`.
    Index 0 is the initial state, index t (t ≥ 1) is the state at the END
    of step t-1.
    """
    mask_id = prior.mask_token_id
    tokens = initial_tokens.clone()
    frozen: torch.Tensor = (
        torch.zeros_like(tokens, dtype=torch.bool)
        if frozen_mask is None else frozen_mask
    )

    history = [tokens.clone()]

    # γ(r) is applied to the MASKABLE region, not the full seq_len. In
    # unconditional sampling frozen is all-False so this equals seq_len and
    # the schedule reduces to vanilla MaskGIT. In conditional inpainting it
    # equals the masked-region size (e.g. 30% of seq_len for our default),
    # so γ(r)·M starts shrinking the survivor budget from step 1 instead of
    # only after r > arccos(M/seq_len)·2/π.
    maskable_count = int((~frozen).sum(dim=1).max().item())

    for step in range(steps):
        unknown = tokens == mask_id
        logits = prior._logits(tokens)                                   # (B, L, K)
        sampled = torch.distributions.Categorical(logits=logits).sample()
        sampled = torch.where(unknown, sampled, tokens)
        # Defensive: also force frozen to original even if `unknown` were
        # somehow contaminated.
        sampled = torch.where(frozen, tokens, sampled)

        ratio = (step + 1) / steps
        remask_ratio = mask_scheduling_fn(ratio)
        remask_count = max(0, int(remask_ratio * maskable_count))

        if remask_count == 0:
            tokens = sampled
            history.append(tokens.clone())
            continue

        probs = logits.softmax(dim=-1)
        sel_probs = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        u = torch.rand_like(sel_probs).clamp(min=1e-20)
        gumbel = -torch.log(-torch.log(u))
        confidence = (
            torch.log(sel_probs.clamp(min=1e-5))
            + choice_temperature * (1.0 - ratio) * gumbel
        )
        # Only positions that were UNKNOWN at the start of this step are
        # eligible for re-masking. Frozen + previously-known positions get
        # +inf and are never picked.
        confidence = torch.where(
            unknown & ~frozen, confidence,
            torch.full_like(confidence, float("inf")),
        )
        low_conf_idx = confidence.topk(
            k=min(remask_count, tokens.shape[1]), dim=-1, largest=False,
        ).indices
        new_tokens = sampled.clone()
        new_tokens.scatter_(1, low_conf_idx, mask_id)
        # Belt-and-braces: never overwrite a frozen position with mask.
        new_tokens = torch.where(frozen, tokens, new_tokens)

        tokens = new_tokens
        history.append(tokens.clone())

    return history


# ════════════════════════════════════════════════════════════════════════════
#   Decoding partial token states (some positions still mask) → waveform
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class DecodeContext:
    """Shapes + spec needed to map token grid → time-domain waveform."""
    target_repr_shape: torch.Size
    spec: Any                     # TransformSpec from stage1.transform
    mask_token_id: int
    codebook_size: int
    C: int
    F: int
    W: int


def make_decode_context(
    cfg: Config, stage1: Stage1VQVAE, prior, device: torch.device,
) -> DecodeContext:
    """Run one dummy stage-1 forward to learn target shapes / spec / latent dims."""
    n_channels = _infer_channels(cfg)
    dummy = torch.zeros(
        (1, n_channels, cfg.dataset.window_length),
        dtype=torch.float32, device=device,
    )
    tf_out = stage1.transform(dummy)
    _, indices, latent_spatial = stage1.encode_tokens(dummy)
    F_, W = int(latent_spatial[0]), int(latent_spatial[1])
    C = int(indices.shape[1])
    return DecodeContext(
        target_repr_shape=tf_out.tensor.shape,
        spec=tf_out.spec,
        mask_token_id=int(prior.mask_token_id),
        codebook_size=int(prior.codebook_size),
        C=C, F=F_, W=W,
    )


@torch.no_grad()
def decode_token_state(
    stage1: Stage1VQVAE,
    tokens_flat: torch.Tensor,
    ctx: DecodeContext,
) -> tuple[torch.Tensor, torch.Tensor]:
    """tokens_flat: (B, C*F*W) → waveform (B, C, T) and mask_fraction (B, C, W).

    Mask positions are temporarily replaced with token 0 to obtain a valid
    embedding so the decoder runs. The returned `mask_fraction[b, c, w]` is
    the fraction of frequencies still at `mask_token_id` for that (channel,
    time-column), which the plot uses to grey-band the meaningless regions.
    """
    B = tokens_flat.shape[0]
    C, F_, W = ctx.C, ctx.F, ctx.W
    tokens_4d = tokens_flat.reshape(B, C, F_, W)
    is_mask = tokens_4d == ctx.mask_token_id
    mask_fraction = is_mask.float().mean(dim=2)                          # (B, C, W)

    fillable = tokens_4d.clone()
    fillable[is_mask] = 0
    indices = fillable.reshape(B, C, F_ * W)
    quantized = stage1.quantizer.embed_indices(indices, (F_, W))
    target_shape = torch.Size((B, *ctx.target_repr_shape[1:]))
    repr_ = stage1.reconstruct_representation(quantized, target_shape)
    waveform = stage1.transform.inverse(repr_, ctx.spec)
    return waveform, mask_fraction


# ════════════════════════════════════════════════════════════════════════════
#   Window collection (sliding windows per split, scaled, with labels)
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class WindowSample:
    """One fixed sliding window with metadata for plotting + matching."""
    inputs: np.ndarray            # (C, T) float32
    labels: np.ndarray            # (T,) int64; -1 if record had no labels
    split: str                    # "train" | "val" | "test"
    record_index: int
    entity_id: str
    dataset_name: str
    feature_names: list[str]
    window_start: int
    window_stop: int


def _infer_channels(cfg: Config) -> int:
    records = load_records(cfg, "train")
    if not records:
        raise RuntimeError("No training records found — cannot determine channel count.")
    return int(records[0].X.shape[1])


def collect_windows_per_split(cfg: Config) -> dict[str, list[WindowSample]]:
    """Slide a window of cfg.dataset.window_length / window_stride over each
    split's pre-scaled records and return the windows with metadata."""
    train_records, val_records, test_records = load_scaled_records(cfg)
    splits_records = {
        "train": train_records,
        "val":   val_records,
        "test":  test_records,
    }
    out: dict[str, list[WindowSample]] = {}
    for split, records in splits_records.items():
        ds = SlidingWindowDataset(
            records, cfg.dataset.window_length, cfg.dataset.window_stride,
            cfg.dataset.window_normalization,
        )
        ws_list: list[WindowSample] = []
        for i in range(len(ds)):
            item = ds[i]
            md = item["metadata"]
            r = records[int(md["record_index"])]
            feat = [] if r.metadata is None else list(r.metadata.feature_names)
            ws_list.append(WindowSample(
                inputs=item["inputs"].numpy().astype(np.float32),
                labels=item["labels"].numpy(),
                split=split,
                record_index=int(md["record_index"]),
                entity_id=str(md["entity_id"]),
                dataset_name=str(md["dataset"]),
                feature_names=feat,
                window_start=int(md["window_start"]),
                window_stop=int(md["window_stop"]),
            ))
        out[split] = ws_list
    return out


# ════════════════════════════════════════════════════════════════════════════
#   Method 1 — Unconditional iterative decoding from all-masked tokens
# ════════════════════════════════════════════════════════════════════════════

def run_unconditional(
    stage1: Stage1VQVAE, stage2: Stage2System,
    n_samples: int, batch_size: int,
    device: torch.device, sampling_seed: int,
    ctx: DecodeContext,
) -> dict[str, Any]:
    """Generate n_samples uncond. samples; keep full per-step token history."""
    prior = stage2.prior
    seq_len = ctx.C * ctx.F * ctx.W
    mask_id = ctx.mask_token_id
    steps = prior.steps

    histories: list[np.ndarray] = []                          # list of (steps+1, seq_len) int32
    final_waveforms: list[np.ndarray] = []
    self_log_prob: list[float] = []

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        bs = end - start
        torch.manual_seed(sampling_seed + start)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(sampling_seed + start)

        initial = torch.full((bs, seq_len), mask_id, dtype=torch.long, device=device)
        history = iterative_decode_with_history(
            prior, initial, frozen_mask=None, steps=steps,
            mask_scheduling_fn=prior.mask_scheduling_fn,
            choice_temperature=prior.choice_temperature,
        )
        stacked = torch.stack(history, dim=1).cpu().numpy().astype(np.int32)   # (bs, steps+1, seq_len)
        for b in range(bs):
            histories.append(stacked[b])

        final_tokens = history[-1]
        waveform, _ = decode_token_state(stage1, final_tokens, ctx)
        final_waveforms.extend(waveform[b].cpu().numpy() for b in range(bs))

        # Self-confidence: re-score the final tokens through the prior and
        # take the mean log-prob of the chosen tokens. Independent of the
        # training set; high values = the prior is happy with what it produced.
        logits = prior._logits(final_tokens)
        log_probs = logits.log_softmax(dim=-1)
        sel = log_probs.gather(-1, final_tokens.unsqueeze(-1)).squeeze(-1)
        self_log_prob.extend(sel.mean(dim=1).cpu().numpy().tolist())

        print(f"[quality_stage2]   uncond batch {start}-{end} of {n_samples} done")

    return {
        "histories": histories,
        "final_waveforms": np.stack(final_waveforms, axis=0),
        "self_log_prob": np.array(self_log_prob),
    }


# ════════════════════════════════════════════════════════════════════════════
#   Method 2 — Conditional inpainting (30% contiguous time mask)
# ════════════════════════════════════════════════════════════════════════════

def _pick_masked_columns(W: int, mask_fraction: float, rng: np.random.Generator) -> tuple[int, int]:
    """Pick a random contiguous interval of `round(mask_fraction*W)` columns."""
    n = max(1, round(mask_fraction * W))
    n = min(n, W - 1)                                  # always keep ≥ 1 col of context
    start = int(rng.integers(0, W - n + 1))
    return start, start + n


def run_conditional(
    stage1: Stage1VQVAE, stage2: Stage2System,
    windows: list[WindowSample], mask_fraction: float, batch_size: int,
    device: torch.device, sampling_seed: int, ctx: DecodeContext,
) -> dict[str, Any]:
    """For each window: encode → blank a random contiguous time interval →
    iterative decode with frozen context → metrics on masked region only."""
    prior = stage2.prior
    seq_len = ctx.C * ctx.F * ctx.W
    mask_id = ctx.mask_token_id
    steps = prior.steps
    C, F_, W = ctx.C, ctx.F, ctx.W
    T = windows[0].inputs.shape[-1] if windows else 0

    histories: list[np.ndarray] = []
    final_waveforms: list[np.ndarray] = []
    original_inputs: list[np.ndarray] = []
    masked_col_ranges: list[tuple[int, int]] = []
    masked_time_ranges: list[tuple[int, int]] = []
    anomaly_intersects: list[bool] = []
    rmse_masked: list[float] = []
    mae_masked: list[float] = []
    cos_masked: list[float] = []
    pear_masked: list[float] = []

    rng = np.random.default_rng(sampling_seed)
    n_samples = len(windows)

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_windows = windows[start: end]
        bs = len(batch_windows)

        x_batch = torch.stack(
            [torch.from_numpy(w.inputs) for w in batch_windows], dim=0,
        ).to(device)

        # Encode → ground-truth tokens (bs, C, F*W).
        _, indices, _ = stage1.encode_tokens(x_batch)
        gt_tokens_3d = indices.long()

        # Pick a random contiguous masked interval per sample.
        col_ranges = [_pick_masked_columns(W, mask_fraction, rng) for _ in range(bs)]

        # Build initial state: ground truth except for the masked columns.
        initial_4d = gt_tokens_3d.reshape(bs, C, F_, W).clone()
        for b, (cs, ce) in enumerate(col_ranges):
            initial_4d[b, :, :, cs: ce] = mask_id
        initial = initial_4d.reshape(bs, seq_len)

        # Frozen mask: True everywhere outside the masked interval (= context).
        frozen_4d = torch.zeros((bs, C, F_, W), dtype=torch.bool, device=device)
        for b, (cs, ce) in enumerate(col_ranges):
            frozen_4d[b, :, :, :cs] = True
            frozen_4d[b, :, :, ce:] = True
        frozen = frozen_4d.reshape(bs, seq_len)

        torch.manual_seed(sampling_seed + start)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(sampling_seed + start)

        history = iterative_decode_with_history(
            prior, initial, frozen_mask=frozen, steps=steps,
            mask_scheduling_fn=prior.mask_scheduling_fn,
            choice_temperature=prior.choice_temperature,
        )
        stacked = torch.stack(history, dim=1).cpu().numpy().astype(np.int32)
        for b in range(bs):
            histories.append(stacked[b])

        final_tokens = history[-1]
        waveform, _ = decode_token_state(stage1, final_tokens, ctx)
        waveform_np = waveform.cpu().numpy()
        x_orig_np = x_batch.cpu().numpy()

        # Per-sample metrics — MASKED REGION ONLY.
        for b, (cs, ce) in enumerate(col_ranges):
            t_start = int(round(cs * T / W))
            t_stop = int(round(ce * T / W))
            t_stop = max(t_stop, t_start + 1)
            x_g = waveform_np[b, :, t_start: t_stop]
            x_o = x_orig_np[b, :, t_start: t_stop]

            diff = (x_g - x_o).astype(np.float64)
            rmse = float(np.sqrt(np.mean(diff ** 2)))
            mae = float(np.mean(np.abs(diff)))
            xg_f = x_g.ravel().astype(np.float64)
            xo_f = x_o.ravel().astype(np.float64)
            denom = float(np.linalg.norm(xg_f) * np.linalg.norm(xo_f))
            cos = float(np.dot(xg_f, xo_f) / denom) if denom > 1e-12 else 0.0
            if np.std(xg_f) < 1e-12 or np.std(xo_f) < 1e-12:
                pear = 1.0 if np.allclose(xg_f, xo_f) else 0.0
            else:
                pear = float(np.corrcoef(xg_f, xo_f)[0, 1])

            labels = batch_windows[b].labels
            anom_inter = False
            if labels.size == T and (labels >= 0).any():
                anom_inter = bool((labels[t_start: t_stop] == 1).any())

            final_waveforms.append(waveform_np[b])
            original_inputs.append(x_orig_np[b])
            masked_col_ranges.append((cs, ce))
            masked_time_ranges.append((t_start, t_stop))
            anomaly_intersects.append(anom_inter)
            rmse_masked.append(rmse)
            mae_masked.append(mae)
            cos_masked.append(cos)
            pear_masked.append(pear)

        print(f"[quality_stage2]   cond batch {start}-{end} of {n_samples} done")

    return {
        "histories": histories,
        "final_waveforms": np.stack(final_waveforms, axis=0) if final_waveforms else np.zeros((0,)),
        "original_inputs": np.stack(original_inputs, axis=0) if original_inputs else np.zeros((0,)),
        "masked_col_ranges": masked_col_ranges,
        "masked_time_ranges": masked_time_ranges,
        "anomaly_intersects": np.array(anomaly_intersects),
        "rmse_masked": np.array(rmse_masked),
        "mae_masked": np.array(mae_masked),
        "cos_masked": np.array(cos_masked),
        "pear_masked": np.array(pear_masked),
        "windows": windows,
    }


# ════════════════════════════════════════════════════════════════════════════
#   Method 3 — Autoregressive sequential decoding (one token per iteration)
# ════════════════════════════════════════════════════════════════════════════
#
# Conceptually: start from all-mask, and at each iteration commit ONE more
# token (in flat sequence order: c-major, freq-middle, time-inner). The
# bidirectional transformer still predicts at every position each step, but
# we only keep position k at iteration k and drop the rest. The "kept-tokens
# context" grows by one at every step, like classic AR sampling — except the
# prior is bidirectional, so position k is conditioned on the already-
# committed prefix [0..k-1] AND on mask placeholders for [k+1..N-1].
#
# Memory: full per-step history is (seq_len + 1) × seq_len longs per sample,
# which explodes for high-channel windows. We keep `n_snaps_per_channel`
# evenly-spaced snapshots WITHIN each channel's AR sweep [c·F·W, (c+1)·F·W],
# union over channels. That is enough to render the per-channel filmstrip
# at the same density as Methods 1/2 without materialising every iteration.

def _per_channel_snapshot_indices(
    C: int, F: int, W: int, n_per_channel: int,
) -> list[int]:
    """Sorted unique global state indices to keep, with `n_per_channel`
    evenly-spaced snapshots inside each channel's AR sweep."""
    indices: set[int] = set()
    for c in range(C):
        chan_start = c * F * W
        chan_end = (c + 1) * F * W
        per_chan = np.linspace(chan_start, chan_end, n_per_channel, dtype=int)
        indices.update(int(x) for x in per_chan.tolist())
    return sorted(indices)


def _channel_snapshot_indices_in_list(
    c: int, F: int, W: int, snap_indices: list[int],
) -> list[int]:
    """Indices INTO `snap_indices` that fall within channel c's AR range.
    Used when slicing a global subsampled history down to one channel's view."""
    chan_start = c * F * W
    chan_end = (c + 1) * F * W
    return [i for i, gi in enumerate(snap_indices) if chan_start <= gi <= chan_end]


@torch.no_grad()
def autoregressive_decode_with_subsampled_history(
    prior, initial_tokens: torch.Tensor, snap_indices: list[int],
) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    """One-token-per-iteration AR decoding. Snapshots saved only at the
    state indices in `snap_indices` (state 0 = before any iteration; state
    k = after iteration k-1, i.e. position k-1 has been committed).

    Returns
    -------
    snapshots : dict[state_idx -> (B, seq_len) uint8 cpu tensor]
    final_tokens : (B, seq_len) long, on device
    """
    seq_len = initial_tokens.shape[1]
    tokens = initial_tokens.clone()
    snap_set = set(snap_indices) | {0, seq_len}
    snapshots: dict[int, torch.Tensor] = {
        0: tokens.detach().to(torch.uint8).cpu().clone(),
    }
    for k in range(seq_len):
        logits = prior._logits(tokens)                                   # (B, L, K)
        sampled = torch.distributions.Categorical(logits=logits).sample()
        # Commit ONLY position k. The bidirectional model produced predictions
        # for every position; we discard all of them except column k.
        tokens = tokens.clone()
        tokens[:, k] = sampled[:, k]
        state_idx = k + 1
        if state_idx in snap_set:
            snapshots[state_idx] = tokens.detach().to(torch.uint8).cpu().clone()
    return snapshots, tokens


def run_autoregressive(
    stage1: Stage1VQVAE, stage2: Stage2System,
    n_samples: int, batch_size: int,
    device: torch.device, sampling_seed: int,
    ctx: DecodeContext,
    n_snaps_per_channel: int,
) -> dict[str, Any]:
    """Generate n_samples uncond AR samples; keep per-channel sub-sampled history."""
    prior = stage2.prior
    seq_len = ctx.C * ctx.F * ctx.W
    mask_id = ctx.mask_token_id

    snap_indices = _per_channel_snapshot_indices(
        ctx.C, ctx.F, ctx.W, n_snaps_per_channel,
    )
    if 0 not in snap_indices:
        snap_indices = [0] + snap_indices
    if seq_len not in snap_indices:
        snap_indices = snap_indices + [seq_len]
    snap_indices = sorted(set(snap_indices))

    histories: list[np.ndarray] = []                          # list of (n_snaps, seq_len) int32
    final_waveforms: list[np.ndarray] = []
    self_log_prob: list[float] = []

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        bs = end - start
        torch.manual_seed(sampling_seed + start)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(sampling_seed + start)

        initial = torch.full((bs, seq_len), mask_id, dtype=torch.long, device=device)
        snapshots, final_tokens = autoregressive_decode_with_subsampled_history(
            prior, initial, snap_indices,
        )

        ordered_keys = sorted(snapshots.keys())
        stacked = torch.stack(
            [snapshots[i] for i in ordered_keys], dim=1,
        ).to(torch.int32).numpy()                                 # (B, n_snaps, seq_len)
        for b in range(bs):
            histories.append(stacked[b])

        waveform, _ = decode_token_state(stage1, final_tokens, ctx)
        final_waveforms.extend(waveform[b].cpu().numpy() for b in range(bs))

        logits = prior._logits(final_tokens)
        log_probs = logits.log_softmax(dim=-1)
        sel = log_probs.gather(-1, final_tokens.unsqueeze(-1)).squeeze(-1)
        self_log_prob.extend(sel.mean(dim=1).cpu().numpy().tolist())

        print(f"[quality_stage2]   AR batch {start}-{end} of {n_samples} done "
              f"(seq_len={seq_len}, snaps={len(snap_indices)})")

    return {
        "histories": histories,
        "final_waveforms": np.stack(final_waveforms, axis=0),
        "self_log_prob": np.array(self_log_prob),
        "snap_indices": snap_indices,
    }


# ════════════════════════════════════════════════════════════════════════════
#   RMSE-to-closest-train (Method 1 triage scalar)
# ════════════════════════════════════════════════════════════════════════════

def rmse_to_closest_train(
    generated: np.ndarray, train_pool: np.ndarray, device: torch.device,
    chunk_query: int = 64, chunk_gallery: int = 1024,
) -> np.ndarray:
    """For each generated window, RMSE to the nearest training window. (N_gen,)."""
    q = generated.reshape(generated.shape[0], -1).astype(np.float32)
    g = train_pool.reshape(train_pool.shape[0], -1).astype(np.float32)
    feat = float(q.shape[1])
    out = np.full(q.shape[0], np.inf, dtype=np.float64)
    with torch.no_grad():
        for qs in range(0, q.shape[0], chunk_query):
            qe = min(qs + chunk_query, q.shape[0])
            qb = torch.from_numpy(q[qs: qe]).to(device)
            qn = (qb * qb).sum(dim=1, keepdim=True)
            best = torch.full((qb.shape[0],), float("inf"), device=device)
            for gs in range(0, g.shape[0], chunk_gallery):
                ge = min(gs + chunk_gallery, g.shape[0])
                gb = torch.from_numpy(g[gs: ge]).to(device)
                gn = (gb * gb).sum(dim=1).unsqueeze(0)
                sse = (qn + gn - 2.0 * (qb @ gb.T)).clamp_(min=0)
                best = torch.minimum(best, sse.min(dim=1).values)
            out[qs: qe] = (best.cpu().numpy() / feat) ** 0.5
    return out


# ════════════════════════════════════════════════════════════════════════════
#   Selection — best / median / worst by metric
# ════════════════════════════════════════════════════════════════════════════

def select_best_median_worst(metric: np.ndarray, k: int = 10) -> dict[str, np.ndarray]:
    """Return indices of {best, median, worst} k samples by `metric` (lower=better)."""
    n = metric.shape[0]
    if n == 0:
        empty = np.array([], dtype=np.int64)
        return {"best": empty, "median": empty, "worst": empty}
    order = np.argsort(metric)
    mid = n // 2
    half = k // 2
    return {
        "best":   order[: min(k, n)],
        "median": order[max(mid - half, 0): max(mid - half, 0) + min(k, n)],
        "worst":  order[max(n - k, 0):],
    }


# ════════════════════════════════════════════════════════════════════════════
#   Plot helpers — per-step state grid + decoded waveform filmstrip
# ════════════════════════════════════════════════════════════════════════════

def _state_grid_for_step(
    history: np.ndarray, step_idx: int, ctx: DecodeContext,
    frozen_mask: np.ndarray | None,
    *, is_autoregressive: bool = False,
) -> np.ndarray:
    """history: (n_states, seq_len) int. Returns (C, F, W) state grid for
    `step_idx`.

    If `is_autoregressive`, "still mask after step > 0" is rendered as
    STATE_MASK (grey) instead of STATE_NEWLY_REMASKED (red). For pure
    AR sampling those positions were never sampled — they are "not yet
    committed", not "actively rejected".
    """
    C, F_, W = ctx.C, ctx.F, ctx.W
    cur = history[step_idx].reshape(C, F_, W)
    if step_idx == 0:
        prev = np.full_like(cur, ctx.mask_token_id)
    else:
        prev = history[step_idx - 1].reshape(C, F_, W)
    if frozen_mask is None:
        frozen_4d = np.zeros((C, F_, W), dtype=bool)
    else:
        frozen_4d = frozen_mask.reshape(C, F_, W)

    is_mask_now = cur == ctx.mask_token_id
    was_mask_prev = prev == ctx.mask_token_id

    state = np.full(cur.shape, STATE_KEPT, dtype=np.int32)
    state[was_mask_prev & ~is_mask_now] = STATE_NEWLY_SAMPLED
    if step_idx > 0:
        if is_autoregressive:
            state[was_mask_prev & is_mask_now] = STATE_MASK
        else:
            state[was_mask_prev & is_mask_now] = STATE_NEWLY_REMASKED
    else:
        state[is_mask_now] = STATE_MASK
    state[frozen_4d] = STATE_FROZEN
    return state


def _decode_history_to_waveforms(
    history: np.ndarray, stage1: Stage1VQVAE, ctx: DecodeContext, device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode every step's tokens at once → (steps+1, C, T) and (steps+1, C, W).

    Applies `stage1.refinement` (when `use_refinement=True`) to every step's
    decoded waveform so the filmstrip overlays gen against original in the
    same magnitude space. Without refinement the inverse-STFT alone produces
    a gen waveform with ~30% of the std of the real signal — which makes the
    overlay visually misleading even though the shape may be correct.
    """
    tokens_t = torch.from_numpy(history.astype(np.int64)).to(device)
    waveforms, mask_fraction = decode_token_state(stage1, tokens_t, ctx)
    if getattr(stage1, "use_refinement", False):
        waveforms = stage1.refinement(waveforms)
    return waveforms.cpu().numpy(), mask_fraction.cpu().numpy()


def plot_sample_channel_evolution(
    out_path: Path,
    history: np.ndarray,                     # (steps+1, seq_len)
    waveforms_per_step: np.ndarray,          # (steps+1, C, T)
    mask_fraction_per_step: np.ndarray,      # (steps+1, C, W)
    channel: int,
    ctx: DecodeContext,
    title: str,
    *,
    frozen_mask: np.ndarray | None = None,   # (seq_len,) or None
    original: np.ndarray | None = None,      # (C, T) or None
    original_rt: np.ndarray | None = None,   # (C, T) or None — stage1.forward(original)
    masked_time_range: tuple[int, int] | None = None,
    anomaly_labels: np.ndarray | None = None,
    metric_summary: str = "",
    is_autoregressive: bool = False,
) -> None:
    """One PNG per (sample, channel) — full step-by-step filmstrip.

    Layout: rows = (steps+1) iterative-decode states, 2 columns:
      LEFT   : F × W token grid (state-coloured + token-id text).
      RIGHT  : decoded waveform for THIS channel at this step, with grey
               bands over still-masked time columns; for cond also overlays
               the original waveform, the masked-region shade, and (test
               only) the red anomaly hatch.

    Three waveform curves on the right panel (when applicable):
      * `original`    — real signal (grey, conditional only).
      * `original_rt` — `stage1.forward(original)` (dashed green, conditional
                        only) — upper bound: even with a perfect prior the
                        gen line cannot beat this.
      * `decoded`     — gen at this step (orange), refinement applied.

    The gap (original ↔ original_rt) is the per-sample stage-1 fidelity
    loss for THIS window; the gap (original_rt ↔ decoded) is the per-sample
    prior fidelity loss at this decoding step.
    """
    C, F_, W = ctx.C, ctx.F, ctx.W
    T = waveforms_per_step.shape[-1]
    n_states = history.shape[0]

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig_h = max(8.0, 0.85 * n_states)
    fig = plt.figure(figsize=(14, fig_h))
    gs = fig.add_gridspec(n_states, 2, width_ratios=[1, 3], hspace=0.35, wspace=0.12)

    # Y range = min/max across original (if any) + original_rt (if any) +
    # every step's decode.
    ts_min = float(waveforms_per_step[:, channel, :].min())
    ts_max = float(waveforms_per_step[:, channel, :].max())
    if original is not None:
        ts_min = min(ts_min, float(original[channel].min()))
        ts_max = max(ts_max, float(original[channel].max()))
    if original_rt is not None:
        ts_min = min(ts_min, float(original_rt[channel].min()))
        ts_max = max(ts_max, float(original_rt[channel].max()))
    pad = 0.05 * (ts_max - ts_min) if ts_max > ts_min else 1.0
    ts_min, ts_max = ts_min - pad, ts_max + pad

    # Adapt cell-text font to grid density.
    fontsize_cell = max(5, min(9, int(60 / max(F_, W))))

    for s in range(n_states):
        # LEFT: token grid for this channel.
        ax_g = fig.add_subplot(gs[s, 0])
        state_grid = _state_grid_for_step(
            history, s, ctx, frozen_mask, is_autoregressive=is_autoregressive,
        )[channel]   # (F, W)
        ax_g.imshow(
            state_grid, cmap=CELL_CMAP, vmin=0, vmax=len(CELL_COLORS) - 1,
            aspect="auto", interpolation="nearest",
        )
        cur_tokens = history[s].reshape(C, F_, W)[channel]
        for f in range(F_):
            for w in range(W):
                tok = int(cur_tokens[f, w])
                txt = "□" if tok == ctx.mask_token_id else f"{tok:d}"
                fg = "white" if state_grid[f, w] in (STATE_FROZEN, STATE_NEWLY_REMASKED) else "black"
                ax_g.text(w, f, txt, ha="center", va="center", fontsize=fontsize_cell, color=fg)
        ax_g.set_xticks([])
        ax_g.set_yticks([])
        label = "init" if s == 0 else f"step {s}"
        ax_g.set_ylabel(label, fontsize=9, rotation=0, labelpad=24, va="center")

        # RIGHT: time-domain waveform (this channel).
        ax_t = fig.add_subplot(gs[s, 1])
        t = np.arange(T)
        if original is not None:
            ax_t.plot(t, original[channel], color="#7f7f7f", linewidth=0.8, alpha=0.7,
                      label="original" if s == 0 else None)
        if original_rt is not None:
            ax_t.plot(t, original_rt[channel], color="#2ca02c", linewidth=0.9,
                      linestyle="--", alpha=0.85,
                      label="stage1_RT(original)" if s == 0 else None)
        ax_t.plot(t, waveforms_per_step[s, channel], color="#ff7f0e", linewidth=1.0,
                  label="decoded" if s == 0 else None)

        # Grey band where the channel still has masked tokens.
        mf = mask_fraction_per_step[s, channel]                             # (W,)
        for w in range(W):
            alpha = float(mf[w]) * 0.35
            if alpha > 0.01:
                t0 = int(round(w * T / W))
                t1 = int(round((w + 1) * T / W))
                ax_t.axvspan(t0, t1, color="gray", alpha=alpha, linewidth=0)

        # Conditional: shade the masked time range yellow for clarity.
        if masked_time_range is not None:
            ts0, ts1 = masked_time_range
            ax_t.axvspan(ts0, ts1, color="khaki", alpha=0.18, linewidth=0)
            # Anomaly intersection (test only) → hatched red.
            if anomaly_labels is not None and (anomaly_labels >= 0).any():
                inter = np.zeros(T, dtype=bool)
                inter[ts0: ts1] = True
                anom = (anomaly_labels == 1) & inter
                if anom.any():
                    ax_t.fill_between(
                        t, ts_min, ts_max, where=anom, color="red", alpha=0.18,
                        step="mid", linewidth=0,
                    )

        ax_t.set_ylim(ts_min, ts_max)
        ax_t.set_xlim(0, T - 1)
        ax_t.tick_params(axis="both", labelsize=7)
        ax_t.grid(alpha=0.2, linewidth=0.4)
        if s == 0:
            ax_t.legend(loc="upper right", fontsize=7)
        if s != n_states - 1:
            ax_t.set_xticklabels([])

    full_title = f"{title}  |  channel={channel}"
    if metric_summary:
        full_title += f"  |  {metric_summary}"
    fig.suptitle(full_title, fontsize=10)

    handles = [mpatches.Rectangle((0, 0), 1, 1, color=c) for c in CELL_COLORS]
    fig.legend(
        handles, CELL_NAMES, loc="lower center", ncol=len(CELL_NAMES),
        fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.0),
    )

    plt.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(str(out_path), dpi=110, bbox_inches="tight")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
#   Distribution sanity plot (Method 1 only)
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _token_histogram_from_windows(
    windows: list[WindowSample], stage1: Stage1VQVAE,
    device: torch.device, ctx: DecodeContext,
) -> np.ndarray:
    """Encode every window with stage 1 and count token-id usage. (codebook_size,)."""
    K = ctx.codebook_size
    counts = np.zeros(K, dtype=np.int64)
    if not windows:
        return counts
    bs = 32
    for s in range(0, len(windows), bs):
        x = torch.stack(
            [torch.from_numpy(w.inputs) for w in windows[s: s + bs]], dim=0,
        ).to(device)
        _, indices, _ = stage1.encode_tokens(x)
        flat = indices.long().reshape(-1).cpu().numpy()
        np.add.at(counts, flat, 1)
    return counts


def _token_histogram_from_histories(
    histories: list[np.ndarray], ctx: DecodeContext,
) -> np.ndarray:
    """Final-step token-id usage across all generated samples."""
    K = ctx.codebook_size
    counts = np.zeros(K, dtype=np.int64)
    for h in histories:
        final = h[-1]
        finite = final[final != ctx.mask_token_id]
        np.add.at(counts, finite.astype(np.int64), 1)
    return counts


@torch.no_grad()
def _stage1_roundtrip(
    pool: np.ndarray,                    # (N, C, T) scaled
    stage1: Stage1VQVAE,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """Pass real windows through `stage1.forward()` — full roundtrip including
    the learned refinement head (if `stage1.use_refinement` is True). This is
    the upper bound for any prior-based generation: even with a perfect prior,
    `gen` cannot beat `stage1_RT`. Used as the third reference curve in the
    distribution sanity plot to decompose the train↔gen gap into
    (train↔stage1_RT) + (stage1_RT↔gen)."""
    out: list[np.ndarray] = []
    for s in range(0, len(pool), batch_size):
        x = torch.from_numpy(pool[s: s + batch_size]).to(device)
        rec = stage1(x)["reconstructed"]
        out.append(rec.cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros_like(pool)


@torch.no_grad()
def _apply_refinement_np(
    waveforms: np.ndarray,                # (N, C, T) inverse-STFT output
    stage1: Stage1VQVAE,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """Run already-decoded (post inverse-STFT) waveforms through the learned
    `RefinementHead`. Stage1 is trained with the refinement applied AFTER the
    inverse STFT, so its decoder learns a representation that the refinement
    expects to fix up. Without refinement the std collapses to ~30% of the
    train target — applying it on generated samples puts gen in the same
    space as `stage1_RT` (apples to apples). No-op if `use_refinement` is off."""
    if not getattr(stage1, "use_refinement", False):
        return waveforms
    out: list[np.ndarray] = []
    for s in range(0, len(waveforms), batch_size):
        x = torch.from_numpy(waveforms[s: s + batch_size]).to(device)
        out.append(stage1.refinement(x).cpu().numpy())
    return np.concatenate(out, axis=0) if out else waveforms


def plot_distribution_sanity(
    out_path: Path,
    gen_waveforms: np.ndarray,           # (N_gen, C, T)
    train_waveforms: np.ndarray,         # (N_ref, C, T)
    gen_token_histogram: np.ndarray,     # (K,)
    train_token_histogram: np.ndarray,   # (K,)
    feature_names: list[str],
    reference_label: str = "train",
    stage1_rt_waveforms: np.ndarray | None = None,    # (N_ref, C, T)
) -> None:
    """Per-channel value histogram, mean |FFT| spectrum, and token-id usage.

    Three curves on histograms + FFT panels:
      * `train`     — real signal in scaled space.
      * `stage1_RT` — real signal passed through `stage1.forward()` (full
                      roundtrip incl. refinement). Upper bound for any
                      prior-based generation.
      * `gen`       — prior-sampled tokens decoded with refinement applied.

    The gap (train ↔ stage1_RT) is the stage-1 fidelity loss; the gap
    (stage1_RT ↔ gen) is the prior fidelity loss. Decomposing them tells you
    where to spend compute (better codebook vs. better prior).

    `reference_label` is used in legends + titles to make explicit which split
    the "real" reference came from (train / val / test). `stage1_rt_waveforms`
    is optional — if None, only train + gen are drawn (legacy 2-curve plot).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    C = gen_waveforms.shape[1]
    names = feature_names or [f"channel_{i}" for i in range(C)]
    fig = plt.figure(figsize=(16, max(6, 2.3 * C + 4)))
    gs = fig.add_gridspec(C + 1, 2, height_ratios=[1] * C + [1.2], hspace=0.55, wspace=0.25)

    for c in range(C):
        ax_h = fig.add_subplot(gs[c, 0])
        ax_h.hist(
            train_waveforms[:, c, :].ravel(), bins=80, density=True, alpha=0.45,
            color="steelblue", label=reference_label,
        )
        if stage1_rt_waveforms is not None:
            ax_h.hist(
                stage1_rt_waveforms[:, c, :].ravel(), bins=80, density=True, alpha=0.45,
                color="forestgreen", label="stage1_RT",
            )
        ax_h.hist(
            gen_waveforms[:, c, :].ravel(), bins=80, density=True, alpha=0.45,
            color="darkorange", label="gen",
        )
        ax_h.set_title(f"{names[c]} value histogram", fontsize=9)
        ax_h.tick_params(axis="both", labelsize=7)
        if c == 0:
            ax_h.legend(fontsize=8)

        ax_s = fig.add_subplot(gs[c, 1])
        gen_fft = np.abs(np.fft.rfft(gen_waveforms[:, c, :], axis=-1)).mean(axis=0)
        train_fft = np.abs(np.fft.rfft(train_waveforms[:, c, :], axis=-1)).mean(axis=0)
        freqs = np.arange(gen_fft.shape[0])
        ax_s.plot(freqs, train_fft, color="steelblue", label=reference_label, linewidth=1.0)
        if stage1_rt_waveforms is not None:
            rt_fft = np.abs(np.fft.rfft(stage1_rt_waveforms[:, c, :], axis=-1)).mean(axis=0)
            ax_s.plot(freqs, rt_fft, color="forestgreen", label="stage1_RT", linewidth=1.0)
        ax_s.plot(freqs, gen_fft, color="darkorange", label="gen", linewidth=1.0)
        ax_s.set_title(
            f"{names[c]} mean |FFT|  "
            f"(N_gen={gen_waveforms.shape[0]} vs N_{reference_label}={train_waveforms.shape[0]})",
            fontsize=9,
        )
        ax_s.tick_params(axis="both", labelsize=7)
        if c == 0:
            ax_s.legend(fontsize=8)

    ax_tk = fig.add_subplot(gs[C, :])
    K = gen_token_histogram.shape[0]
    x = np.arange(K)
    train_norm = train_token_histogram / max(train_token_histogram.sum(), 1)
    gen_norm = gen_token_histogram / max(gen_token_histogram.sum(), 1)
    ax_tk.bar(x - 0.2, train_norm, width=0.4, color="steelblue", label=reference_label)
    ax_tk.bar(x + 0.2, gen_norm, width=0.4, color="darkorange", label="gen")
    ax_tk.set_title(f"token id usage (relative frequency, K={K})", fontsize=9)
    ax_tk.set_xlabel("token id")
    ax_tk.tick_params(axis="both", labelsize=7)
    ax_tk.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
#   Plot orchestration helpers (parallelisable)
# ════════════════════════════════════════════════════════════════════════════
#
# Two-phase pattern: the main process does GPU/torch work (decoding token
# histories into waveforms) and packs each per-channel render into a
# self-contained `dict` task; rendering itself is matplotlib-only and is
# safe to dispatch to a ProcessPoolExecutor of CPU workers. We use
# multiprocessing (not threading): matplotlib is not thread-safe and the
# GIL would serialise the rendering anyway.
#
# Worker function MUST be at module level so it pickles for spawn-based
# pool starts (Windows default).

def _plot_task_worker(task: dict[str, Any]) -> None:
    """Run a single per-channel render. Self-contained; safe in a subprocess."""
    plot_sample_channel_evolution(**task)


def _dispatch_plot_tasks(
    tasks: list[dict[str, Any]],
    pool: Any,                                          # ProcessPoolExecutor | None
    label: str = "",
) -> None:
    """Render `tasks` either serially (pool=None) or in parallel via `pool`."""
    if not tasks:
        return
    if pool is None:
        for t in tasks:
            _plot_task_worker(t)
        return
    print(f"[quality_stage2]   dispatching {len(tasks)} plot tasks "
          f"({label}) across pool workers...")
    # `chunksize` larger than 1 amortises IPC overhead over a few tasks each.
    for _ in pool.map(_plot_task_worker, tasks, chunksize=4):
        pass


def _plot_uncond_samples(
    indices: np.ndarray,
    uncond: dict[str, Any],
    rmse_uncond: np.ndarray,
    output_dir: Path,
    stage1: Stage1VQVAE, ctx: DecodeContext, device: torch.device,
    pool: Any = None,
) -> None:
    """Decode each selected sample once (GPU work in main process), build a
    per-channel render task, then dispatch them serially or to `pool`."""
    tasks: list[dict[str, Any]] = []
    for i in indices:
        i = int(i)
        history = uncond["histories"][i]
        waveforms_per_step, mf_per_step = _decode_history_to_waveforms(
            history, stage1, ctx, device,
        )
        sample_dir = output_dir / f"sample_{i:04d}"
        title = (
            f"Method 1 (uncond) sample {i}  |  "
            f"rmse_closest_train={rmse_uncond[i]:.4f}  |  "
            f"self_logp={uncond['self_log_prob'][i]:+.3f}"
        )
        metric_str = f"rmse_closest_train={rmse_uncond[i]:.4f}"
        for c in range(ctx.C):
            tasks.append(dict(
                out_path=sample_dir / f"channel_{c:02d}.png",
                history=history,
                waveforms_per_step=waveforms_per_step,
                mask_fraction_per_step=mf_per_step,
                channel=c, ctx=ctx, title=title,
                metric_summary=metric_str,
            ))
        print(f"[quality_stage2]   uncond sample {i}: decoded ({ctx.C} channels)")
    _dispatch_plot_tasks(tasks, pool, label="uncond")


def _plot_ar_samples(
    indices: np.ndarray,
    ar_result: dict[str, Any],
    rmse_ar: np.ndarray,
    output_dir: Path,
    stage1: Stage1VQVAE, ctx: DecodeContext, device: torch.device,
    pool: Any = None,
) -> None:
    """Same two-phase pattern as `_plot_uncond_samples`. Per channel we
    slice the global subsampled history down to that channel's AR sweep
    before decoding."""
    snap_indices: list[int] = ar_result["snap_indices"]
    F_, W = ctx.F, ctx.W

    tasks: list[dict[str, Any]] = []
    for i in indices:
        i = int(i)
        full_history = ar_result["histories"][i]                # (n_snaps, seq_len)
        sample_dir = output_dir / f"sample_{i:04d}"
        title = (
            f"Method 3 (AR) sample {i}  |  "
            f"rmse_closest_train={rmse_ar[i]:.4f}  |  "
            f"self_logp={ar_result['self_log_prob'][i]:+.3f}"
        )
        metric_str = f"rmse_closest_train={rmse_ar[i]:.4f}"
        for c in range(ctx.C):
            local_idx = _channel_snapshot_indices_in_list(c, F_, W, snap_indices)
            if len(local_idx) < 2:
                continue
            chan_history = full_history[local_idx]              # (~K, seq_len)
            waveforms_per_step, mf_per_step = _decode_history_to_waveforms(
                chan_history, stage1, ctx, device,
            )
            tasks.append(dict(
                out_path=sample_dir / f"channel_{c:02d}.png",
                history=chan_history,
                waveforms_per_step=waveforms_per_step,
                mask_fraction_per_step=mf_per_step,
                channel=c, ctx=ctx, title=title,
                metric_summary=metric_str,
                is_autoregressive=True,
            ))
        print(f"[quality_stage2]   AR sample {i}: decoded ({ctx.C} channels)")
    _dispatch_plot_tasks(tasks, pool, label="AR")


def _plot_cond_selection(
    cat_name: str, sel_idx: np.ndarray,
    res: dict[str, Any], output_dir: Path,
    stage1: Stage1VQVAE, ctx: DecodeContext, device: torch.device,
    plots_per_category: int,
    pool: Any = None,
) -> None:
    """Decode + build per-channel render tasks for one cond category, then
    dispatch them serially or to `pool`. best/median/worst here is genuine:
    it ranks by RMSE on the masked region against the original."""
    if sel_idx.size == 0:
        print(f"[quality_stage2]   cond {cat_name}: no samples in this category, skipped")
        return
    sub_rmse = res["rmse_masked"][sel_idx]
    picks = select_best_median_worst(sub_rmse, k=plots_per_category)

    tasks: list[dict[str, Any]] = []
    for tag, local_idx in picks.items():
        for li in local_idx:
            i = int(sel_idx[int(li)])
            history = res["histories"][i]
            waveforms_per_step, mf_per_step = _decode_history_to_waveforms(
                history, stage1, ctx, device,
            )
            # Per-sample stage1 roundtrip on the original window — upper
            # bound for the prior. Decomposes the gen↔original gap into
            # (stage1 fidelity) + (prior fidelity) per step.
            with torch.no_grad():
                orig_t = torch.from_numpy(
                    res["original_inputs"][i][None, ...]
                ).to(device)
                original_rt_i = stage1(orig_t)["reconstructed"][0].cpu().numpy()
            sample_dir = output_dir / cat_name / tag / f"sample_{i:04d}"
            cs, ce = res["masked_col_ranges"][i]
            ts, te = res["masked_time_ranges"][i]
            frozen_4d = np.zeros((ctx.C, ctx.F, ctx.W), dtype=bool)
            frozen_4d[:, :, :cs] = True
            frozen_4d[:, :, ce:] = True
            frozen_flat = frozen_4d.reshape(-1)

            win = res["windows"][i]
            metric_str = (
                f"rmse={res['rmse_masked'][i]:.4f}  "
                f"mae={res['mae_masked'][i]:.4f}  "
                f"cos={res['cos_masked'][i]:.4f}  "
                f"pear={res['pear_masked'][i]:.4f}"
            )
            title = (
                f"Method 2 (cond) [{cat_name}/{tag}] sample {i}  |  "
                f"{win.entity_id} start={win.window_start}  |  "
                f"masked cols=[{cs},{ce})  time=[{ts},{te})  "
                f"anom_inter={bool(res['anomaly_intersects'][i])}"
            )
            anomaly_labels = win.labels if (win.labels >= 0).any() else None
            for c in range(ctx.C):
                tasks.append(dict(
                    out_path=sample_dir / f"channel_{c:02d}.png",
                    history=history,
                    waveforms_per_step=waveforms_per_step,
                    mask_fraction_per_step=mf_per_step,
                    channel=c, ctx=ctx, title=title,
                    frozen_mask=frozen_flat,
                    original=res["original_inputs"][i],
                    original_rt=original_rt_i,
                    masked_time_range=(ts, te),
                    anomaly_labels=anomaly_labels,
                    metric_summary=metric_str,
                ))
            print(f"[quality_stage2]   cond {cat_name}/{tag} sample {i}: "
                  f"decoded ({ctx.C} channels)")
    _dispatch_plot_tasks(tasks, pool, label=f"cond/{cat_name}")


# ════════════════════════════════════════════════════════════════════════════
#   Public entry point
# ════════════════════════════════════════════════════════════════════════════

def evaluate(
    cfg: Config | None = None,
    stage1_ckpt: str | Path | None = None,
    stage2_ckpt: str | Path | None = None,
    output_dir: Path | None = None,
    sampling_seed: int | None = None,
    batch_size: int = 64,
    samples_per_split: int = 1,             # multiplier on N_split sliding windows
    mask_fraction: float = 0.30,            # cond: 30% contiguous time mask
    plots_per_category: int = 5,            # best / median / worst plots
    ar_snaps_per_channel: int | None = None,  # AR: snapshots per channel's sweep (None → F*W+1, exactly one token per row)
    plot_workers: int | None = None,        # None → os.cpu_count()-1; 0 = serial; >0 → ProcessPoolExecutor pool size
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """End-to-end: generate samples (uncond + cond), compute metrics, plot."""
    configure_logging()
    cfg = cfg or load_config()
    seed_everything(cfg.seed)
    seed = int(sampling_seed if sampling_seed is not None else cfg.seed)
    if device is None:
        device = resolve_device()
    else:
        device = torch.device(device)

    stage1_ckpt = Path(stage1_ckpt) if stage1_ckpt else best_checkpoint(cfg, "stage1")
    stage2_ckpt = Path(stage2_ckpt) if stage2_ckpt else best_checkpoint(cfg, "stage2")
    if not stage1_ckpt.exists():
        raise FileNotFoundError(f"Stage 1 checkpoint missing: {stage1_ckpt}")
    if not stage2_ckpt.exists():
        raise FileNotFoundError(f"Stage 2 checkpoint missing: {stage2_ckpt}")

    # ── Load models ────────────────────────────────────────────────────────
    n_channels = _infer_channels(cfg)
    example = torch.zeros(
        (1, n_channels, cfg.dataset.window_length), dtype=torch.float32,
    )
    print(f"[quality_stage2] loading stage1: {stage1_ckpt}")
    stage1 = load_stage1(stage1_ckpt, cfg, example, device=device)
    print(f"[quality_stage2] loading stage2: {stage2_ckpt}")
    stage2 = load_stage2(
        stage2_ckpt, cfg, stage1_ckpt=stage1_ckpt,
        stage1_example_inputs=example, device=device,
    )
    stage1.eval()
    stage2.prior.eval()

    ctx = make_decode_context(cfg, stage1, stage2.prior, device)
    print(f"[quality_stage2] latent grid: C={ctx.C}, F={ctx.F}, W={ctx.W}  "
          f"→ seq_len={ctx.C * ctx.F * ctx.W}")
    print(f"[quality_stage2] iterative-decode steps T = {stage2.prior.steps}")

    # AR default: one snapshot per token transition. F*W + 1 captures the
    # state before any commit AND after every commit within the channel.
    if ar_snaps_per_channel is None:
        ar_snaps_per_channel = ctx.F * ctx.W + 1
    print(f"[quality_stage2] AR snapshots per channel = {ar_snaps_per_channel} "
          f"(channel sweep length = {ctx.F * ctx.W})")

    # ── Output dir ─────────────────────────────────────────────────────────
    if output_dir is None:
        root = resolve_path(cfg.paths.model_quality) / "stage2_generation"
        output_dir = root / run_dir_for(cfg, "stage2").relative_to(
            resolve_path(cfg.paths.runs) / "stage2"
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Window collection ─────────────────────────────────────────────────
    print("[quality_stage2] collecting windows per split...")
    splits = collect_windows_per_split(cfg)
    n_train = len(splits["train"])
    n_val = len(splits["val"])
    n_test = len(splits["test"])
    n_total = n_train + n_val + n_test
    print(f"[quality_stage2] sliding windows: "
          f"train={n_train}  val={n_val}  test={n_test}  → total={n_total}")
    if n_train == 0:
        raise RuntimeError("No training windows — refusing to evaluate.")

    feature_names = splits["train"][0].feature_names if splits["train"] else []

    # ── Plot pool ─────────────────────────────────────────────────────────
    # One persistent ProcessPoolExecutor for the whole evaluation: each plot
    # function decodes (in main process, GPU work) and dispatches per-channel
    # render tasks to the pool. Workers are spawned once; spawn cost (~5s on
    # Windows for 4 procs) is amortised over thousands of plots. plot_workers=0
    # falls back to in-process rendering.
    if plot_workers is None:
        import os
        plot_workers = max(1, (os.cpu_count() or 2) - 1)
    plot_pool: Any = None
    if plot_workers > 0:
        from concurrent.futures import ProcessPoolExecutor
        plot_pool = ProcessPoolExecutor(max_workers=plot_workers)
        print(f"[quality_stage2] plot pool: {plot_workers} workers")
    try:
        evaluate_inner(
            cfg, stage1, stage2, ctx, splits, feature_names,
            output_dir, seed, batch_size, samples_per_split,
            mask_fraction, plots_per_category, ar_snaps_per_channel,
            stage1_ckpt, stage2_ckpt, n_train, n_val, n_test, device,
            plot_pool,
        )
    finally:
        if plot_pool is not None:
            plot_pool.shutdown(wait=True)
            print("[quality_stage2] plot pool shut down")

    print(f"[quality_stage2] DONE. Output → {output_dir.resolve()}")
    return {"output_dir": str(output_dir)}


def evaluate_inner(
    cfg: Config, stage1: Stage1VQVAE, stage2: Stage2System,
    ctx: DecodeContext,
    splits: dict[str, list[WindowSample]],
    feature_names: list[str],
    output_dir: Path, seed: int, batch_size: int,
    samples_per_split: int, mask_fraction: float,
    plots_per_category: int, ar_snaps_per_channel: int,
    stage1_ckpt: Path, stage2_ckpt: Path,
    n_train: int, n_val: int, n_test: int,
    device: torch.device,
    plot_pool: Any,
) -> None:
    """Body of evaluate() — split out so the persistent plot_pool is held
    by the caller's try/finally."""
    # ════════════════════════════════════════════════════════════════════════
    # METHOD 1 — UNCONDITIONAL  (train-pool reference only)
    # ════════════════════════════════════════════════════════════════════════
    # Both Methods 1 and 3 are unconditional — the prior produces samples
    # that don't depend on any input split. We compare gen vs the TRAIN pool
    # only. There is no per-sample ground truth, so we DO NOT rank
    # best/median/worst here — that would just measure similarity to
    # existing training windows, not generation quality. We pick a flat
    # random subset of `n_uncond_plots` samples to plot. The actual quality
    # signal is in `distributions.png` (value histograms, FFT spectra,
    # token-id usage).
    train_windows = splits["train"]
    train_pool = np.stack([w.inputs for w in train_windows], axis=0)
    plot_rng = np.random.default_rng(seed + 101)
    n_plot_per_method = 3 * plots_per_category    # match prior visual budget

    print("[quality_stage2] === Method 1: unconditional (train pool) ===")
    uncond_dir = output_dir / "unconditional"
    uncond_dir.mkdir(parents=True, exist_ok=True)
    # Hard cap at 10% of n_train (always). Generation QA doesn't need a sample
    # per training window — a 10% random pool surfaces distributional issues at
    # a fraction of the cost. The `samples_per_split` arg is preserved for
    # backward compat but bounded by the 10% cap.
    n_uncond = min(max(1, int(0.1 * n_train)), samples_per_split * n_train)
    print(f"[quality_stage2]   generating {n_uncond} samples "
          f"(10% cap of N_train={n_train})")

    uncond = run_unconditional(
        stage1, stage2,
        n_samples=n_uncond, batch_size=batch_size, device=device,
        sampling_seed=seed + 7, ctx=ctx,
    )
    rmse_uncond = rmse_to_closest_train(
        uncond["final_waveforms"], train_pool, device=device,
    )
    with (uncond_dir / "summary.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["sample_id", "rmse_to_closest_train", "self_log_prob"])
        for i in range(n_uncond):
            writer.writerow([
                i, float(rmse_uncond[i]), float(uncond["self_log_prob"][i]),
            ])

    print(f"[quality_stage2]   uncond: plotting {min(n_plot_per_method, n_uncond)} "
          f"random samples...")
    plot_indices_uncond = plot_rng.choice(
        n_uncond, size=min(n_plot_per_method, n_uncond), replace=False,
    )
    plot_indices_uncond.sort()
    _plot_uncond_samples(
        plot_indices_uncond, uncond, rmse_uncond, uncond_dir,
        stage1, ctx, device, pool=plot_pool,
    )

    print("[quality_stage2]   uncond: distribution sanity plot...")
    train_token_hist = _token_histogram_from_windows(
        train_windows, stage1, device, ctx,
    )
    gen_token_hist = _token_histogram_from_histories(uncond["histories"], ctx)
    # Decompose train↔gen gap: pass real train through stage1.forward() (incl.
    # refinement) for the upper-bound reference, and apply refinement to gen
    # so both gen and stage1_RT live in the same post-refinement space as
    # train (apples-to-apples — gen via inverse-STFT alone has ~30% of the
    # std the rest of the pipeline operates in).
    stage1_rt_pool = _stage1_roundtrip(train_pool, stage1, device, batch_size=batch_size)
    gen_refined = _apply_refinement_np(
        uncond["final_waveforms"], stage1, device, batch_size=batch_size,
    )
    plot_distribution_sanity(
        uncond_dir / "distributions.png",
        gen_waveforms=gen_refined,
        train_waveforms=train_pool,
        gen_token_histogram=gen_token_hist,
        train_token_histogram=train_token_hist,
        feature_names=feature_names,
        reference_label="train",
        stage1_rt_waveforms=stage1_rt_pool,
    )
    del uncond, rmse_uncond, gen_token_hist, gen_refined

    # ════════════════════════════════════════════════════════════════════════
    # METHOD 3 — AUTOREGRESSIVE  (DISABLED — re-enable by removing the
    # surrounding `"""..."""` markers below)
    # ════════════════════════════════════════════════════════════════════════
    # AR is the dominant compute cost of this script: C·F·W forward passes
    # per generated sample. On toy_*_channel_anomalies (C=6, F=3, W=32) that
    # is 576 forwards × ~300 samples (10% cap) = ~170k forwards per pipeline,
    # multiplied by 50 pipelines on the parallel runner. Distribution sanity
    # over the train pool is already covered by Method 1 (unconditional), so
    # AR is mostly a redundant signal here. Kept verbatim below for ablation.
    n_ar = 0
    """
    print("[quality_stage2] === Method 3: autoregressive (train pool) ===")
    ar_dir = output_dir / "autoregressive"
    ar_dir.mkdir(parents=True, exist_ok=True)
    # Same 10% cap as uncond.
    n_ar = min(max(1, int(0.1 * n_train)), samples_per_split * n_train)
    print(f"[quality_stage2]   generating {n_ar} samples "
          f"(10% cap of N_train={n_train})")

    ar_result = run_autoregressive(
        stage1, stage2,
        n_samples=n_ar, batch_size=batch_size, device=device,
        sampling_seed=seed + 11, ctx=ctx,
        n_snaps_per_channel=ar_snaps_per_channel,
    )
    rmse_ar = rmse_to_closest_train(
        ar_result["final_waveforms"], train_pool, device=device,
    )
    with (ar_dir / "summary.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["sample_id", "rmse_to_closest_train", "self_log_prob"])
        for i in range(n_ar):
            writer.writerow([
                i, float(rmse_ar[i]), float(ar_result["self_log_prob"][i]),
            ])

    print(f"[quality_stage2]   AR: plotting {min(n_plot_per_method, n_ar)} "
          f"random samples...")
    plot_indices_ar = plot_rng.choice(
        n_ar, size=min(n_plot_per_method, n_ar), replace=False,
    )
    plot_indices_ar.sort()
    _plot_ar_samples(
        plot_indices_ar, ar_result, rmse_ar, ar_dir,
        stage1, ctx, device, pool=plot_pool,
    )

    print("[quality_stage2]   AR: distribution sanity plot...")
    gen_token_hist_ar = _token_histogram_from_histories(ar_result["histories"], ctx)
    # Reuse stage1_RT computed above (same train_pool); refine AR gen samples.
    gen_refined_ar = _apply_refinement_np(
        ar_result["final_waveforms"], stage1, device, batch_size=batch_size,
    )
    plot_distribution_sanity(
        ar_dir / "distributions.png",
        gen_waveforms=gen_refined_ar,
        train_waveforms=train_pool,
        gen_token_histogram=gen_token_hist_ar,
        train_token_histogram=train_token_hist,
        feature_names=feature_names,
        reference_label="train",
        stage1_rt_waveforms=stage1_rt_pool,
    )
    del ar_result, rmse_ar, gen_token_hist_ar, train_token_hist, gen_refined_ar, stage1_rt_pool
    """
    # The disabled AR block would have deleted these two; do it explicitly
    # so they don't linger until the function returns.
    del train_token_hist, stage1_rt_pool

    # ════════════════════════════════════════════════════════════════════════
    # METHOD 2 — CONDITIONAL INPAINTING
    # ════════════════════════════════════════════════════════════════════════
    print(f"[quality_stage2] === Method 2: conditional inpainting, "
          f"{samples_per_split}× per split, mask_fraction={mask_fraction:.2f} ===")
    cond_dir = output_dir / "conditional"
    cond_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed + 1)
    cond_results: dict[str, dict[str, Any]] = {}
    split_seed_offset = {"train": 1, "val": 2, "test": 3}
    for split, win_list in splits.items():
        if not win_list:
            continue
        # 10% cap per split — same rationale as uncond/AR.
        n_target = min(
            max(1, int(0.1 * len(win_list))),
            samples_per_split * len(win_list),
        )
        idxs = rng.integers(0, len(win_list), size=n_target)
        windows_for_split = [win_list[int(i)] for i in idxs]
        print(f"[quality_stage2]   cond {split}: {len(windows_for_split)} samples "
              f"(10% cap of N_{split}={len(win_list)})")
        cond_results[split] = run_conditional(
            stage1, stage2,
            windows=windows_for_split, mask_fraction=mask_fraction,
            batch_size=batch_size, device=device,
            sampling_seed=seed + 13 * split_seed_offset[split],
            ctx=ctx,
        )

    # Persist a flat per-sample summary.
    with (cond_dir / "summary.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "split", "sample_id", "anomaly_intersected",
            "rmse_masked", "mae_masked", "cos_masked", "pear_masked",
            "col_start", "col_stop", "t_start", "t_stop",
            "entity_id", "window_start", "window_stop",
        ])
        for split, res in cond_results.items():
            n = len(res["rmse_masked"])
            for i in range(n):
                cs, ce = res["masked_col_ranges"][i]
                ts, te = res["masked_time_ranges"][i]
                w = res["windows"][i]
                writer.writerow([
                    split, i, bool(res["anomaly_intersects"][i]),
                    float(res["rmse_masked"][i]),
                    float(res["mae_masked"][i]),
                    float(res["cos_masked"][i]),
                    float(res["pear_masked"][i]),
                    int(cs), int(ce), int(ts), int(te),
                    w.entity_id, int(w.window_start), int(w.window_stop),
                ])

    # ── Plotting per (category, best/median/worst, sample, channel) ─────
    for split, res in cond_results.items():
        if split == "test":
            anom_mask = res["anomaly_intersects"]
            cats = {
                "test_clean":   ~anom_mask,
                "test_anomaly":  anom_mask,
            }
        else:
            cats = {split: np.ones(res["rmse_masked"].shape[0], dtype=bool)}
        for cat_name, sel_mask in cats.items():
            sel_idx = np.where(sel_mask)[0]
            _plot_cond_selection(
                cat_name=cat_name, sel_idx=sel_idx,
                res=res, output_dir=cond_dir,
                stage1=stage1, ctx=ctx, device=device,
                plots_per_category=plots_per_category,
                pool=plot_pool,
            )

    # ── Snapshot of run config ────────────────────────────────────────────
    with (output_dir / "config.json").open("w", encoding="utf-8") as fh:
        json.dump({
            "n_uncond_total": int(n_uncond),
            "n_ar_total": int(n_ar),
            "ar_snaps_per_channel": int(ar_snaps_per_channel),
            "n_cond_per_split": {
                s: min(
                    max(1, int(0.1 * len(splits[s]))),
                    samples_per_split * len(splits[s]),
                )
                for s in splits if splits[s]
            },
            "cap_fraction": 0.10,
            "mask_fraction": float(mask_fraction),
            "plots_per_category": int(plots_per_category),
            "stage1_ckpt": str(stage1_ckpt),
            "stage2_ckpt": str(stage2_ckpt),
            "prior_T": int(stage2.prior.steps),
            "latent_C_F_W": [int(ctx.C), int(ctx.F), int(ctx.W)],
            "window_length": int(cfg.dataset.window_length),
            "window_stride": int(cfg.dataset.window_stride),
            "seed": int(seed),
        }, fh, indent=2)


# %%
if __name__ == "__main__":
    set_process_title()
    from lib.profiling import profile_run        # opt-in (TVQ_PROFILE=1); no-op when off
    with profile_run("quality_stage2"):
        evaluate()
