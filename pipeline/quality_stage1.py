"""
=============================================================================
  quality_stage1.py — "is stage 1 reconstructing well?"
=============================================================================

For every training window: encode → quantize → decode → compare to the
original. If reconstruction is poor, stage 2 and detection are both doomed.

Produces:
  * `per_window_metrics.csv`  — MAE / MSE / RMSE / cosine / Pearson per window
                                (plus per-channel breakdowns).
  * `summary.json`            — mean / std / percentiles + codebook usage.
  * `highlights/best|worst|median/*.png` — hand-picked comparison plots.
  * `plots/*.png`             — optional (off by default; set max_plots > 0).

Usage:
    from quality_stage1 import evaluate
    evaluate()                              # default config, stage1 best.ckpt

Debug tips:
  * Set `max_plots=10` to eyeball a handful of windows quickly.
  * A very low `fidelity_rate` (< 0.5) means stage 1 is broken — no point training stage 2.
  * `dead_codes` near codebook_size → your VQ has collapsed. Try more data, warmup, or smaller codebook.
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import Config, load_config
from data import SlidingWindowDataset, TimeSeriesRecord, load_scaled_records
from stage1 import load_stage1
from utils import (
    best_checkpoint, configure_logging, resolve_path, run_dir_for,
    seed_everything,
)


# ─── Metrics ─────────────────────────────────────────────────────────────────

def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    if np.std(a) < 1e-12 and np.std(b) < 1e-12:
        return 1.0 if np.allclose(a, b) else 0.0
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.dot(a, b) / denom)


def _window_metrics(original: np.ndarray, reconstructed: np.ndarray) -> dict[str, Any]:
    """Inputs shape (C, T). Returns a flat dict of scalar & per-channel metrics."""
    diff = reconstructed.astype(np.float64) - original.astype(np.float64)
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    ch_mae = [float(np.mean(np.abs(diff[c]))) for c in range(original.shape[0])]
    ch_rmse = [float(np.sqrt(np.mean(diff[c] ** 2))) for c in range(original.shape[0])]
    ch_pearson = [_safe_pearson(original[c], reconstructed[c]) for c in range(original.shape[0])]
    return {
        "mae": mae, "mse": mse, "rmse": rmse,
        "cosine_similarity": _cosine(original, reconstructed),
        "pearson_correlation": _safe_pearson(original, reconstructed),
        "per_channel_mae": ch_mae,
        "per_channel_rmse": ch_rmse,
        "per_channel_pearson": ch_pearson,
    }


# ─── Plotting (per-window comparison) ────────────────────────────────────────

def _save_reconstruction_plot(
    path: Path,
    original: np.ndarray,
    reconstructed: np.ndarray,
    feature_names: list[str],
    title: str,
    metrics: dict[str, Any],
    labels: np.ndarray | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = original.shape[0]
    names = feature_names or [f"channel_{i}" for i in range(n)]
    t = np.arange(original.shape[-1])

    fig, axes = plt.subplots(n, 1, figsize=(16, max(4.5, 2.2 * n)), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    lo = float(min(original.min(), reconstructed.min()))
    hi = float(max(original.max(), reconstructed.max()))
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    anomaly_mask = (
        (labels == 1) if labels is not None and labels.size == original.shape[-1] else None
    )
    for c, ax in enumerate(axes):
        ax.plot(t, original[c], color="steelblue", linewidth=1.0, label="Original")
        ax.plot(t, reconstructed[c], color="darkorange", linewidth=1.0, label="Reconstructed")
        ax.set_ylabel(names[c], fontsize=8, rotation=0, labelpad=48, va="center")
        ax.set_ylim(lo - pad, hi + pad)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(alpha=0.2, linewidth=0.4)
        if anomaly_mask is not None and anomaly_mask.any():
            ax.fill_between(
                t, lo - pad, hi + pad, where=anomaly_mask,
                color="red", alpha=0.15, step="mid",
                label="anomaly" if c == 0 else None,
            )
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time step")
    fig.suptitle(
        f"{title}\nRMSE={metrics['rmse']:.4f} | MAE={metrics['mae']:.4f} | "
        f"Cos={metrics['cosine_similarity']:.4f} | Pearson={metrics['pearson_correlation']:.4f}",
        fontsize=11,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─── Collect training windows (same as stage 1 / stage 2) ────────────────────

def _records_to_windows(
    records: list[TimeSeriesRecord], window_length: int, stride: int,
    window_normalization: str = "none",
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """Slide windows over already-scaled records and return (windows, meta)."""
    ds = SlidingWindowDataset(records, window_length, stride, window_normalization)
    windows, meta = [], []
    for i in range(len(ds)):
        item = ds[i]
        md = dict(item["metadata"])
        r = records[int(md["record_index"])]
        feat = [] if r.metadata is None else list(r.metadata.feature_names)
        windows.append(np.asarray(item["inputs"].numpy(), dtype=np.float32))
        meta.append({
            "entity_id": str(md["entity_id"]),
            "dataset_name": str(md["dataset"]),
            "record_index": int(md["record_index"]),
            "window_start": int(md["window_start"]),
            "window_stop": int(md["window_stop"]),
            "feature_names": feat,
        })
    return windows, meta


# ─── Full-series reconstruction (entire record, not just one window) ────────

@torch.no_grad()
def _reconstruct_full_record(
    model, X: np.ndarray, window_length: int,
    device: torch.device, batch_size: int = 64,
) -> np.ndarray | None:
    """Reconstruct an entire record by tiling non-overlapping windows.

    `X`: (T, C) already-scaled. Splits into non-overlapping windows of
    `window_length`; if T is not a multiple, a tail-aligned window covers the
    remainder (its overlap with the previous window is harmless — both are
    valid reconstructions and the tail's content is what's plotted at the end).
    Returns (T, C) or None if T < window_length.
    """
    T, C = X.shape
    W = window_length
    if T < W:
        return None
    starts = list(range(0, T - W + 1, W))
    if not starts or starts[-1] + W < T:
        starts.append(T - W)
    recon = np.zeros((T, C), dtype=np.float32)
    for cs in range(0, len(starts), batch_size):
        chunk_starts = starts[cs: cs + batch_size]
        batch = np.stack([X[s: s + W].T for s in chunk_starts])      # (B, C, W)
        out = model(torch.from_numpy(batch).to(device))
        recs = out["reconstructed"].detach().cpu().numpy().astype(np.float32)
        for s, r in zip(chunk_starts, recs):
            recon[s: s + W] = r.T
    return recon


def _save_full_series_plot(
    path: Path, original: np.ndarray, reconstructed: np.ndarray,
    feature_names: list[str], title: str, labels: np.ndarray | None = None,
) -> None:
    """Plot original vs reconstructed for an entire record (T, C layout)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    T, C = original.shape
    names = feature_names or [f"channel_{i}" for i in range(C)]
    t = np.arange(T)
    fig, axes = plt.subplots(C, 1, figsize=(20, max(4.5, 1.8 * C)), sharex=True)
    axes = np.atleast_1d(axes)
    diff = reconstructed - original
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    for c, ax in enumerate(axes):
        ax.plot(t, original[:, c], color="steelblue", linewidth=0.6, label="Original")
        ax.plot(t, reconstructed[:, c], color="darkorange",
                linewidth=0.6, alpha=0.85, label="Reconstructed")
        ax.set_ylabel(names[c], fontsize=8, rotation=0, labelpad=48, va="center")
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(alpha=0.2, linewidth=0.4)
        if labels is not None and labels.size == T:
            mask = labels == 1
            if mask.any():
                ymin, ymax = ax.get_ylim()
                ax.fill_between(
                    t, ymin, ymax, where=mask, color="red", alpha=0.15,
                    step="mid", label="anomaly" if c == 0 else None,
                )
                ax.set_ylim(ymin, ymax)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time step")
    fig.suptitle(f"{title}\nRMSE={rmse:.4f} | MAE={mae:.4f}", fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _save_split_highlights(
    highlights_dir: Path, split: str,
    windows: list[np.ndarray], recons: list[np.ndarray],
    meta: list[dict[str, Any]], all_metrics: list[dict[str, Any]],
) -> None:
    """Save best/worst/median (by RMSE) reconstruction plots for one split."""
    rmse_values = [m["rmse"] for m in all_metrics]
    if not rmse_values:
        return
    sorted_idx = np.argsort(rmse_values)
    categories = {
        "best":   sorted_idx[:5],
        "worst":  sorted_idx[-5:],
        "median": sorted_idx[len(sorted_idx) // 2 - 2: len(sorted_idx) // 2 + 3],
    }
    for cat, idxs in categories.items():
        cat_dir = highlights_dir / cat
        cat_dir.mkdir(parents=True, exist_ok=True)
        for idx in idxs:
            i = int(idx)
            md = meta[i]
            name = (
                f"window_{i:04d}_{md['entity_id']}_"
                f"{md['window_start']:06d}_{md['window_stop']:06d}.png"
            )
            _save_reconstruction_plot(
                cat_dir / name,
                windows[i], recons[i], md["feature_names"],
                f"{md['dataset_name']} - {md['entity_id']} - "
                f"{split}[{md['window_start']}:{md['window_stop']}] ({cat})",
                all_metrics[i],
            )


def _full_series_reconstruction(
    cfg: Config, model, output_dir: Path,
    device: torch.device, batch_size: int,
    splits: dict[str, list[TimeSeriesRecord]],
) -> None:
    """For every split × record: reconstruct end-to-end and save one plot."""
    full_dir = output_dir / "full_series"
    full_dir.mkdir(parents=True, exist_ok=True)
    W = cfg.dataset.window_length
    for split, records in splits.items():
        for r in records:
            if r.metadata is None:
                continue
            entity = r.metadata.entity_id
            recon = _reconstruct_full_record(model, r.X, W, device, batch_size)
            if recon is None:
                print(f"[quality_stage1] {split}/{entity}: T={r.X.shape[0]} < "
                      f"window={W}, skipping full-series plot")
                continue
            _save_full_series_plot(
                full_dir / f"{split}_{entity}.png",
                original=r.X, reconstructed=recon,
                feature_names=list(r.metadata.feature_names),
                title=f"{r.metadata.dataset} - {entity} - {split} "
                      f"(T={r.X.shape[0]}, full-series reconstruction)",
                labels=r.y,
            )
            print(f"[quality_stage1] full-series saved: {split}/{entity} (T={r.X.shape[0]})")


# ─── Core evaluation ─────────────────────────────────────────────────────────

def evaluate(
    cfg: Config | None = None,
    stage1_ckpt: str | Path | None = None,
    output_dir: Path | None = None,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 64,
    max_plots: int = 0,                      # 0 = no per-window plots, only highlights
) -> dict[str, Path]:
    """Evaluate stage 1 VQ-VAE reconstruction on all training windows.

    `max_plots`:
      * 0  → no per-window plots (default, fast).
      * N  → render at most N per-window plots (parallel).
      * -1 → every window (legacy, disk-heavy).
    """
    configure_logging()
    cfg = cfg or load_config()
    seed_everything(cfg.seed)
    stage1_ckpt = Path(stage1_ckpt) if stage1_ckpt else best_checkpoint(cfg, "stage1")
    if not stage1_ckpt.exists():
        raise FileNotFoundError(f"Stage 1 checkpoint missing: {stage1_ckpt}")

    # ── Load windows + model ───────────────────────────────────────────────
    train_records, val_records, test_records = load_scaled_records(cfg)
    windows, meta = _records_to_windows(
        train_records, cfg.dataset.window_length, cfg.dataset.window_stride,
        cfg.dataset.window_normalization,
    )
    if not windows:
        raise ValueError("No training windows found.")
    n_ch, n_win = windows[0].shape[0], len(windows)
    print(f"[quality_stage1] {n_win} training windows, {n_ch} channels, "
          f"length {cfg.dataset.window_length}")

    # Hard cap at 10% of the dataset (always, no knob). Reconstructing every
    # training window is wasteful for QA: a random 10% sample is enough to
    # surface systematic reconstruction issues without paying the full cost.
    # Cap is uniform random (seeded) so the sample is representative, not
    # biased toward the head/tail of the series.
    n_cap = max(1, int(0.1 * n_win))
    if n_win > n_cap:
        rng = np.random.default_rng(cfg.seed + 31337)
        sel = np.sort(rng.choice(n_win, size=n_cap, replace=False)).tolist()
        windows = [windows[i] for i in sel]
        meta = [meta[i] for i in sel]
        n_win = len(windows)
        print(f"[quality_stage1] capped to {n_win} windows (10% of dataset)")

    example = torch.zeros((1, n_ch, cfg.dataset.window_length), dtype=torch.float32)
    model = load_stage1(stage1_ckpt, cfg, example, device=torch.device(device))

    # ── Reconstruct in batches ────────────────────────────────────────────
    recons: list[np.ndarray] = []
    all_indices: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, n_win, batch_size):
            stop = min(start + batch_size, n_win)
            batch = torch.stack([torch.from_numpy(w) for w in windows[start: stop]]).to(device)
            out = model(batch)
            recons.extend(out["reconstructed"].detach().cpu().numpy().astype(np.float32))
            all_indices.extend(out["quantizer_output"].indices.detach().cpu().numpy())
            print(f"  reconstructed {stop}/{n_win}", end="\r")
    print()

    # ── Output directory ──────────────────────────────────────────────────
    if output_dir is None:
        root = resolve_path(cfg.paths.model_quality) / "stage1_reconstruction"
        output_dir = root / run_dir_for(cfg, "stage1").relative_to(resolve_path(cfg.paths.runs) / "stage1")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    highlights_train_dir = output_dir / "highlights_train"
    highlights_train_dir.mkdir(parents=True, exist_ok=True)
    render_plots = max_plots != 0
    per_win_limit = n_win if max_plots < 0 else min(int(max_plots), n_win)
    plots_dir = output_dir / "plots" if render_plots else None
    if plots_dir is not None:
        plots_dir.mkdir(parents=True, exist_ok=True)

    # ── Per-window metrics + optional plots (parallel) ────────────────────
    rows: list[dict[str, Any]] = []
    all_metrics: list[dict[str, Any]] = []
    plot_tasks = []
    executor = ProcessPoolExecutor() if render_plots else None
    try:
        for i in range(n_win):
            m = _window_metrics(windows[i], recons[i])
            all_metrics.append(m)
            md = meta[i]
            plot_path = ""
            if plots_dir is not None and i < per_win_limit:
                plot_name = (
                    f"window_{i:04d}_{md['entity_id']}_"
                    f"{md['window_start']:06d}_{md['window_stop']:06d}.png"
                )
                plot_path = str(plots_dir / plot_name)
                plot_tasks.append(executor.submit(
                    _save_reconstruction_plot, Path(plot_path),
                    windows[i], recons[i], md["feature_names"],
                    f"{md['dataset_name']} - {md['entity_id']} - "
                    f"train[{md['window_start']}:{md['window_stop']}]",
                    m,
                ))
            rows.append({
                "window_index": i,
                "entity_id": md["entity_id"], "dataset_name": md["dataset_name"],
                "record_index": md["record_index"],
                "window_start": md["window_start"], "window_stop": md["window_stop"],
                "mae": m["mae"], "mse": m["mse"], "rmse": m["rmse"],
                "cosine_similarity": m["cosine_similarity"],
                "pearson_correlation": m["pearson_correlation"],
                "per_channel_mae": json.dumps(m["per_channel_mae"]),
                "per_channel_rmse": json.dumps(m["per_channel_rmse"]),
                "per_channel_pearson": json.dumps(m["per_channel_pearson"]),
                "plot_path": plot_path,
            })
        if plot_tasks:
            print(f"[quality_stage1] rendering {len(plot_tasks)} per-window plots...")
            for i, _ in enumerate(as_completed(plot_tasks), 1):
                if i % 100 == 0 or i == len(plot_tasks):
                    print(f"  saved {i}/{len(plot_tasks)}", end="\r")
            print()
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    # ── Write per-window CSV ──────────────────────────────────────────────
    csv_path = output_dir / "per_window_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # ── Highlight plots: best / worst / median (by RMSE) for train ───────
    rmse_values = [r["rmse"] for r in rows]
    _save_split_highlights(
        highlights_train_dir, "train", windows, recons, meta, all_metrics,
    )

    # ── Summary: aggregates + codebook usage ─────────────────────────────
    cb_size = cfg.quantizer.codebook_size
    all_codes = np.concatenate([idx.ravel() for idx in all_indices])
    unique = set(int(c) for c in all_codes)
    summary: dict[str, Any] = {}
    rmse_arr = np.asarray(rmse_values)
    for key in ("mae", "mse", "rmse", "cosine_similarity", "pearson_correlation"):
        vals = [r[key] for r in rows]
        summary[f"mean_{key}"] = float(np.mean(vals))
        summary[f"std_{key}"] = float(np.std(vals))
    summary["rmse_p50"] = float(np.percentile(rmse_arr, 50))
    summary["rmse_p90"] = float(np.percentile(rmse_arr, 90))
    summary["rmse_p95"] = float(np.percentile(rmse_arr, 95))
    summary["rmse_p99"] = float(np.percentile(rmse_arr, 99))
    fidelity_threshold = summary["mean_rmse"] + summary["std_rmse"]
    summary["fidelity_threshold"] = fidelity_threshold
    summary["fidelity_rate"] = float(np.mean(rmse_arr < fidelity_threshold))
    summary["fidelity_count"] = int(np.sum(rmse_arr < fidelity_threshold))
    summary["codebook_size"] = cb_size
    summary["unique_codes_used"] = len(unique)
    summary["codebook_usage_percent"] = 100.0 * len(unique) / max(cb_size, 1)
    summary["dead_codes"] = cb_size - len(unique)
    summary["num_windows"] = n_win
    summary["num_channels"] = n_ch
    summary["window_length"] = cfg.dataset.window_length
    summary["stage1_checkpoint"] = str(stage1_ckpt)

    with (output_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("[quality_stage1] summary:")
    print(f"  RMSE:              {summary['mean_rmse']:.4f} ± {summary['std_rmse']:.4f}")
    print(f"  Pearson:           {summary['mean_pearson_correlation']:.4f}")
    print(f"  Cosine:            {summary['mean_cosine_similarity']:.4f}")
    print(f"  Codebook usage:    {summary['codebook_usage_percent']:.1f}% "
          f"({summary['unique_codes_used']}/{cb_size}, {summary['dead_codes']} dead)")
    print(f"  Fidelity rate:     {summary['fidelity_rate']:.1%} "
          f"({summary['fidelity_count']}/{n_win})")

    # ── Highlight plots for val / test ───────────────────────────────────
    for split, split_records in [("val", val_records), ("test", test_records)]:
        s_windows, s_meta = _records_to_windows(
            split_records, cfg.dataset.window_length, cfg.dataset.window_stride,
            cfg.dataset.window_normalization,
        )
        if not s_windows:
            print(f"[quality_stage1] {split}: no windows, skipping highlights")
            continue
        s_n = len(s_windows)
        print(f"[quality_stage1] {split}: {s_n} windows, reconstructing for highlights...")
        s_recons: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, s_n, batch_size):
                stop = min(start + batch_size, s_n)
                batch = torch.stack([torch.from_numpy(w) for w in s_windows[start: stop]]).to(device)
                out = model(batch)
                s_recons.extend(out["reconstructed"].detach().cpu().numpy().astype(np.float32))
                print(f"  reconstructed {stop}/{s_n}", end="\r")
        print()
        s_metrics = [_window_metrics(w, r) for w, r in zip(s_windows, s_recons)]
        _save_split_highlights(
            output_dir / f"highlights_{split}", split,
            s_windows, s_recons, s_meta, s_metrics,
        )
        if split == "test":
            anom_dir = output_dir / "highlights_test" / "anomaly_window"
            anom_dir.mkdir(parents=True, exist_ok=True)
            saved = 0
            for i, md in enumerate(s_meta):
                if saved >= 10:
                    break
                rec = test_records[md["record_index"]]
                if rec.y is None:
                    continue
                window_labels = rec.y[md["window_start"]: md["window_stop"]]
                if not np.any(window_labels == 1):
                    continue
                name = (
                    f"window_{i:04d}_{md['entity_id']}_"
                    f"{md['window_start']:06d}_{md['window_stop']:06d}.png"
                )
                _save_reconstruction_plot(
                    anom_dir / name,
                    s_windows[i], s_recons[i], md["feature_names"],
                    f"{md['dataset_name']} - {md['entity_id']} - "
                    f"test[{md['window_start']}:{md['window_stop']}] (anomaly_window)",
                    s_metrics[i],
                    labels=window_labels,
                )
                saved += 1
            print(f"[quality_stage1] test: saved {saved} anomaly window plot(s)")

    # ── Full-series reconstruction for every split / entity ──────────────
    print("[quality_stage1] full-series reconstruction (train/val/test)...")
    _full_series_reconstruction(
        cfg, model, output_dir, torch.device(device), batch_size,
        splits={"train": train_records, "val": val_records, "test": test_records},
    )

    print(f"[quality_stage1] output: {output_dir.resolve()}")
    return {"root": output_dir, "metrics": csv_path, "summary": output_dir / "summary.json"}


if __name__ == "__main__":
    set_process_title()
    from lib.profiling import profile_run        # opt-in (TVQ_PROFILE=1); no-op when off
    with profile_run("quality_stage1"):
        evaluate()
