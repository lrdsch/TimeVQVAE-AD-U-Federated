#!/usr/bin/env python3
"""Render training-visualization figures from a TVQ_VIZ capture dir.

Consumes the .npz dumps written by lib/train_viz.py during a federated cb_only
run (capture-in-loop) and produces the figures (render-offline):

  (1) recon before/after  -> <fig>/clients/<entity>/recon_round_<r>.png
  (2) latent 2D per client -> <fig>/clients/<entity>/latent_round_<r>.png  (+ grid)
  (3) codebook EMA drift   -> <fig>/codebook_drift_trails.png, cb_drift_curve.png

KEY IDEA (features 2 & 3): a SINGLE PCA basis is fit ONCE on the pooled codebook
snapshots (all rounds) plus a z subsample, then applied to every round. That is
what makes the latent scatter and the codebook trajectories comparable across
rounds — the whole point of "watching it move over time". (sklearn's TSNE has no
.transform(), so it cannot do this; PCA fit-once/transform-always can.)

Usage:
  python scripts/plot_train_viz.py --viz-dir <TVQ_VIZ_DIR>/<tag>/stage1
  # --out defaults to <viz-dir>/figures ; --stage picks the RVQ stage (default 0)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402


# ─── numpy PCA (fit once, transform every frame) ─────────────────────────────

def pca_fit(X: np.ndarray, k: int = 2):
    """Return (mean (D,), comps (k, D)) via SVD. Deterministic, no sklearn."""
    mean = X.mean(axis=0)
    Xc = X - mean
    # economy SVD; rows of Vt are principal axes
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(k, Vt.shape[0])
    return mean, Vt[:k]


def pca_transform(X: np.ndarray, mean: np.ndarray, comps: np.ndarray) -> np.ndarray:
    return (X - mean) @ comps.T


def code_colors(K: int) -> np.ndarray:
    """Stable per-code-id RGBA, reused across every figure so a code keeps its hue."""
    return plt.cm.hsv(np.linspace(0, 1, K, endpoint=False))


# ─── loading ─────────────────────────────────────────────────────────────────

def load_history(viz_dir: Path):
    p = viz_dir / "codebook_history.npz"
    if not p.exists():
        return None
    d = np.load(p)
    return {k: d[k] for k in d.files}


def client_rounds(viz_dir: Path):
    """{entity: {round: npz-path}} for every per-client dump."""
    out: dict[str, dict[int, Path]] = {}
    cdir = viz_dir / "clients"
    if not cdir.is_dir():
        return out
    for ent_dir in sorted(cdir.iterdir()):
        if not ent_dir.is_dir():
            continue
        rounds = {}
        for f in sorted(ent_dir.glob("round_*.npz")):
            rounds[int(f.stem.split("_")[1])] = f
        if rounds:
            out[ent_dir.name] = rounds
    return out


# ─── (1) reconstruction before/after ─────────────────────────────────────────

def plot_recon(npz_path: Path, out_path: Path, entity: str, r: int, max_windows: int = 4):
    d = np.load(npz_path)
    inp, rec = d["inputs"], d["recon"]                 # (n, C, W)
    n = min(max_windows, inp.shape[0])
    C = inp.shape[1]
    fig, axes = plt.subplots(n, 1, figsize=(12, max(2.2, 1.7 * n)), sharex=True)
    axes = np.atleast_1d(axes)
    for i, ax in enumerate(axes):
        for c in range(C):
            ax.plot(inp[i, c], color="steelblue", lw=0.9,
                    label="original" if c == 0 else None)
            ax.plot(rec[i, c], color="darkorange", lw=0.9,
                    label="reconstruction" if c == 0 else None)
        ax.grid(alpha=0.2, lw=0.4)
        ax.set_ylabel(f"win {i}", fontsize=8)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time step")
    fig.suptitle(f"{entity} — reconstruction @ round {r}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ─── (2) latent 2D scatter, shared basis ─────────────────────────────────────

def plot_latent(npz_path: Path, out_path: Path, entity: str, r: int,
                basis, colors, codebook_2d: np.ndarray | None):
    d = np.load(npz_path)
    z, idx = d["z"], d["idx"]                           # (M, dim), (M,)
    z2 = pca_transform(z, *basis)
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.scatter(z2[:, 0], z2[:, 1], s=6, c=colors[idx % len(colors)], alpha=0.55,
               linewidths=0, rasterized=True)
    if codebook_2d is not None:
        ax.scatter(codebook_2d[:, 0], codebook_2d[:, 1], s=90, marker="*",
                   facecolors="none", edgecolors="black", linewidths=1.1,
                   label="codebook")
        ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"{entity} — latent z (PCA-2D, shared basis) @ round {r}", fontsize=10)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.grid(alpha=0.2, lw=0.4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ─── (1b) full-series reconstruction BAND (all overlapping windows per timestep) ─

def plot_series_band(npz_path: Path, out_path: Path, entity: str, r: int):
    d = np.load(npz_path)
    orig, mean = d["orig"], d["mean"]
    t = np.arange(len(orig))
    fig, ax = plt.subplots(figsize=(min(34, max(14, len(orig) / 480)), 4.4))
    ax.fill_between(t, d["bmin"], d["bmax"], color="darkorange", alpha=0.12,
                    label="recon range (all overlapping windows)")
    ax.fill_between(t, d["p10"], d["p90"], color="darkorange", alpha=0.30, label="recon 10–90 pct")
    ax.plot(t, orig, color="steelblue", lw=0.7, label="original", zorder=3)
    ax.plot(t, mean, color="darkorange", lw=0.8, label="recon mean", zorder=4)
    ax.set_title(f"{entity} — full-series reconstruction band @ round {r}  "
                 f"(N={int(d['N'])} windows, W={int(d['W'])}, nMSE={float(d['nmse']):.3f})", fontsize=10)
    ax.set_xlabel("time step"); ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(alpha=0.2, lw=0.4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_series_evolution(npz_paths, out_path: Path, entity: str):
    """Overlay the mean reconstruction of every captured round (gradient by round)
    over the original — see the full-series fit sharpen across training."""
    items = sorted((int(np.load(p)["round"]), p) for p in npz_paths)
    if not items:
        return
    orig = np.load(items[0][1])["orig"]
    t = np.arange(len(orig))
    fig, ax = plt.subplots(figsize=(min(34, max(14, len(orig) / 480)), 4.6))
    ax.plot(t, orig, color="black", lw=0.9, label="original", zorder=len(items) + 2)
    cmap = plt.cm.viridis
    for k, (r, p) in enumerate(items):
        frac = k / max(len(items) - 1, 1)
        ax.plot(t, np.load(p)["mean"], color=cmap(frac), lw=0.7, alpha=0.85,
                label=f"round {r}", zorder=k)
    ax.set_title(f"{entity} — full-series reconstruction MEAN over training "
                 f"(dark→bright = early→late round)", fontsize=10)
    ax.set_xlabel("time step"); ax.legend(loc="upper right", fontsize=7, ncol=3)
    ax.grid(alpha=0.2, lw=0.4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ─── (3) codebook drift trails + curve ───────────────────────────────────────

def plot_codebook_trails(hist, basis, out_path: Path, stage: int = 0):
    cbs = hist["codebooks"][:, stage]                   # (R, K, D)
    R, K, D = cbs.shape
    proj = pca_transform(cbs.reshape(-1, D), *basis).reshape(R, K, 2)
    colors = code_colors(K)
    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    for k in range(K):
        tr = proj[:, k]                                # (R, 2)
        ax.plot(tr[:, 0], tr[:, 1], "-", color=colors[k], alpha=0.35, lw=0.8)
        ax.scatter(tr[0, 0], tr[0, 1], s=22, facecolors="none",
                   edgecolors=colors[k], linewidths=0.9)         # start: hollow
        ax.scatter(tr[-1, 0], tr[-1, 1], s=26, color=colors[k])  # end: filled
        if R > 1:                                       # arrow into the final position
            ax.annotate("", xy=tr[-1], xytext=tr[-2],
                        arrowprops=dict(arrowstyle="->", color=colors[k], alpha=0.7, lw=0.8))
    ax.set_title(f"codebook EMA drift — {K} codes over {R} rounds (stage {stage})\n"
                 "hollow = round 0, filled = final", fontsize=10)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.grid(alpha=0.2, lw=0.4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_drift_curve(hist, out_path: Path):
    rounds = hist["rounds"]
    drift = hist["cb_drift"]
    n_dead = hist["n_dead"].sum(axis=1) if hist["n_dead"].ndim == 2 else hist["n_dead"]
    has_ema = "ema_w" in hist
    fig, ax1 = plt.subplots(figsize=(7.5, 4.4))
    ax1.plot(rounds, drift, "o-", color="crimson", label="cb_drift ‖e^t−e^{t-1}‖")
    ax1.set_xlabel("round"); ax1.set_ylabel("codebook drift", color="crimson")
    ax1.tick_params(axis="y", labelcolor="crimson")
    ax1.grid(alpha=0.2, lw=0.4)
    ax2 = ax1.twinx()
    ax2.plot(rounds, n_dead, "s--", color="slategray", alpha=0.8, label="dead/revived codes")
    ax2.set_ylabel("dead codes", color="slategray")
    ax2.tick_params(axis="y", labelcolor="slategray")
    title = "codebook drift per round"
    if has_ema:
        title += f"  (server EMA on: mean bias-corr w={float(hist['ema_w'][-1].mean()):.3f})"
    ax1.set_title(title, fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ─── shared basis (codebooks ∪ z subsample) ──────────────────────────────────

def build_shared_basis(hist, clients, viz_dir: Path, stage: int, z_cap: int = 5000):
    pool = []
    if hist is not None:
        cbs = hist["codebooks"][:, stage]              # (R, K, D)
        pool.append(cbs.reshape(-1, cbs.shape[-1]))
    zs = []
    for ent, rounds in clients.items():
        for f in rounds.values():
            zs.append(np.load(f)["z"])
    if zs:
        z_all = np.concatenate(zs, axis=0)
        if z_all.shape[0] > z_cap:                     # deterministic stride subsample
            z_all = z_all[:: max(1, z_all.shape[0] // z_cap)]
        pool.append(z_all)
    X = np.concatenate(pool, axis=0)
    return pca_fit(X, k=2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--viz-dir", required=True, type=Path,
                    help="<TVQ_VIZ_DIR>/<tag>/stage1 produced by lib/train_viz.py")
    ap.add_argument("--out", type=Path, default=None, help="figure dir (default <viz-dir>/figures)")
    ap.add_argument("--stage", type=int, default=0, help="RVQ stage to plot (default 0)")
    ap.add_argument("--max-windows", type=int, default=4, help="recon windows per figure")
    args = ap.parse_args()

    viz_dir = args.viz_dir
    if not viz_dir.is_dir():
        raise SystemExit(f"not a dir: {viz_dir}")
    out = args.out or (viz_dir / "figures")
    out.mkdir(parents=True, exist_ok=True)

    meta = {}
    mp = viz_dir / "meta.json"
    if mp.exists():
        meta = json.loads(mp.read_text())

    hist = load_history(viz_dir)
    clients = client_rounds(viz_dir)
    print(f"[plot] {viz_dir}: {len(clients)} client(s), "
          f"history={'yes' if hist is not None else 'no'} -> {out}")

    # Shared PCA basis (features 2 & 3 live in the same coordinates).
    basis = None
    K = int(meta.get("codebook_size", 0)) or (hist["codebooks"].shape[2] if hist is not None else 0)
    colors = code_colors(max(K, 1))
    if hist is not None or clients:
        basis = build_shared_basis(hist, clients, viz_dir, args.stage)

    # (3) codebook drift.
    if hist is not None and basis is not None:
        plot_codebook_trails(hist, basis, out / "codebook_drift_trails.png", stage=args.stage)
        plot_drift_curve(hist, out / "cb_drift_curve.png")
        cb_by_round = {int(r): hist["codebooks"][i, args.stage]
                       for i, r in enumerate(hist["rounds"])}
        print(f"[plot] wrote codebook_drift_trails.png + cb_drift_curve.png")
    else:
        cb_by_round = {}

    # (1)+(2) per client, per round.
    for ent, rounds in clients.items():
        for r, f in sorted(rounds.items()):
            plot_recon(f, out / "clients" / ent / f"recon_round_{r:04d}.png",
                       ent, r, max_windows=args.max_windows)
            if basis is not None:
                cb2d = (pca_transform(cb_by_round[r], *basis)
                        if r in cb_by_round else None)
                plot_latent(f, out / "clients" / ent / f"latent_round_{r:04d}.png",
                            ent, r, basis, colors, cb2d)
        # (1b) full-series reconstruction band, if captured (TVQ_VIZ_SERIES=1).
        series_files = sorted((viz_dir / "clients" / ent).glob("series_round_*.npz"))
        for sf in series_files:
            r = int(sf.stem.split("_")[2])
            plot_series_band(sf, out / "clients" / ent / f"series_round_{r:04d}.png", ent, r)
        if series_files:
            plot_series_evolution(series_files, out / "clients" / ent / "series_evolution.png", ent)
        extra = f" + {len(series_files)} series-band" if series_files else ""
        print(f"[plot] {ent}: {len(rounds)} round(s) -> recon + latent{extra}")

    print(f"[plot] done -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
