#!/usr/bin/env python3
"""Full-series Stage-1 reconstruction: train + test side by side, with the
overlap-window RECONSTRUCTION BAND.

For each of 3 clients (toy / ucr / wsd), trains Stage-1 as configured NOW
(wb16 / td64 / window 128), then reconstructs the ENTIRE train series and the
ENTIRE test series with stride-1 sliding windows. Because windows overlap, every
timestep is reconstructed by up to `window_length` different windows — we draw
that spread as a shaded band (min–max + 10–90 pct) plus the mean, over the
original. Train and test are laid out on ONE x-axis with a divider.

Normalization matches the pipeline exactly: per-entity z-score fit on TRAIN
(`scaling="per_entity_standard"`), no per-window norm (`window_normalization="none"`),
so the whole series lives in ONE normalized space and the band is coherent.

    CUDA_VISIBLE_DEVICES=1 python scripts/recon_full_series.py --steps 3000 --dpi 300
"""
from __future__ import annotations
import argparse, glob, os, sys
import numpy as np, torch
sys.path.insert(0, "pipeline"); sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from config import load_config                                        # noqa: E402
import local_recon_autopsy as A                                       # variants/get_windows  # noqa: E402
from retrain_capacity import train_steps, recalibrate_bn             # noqa: E402
import matplotlib; matplotlib.use("Agg")                              # noqa: E402
import matplotlib.pyplot as plt                                       # noqa: E402

PAIRS = [
    ("toy_fed_uni",  "uni_00",  "toy",  "toy  (toy_fed_uni)",     "#1c7a4b"),
    ("ucr_pool",     "ucr_001", "ucr",  "ucr  (ucr_pool, real)",  "#d98a00"),
    ("wsd_fed",      "kpi_015", "wsd",  "wsd  (wsd_fed, real)",   "#b3323f"),
]


def cfg_for(ds, ent):
    os.environ["DATASET_NAME"] = ds; os.environ["DATASET_ENTITY"] = ent
    return load_config()


def load_series(ds, ent):
    tr = np.load(glob.glob(f"data/raw/{ds}/train/{ent}*")[0]).astype(np.float32)   # (Ltr, C)
    te = np.load(glob.glob(f"data/raw/{ds}/test/{ent}*")[0]).astype(np.float32)    # (Lte, C)
    lf = glob.glob(f"data/raw/{ds}/test_label/{ent}*")
    lab = np.load(lf[0]).astype(np.float32).reshape(-1) if lf else np.zeros(len(te), np.float32)
    return tr, te, lab


def zscore(series, mean, std):
    return (series - mean) / std


def recon_band(m, series_norm, W, device, bs=512):
    """series_norm: (L, C=1). Returns per-timestep stats across overlapping windows."""
    L = len(series_norm); N = L - W + 1
    idx = np.arange(N)[:, None] + np.arange(W)[None, :]          # (N, W)
    Xnp = np.transpose(series_norm[idx], (0, 2, 1)).copy()       # (N, C, W)
    recs = []
    for i in range(0, N, bs):
        xb = torch.from_numpy(Xnp[i:i + bs]).to(device)
        recs.append(A.variants(m, xb)[0]["full"].numpy())       # (b, C, W)
    rec = np.concatenate(recs, 0)                                # (N, C, W)
    band = np.full((L, W), np.nan, np.float32)                  # scatter along the diagonal
    ar = np.arange(N)
    for j in range(W):
        band[ar + j, j] = rec[:, 0, j]
    lo, hi = np.nanpercentile(band, [10, 90], axis=1)
    return dict(orig=series_norm[:, 0], mean=np.nanmean(band, 1),
                bmin=np.nanmin(band, 1), bmax=np.nanmax(band, 1), p10=lo, p90=hi,
                nmse=float(np.nansum((np.nanmean(band, 1) - series_norm[:, 0]) ** 2) /
                           max(np.sum(series_norm[:, 0] ** 2), 1e-12)))


def compute_one(ds, ent, steps, lr, seed, device):
    cfg = cfg_for(ds, ent); W = cfg.dataset.window_length
    tr, te, lab = load_series(ds, ent)
    mean = tr.mean(0); std = np.maximum(tr.std(0), 1e-4)         # per-entity z-score, fit on TRAIN
    m, c, _, npar = train_steps(cfg, ent, steps, lr, device, seed=seed)
    Xtr, _ = A.get_windows(cfg, ent, device, split="train")
    recalibrate_bn(m, Xtr)
    B_tr = recon_band(m, zscore(tr, mean, std), W, device)
    B_te = recon_band(m, zscore(te, mean, std), W, device)
    print(f"  {ds}/{ent}: W={W} params={npar/1000:.0f}k  "
          f"train nMSE={B_tr['nmse']:.3f}  test nMSE={B_te['nmse']:.3f}  "
          f"Ltr={len(tr)} Lte={len(te)}", flush=True)
    return dict(W=W, npar=int(npar), tr=B_tr, te=B_te,
                lab=(lab / (lab.max() if lab.max() > 0 else 1.0)), Ltr=len(tr), Lte=len(te))


def plot_one(r, label, col, dpi, out, figw=0.0):
    tr, te = r["tr"], r["te"]; Ltr = r["Ltr"]
    L = r["Ltr"] + r["Lte"]
    if figw <= 0:
        figw = float(np.clip(L / 480.0, 16, 34))
    # Agg backend hard-caps a dimension at 2^15 px — keep figw*dpi under it.
    max_in = 32000.0 / dpi
    if figw > max_in:
        print(f"[recon-full] figw {figw:.0f}in x {dpi}dpi exceeds Agg 32k-px limit; "
              f"clamping to {max_in:.0f}in", flush=True)
        figw = max_in
    fig, ax = plt.subplots(figsize=(figw, 4.6))
    # anomaly shading (test portion)
    lab = r["lab"]; t_te = np.arange(r["Lte"]) + Ltr
    if (lab > 0).any():
        ymin = min(tr["orig"].min(), te["orig"].min()); ymax = max(tr["orig"].max(), te["orig"].max())
        ax.fill_between(t_te, ymin, ymax, where=(lab > 0), color="#f2c14e", alpha=0.30,
                        label="anomaly (test)", zorder=0)
    for seg, x0, tag in ((tr, 0, "train"), (te, Ltr, "test")):
        t = np.arange(len(seg["orig"])) + x0
        first = (x0 == 0)
        ax.fill_between(t, seg["bmin"], seg["bmax"], color=col, alpha=0.12,
                        label=("recon range (all overlap windows)" if first else None), zorder=1)
        ax.fill_between(t, seg["p10"], seg["p90"], color=col, alpha=0.28,
                        label=("recon 10–90 pct" if first else None), zorder=2)
        ax.plot(t, seg["mean"], color=col, lw=0.7,
                label=("recon mean" if first else None), zorder=3)
        ax.plot(t, seg["orig"], color="#2b2f36", lw=0.7,
                label=("original" if first else None), zorder=4)
    ax.axvline(Ltr, color="#444", ls="--", lw=1.2)
    ax.text(Ltr, ax.get_ylim()[1], "  test →", va="top", ha="left", fontsize=10, color="#444")
    ax.text(Ltr, ax.get_ylim()[1], "← train  ", va="top", ha="right", fontsize=10, color="#444")
    ax.set_title(f"Full-series Stage-1 reconstruction — {label}\n"
                 f"CURRENT config: width_base=16 · token_dim=64 · window={r['W']}  |  "
                 f"{r['npar']/1000:.0f}k params · train nMSE={tr['nmse']:.3f} · test nMSE={te['nmse']:.3f}  "
                 f"(band = spread across the {r['W']} overlapping windows per timestep)",
                 fontsize=11, loc="left")
    ax.set_xlabel("time (normalized units; train then test)"); ax.set_ylabel("z-scored value")
    ax.grid(alpha=.15); ax.legend(fontsize=9, loc="upper right", framealpha=.92, ncol=2)
    ax.margins(x=0.005)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[recon-full] saved -> {out}  ({dpi} dpi, {figw:.0f}in wide)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--outdir", default="plots/recon_demo")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--figw", type=float, default=0.0, help="figure width in inches (0 = auto)")
    ap.add_argument("--only", default="", help="restrict to one slug (toy|ucr|wsd)")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.outdir, exist_ok=True)

    for ds, ent, slug, label, col in PAIRS:
        if args.only and slug != args.only:
            continue
        cache = f"{args.outdir}/cache_full_{slug}.npz"
        if os.path.exists(cache) and not args.retrain:
            print(f"\n===== {ds} / {ent} (cache) =====", flush=True)
            z = np.load(cache, allow_pickle=True); r = z["r"].item()
        else:
            print(f"\n===== {ds} / {ent} (train {args.steps} steps) =====", flush=True)
            r = compute_one(ds, ent, args.steps, args.lr, args.seed, device)
            np.savez_compressed(cache, r=r)
        plot_one(r, label, col, args.dpi, f"{args.outdir}/recon_full_{slug}.png", figw=args.figw)


if __name__ == "__main__":
    main()
