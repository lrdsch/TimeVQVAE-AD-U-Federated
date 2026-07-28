#!/usr/bin/env python3
"""Stage-1 reconstruction demo on 3 datasets (toy / ucr / wsd), CURRENT config.

Trains a fresh Stage-1 VQ-VAE tokenizer (as configured *now* — wb16 / td64 /
window 128) on ONE client from each of three datasets, then saves ONE
high-resolution figure PER DATASET (original-vs-reconstruction on the
highest-variance TEST windows, legend on each). BN-recalibrated eval so the
nMSE is not BatchNorm-contaminated.

Reconstruction arrays are cached to <outdir>/cache_<slug>.npz — a re-run reuses
them (no retrain) so plot styling can be iterated cheaply. Use --retrain to force.

    CUDA_VISIBLE_DEVICES=1 python scripts/recon_demo_3ds.py --steps 3000 --dpi 600
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np, torch
sys.path.insert(0, "pipeline"); sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from config import load_config                                        # noqa: E402
import local_recon_autopsy as A                                       # get_windows/variants/nmse  # noqa: E402
from retrain_capacity import train_steps, recalibrate_bn             # noqa: E402
import matplotlib; matplotlib.use("Agg")                              # noqa: E402
import matplotlib.pyplot as plt                                       # noqa: E402

# (dataset, entity, slug, pretty label, colour)
PAIRS = [
    ("toy_fed_uni",  "uni_00",  "toy",  "toy  (toy_fed_uni)",      "#1c7a4b"),
    ("ucr_pool",     "ucr_001", "ucr",  "ucr  (ucr_pool, real)",   "#d98a00"),
    ("wsd_fed",      "kpi_015", "wsd",  "wsd  (wsd_fed, real)",    "#b3323f"),
]


def cfg_for(ds, ent):
    os.environ["DATASET_NAME"] = ds
    os.environ["DATASET_ENTITY"] = ent
    return load_config()


def top_nonoverlap(var, W, k=3, min_sep=None):
    """Indices of the k highest-variance windows kept `min_sep` apart (default W//8)."""
    if min_sep is None:
        min_sep = max(1, W // 8)
    order = torch.argsort(var, descending=True).tolist()
    picks = []
    for idx in order:
        if all(abs(idx - p) >= min_sep for p in picks):
            picks.append(idx)
        if len(picks) == k:
            break
    return picks


def compute_one(ds, ent, steps, lr, seed, device):
    cfg = cfg_for(ds, ent)
    W = cfg.dataset.window_length
    m, c, _, npar = train_steps(cfg, ent, steps, lr, device, seed=seed)
    Xtr, _ = A.get_windows(cfg, ent, device, split="train")
    Xte, Yte = A.get_windows(cfg, ent, device, split="test")
    recalibrate_bn(m, Xtr)
    recs, diag = A.variants(m, Xte)
    return dict(W=W, npar=int(npar), macro=A.nmse(recs["full"], Xte.cpu()),
                ppl=float(diag["perplexity"]), active=int(diag["active"]), K=int(diag["K"]),
                Xc=Xte.cpu().numpy(), rec=recs["full"].numpy(), Yte=Yte.cpu().numpy())


def plot_one(r, slug, label, col, ncols, dpi, out):
    Xc, rec, Yte, W = r["Xc"], r["rec"], r["Yte"], r["W"]
    var = torch.from_numpy(Xc).reshape(len(Xc), -1).std(1)
    picks = top_nonoverlap(var, W, k=ncols)
    fig, axes = plt.subplots(1, len(picks), figsize=(6.0 * len(picks), 3.4), squeeze=False)
    for j, k in enumerate(picks):
        ax = axes[0][j]
        x = Xc[k, 0]; rc = rec[k, 0]; y = Yte[k]; t = np.arange(len(x))
        if (y > 0).any():
            ax.fill_between(t, x.min(), x.max(), where=(y > 0),
                            color="#f2c14e", alpha=0.35, label="anomaly", zorder=0)
        ax.plot(t, x, color="#2b2f36", lw=1.7, label="original", zorder=4)
        ax.plot(t, rc, color=col, lw=1.6, label="reconstruction", zorder=3)
        wnm = float(((rec[k:k+1] - Xc[k:k+1]) ** 2).sum() / max((Xc[k:k+1] ** 2).sum(), 1e-12))
        ax.set_title(f"test win #{k}   nMSE={wnm:.3f}", fontsize=11, loc="left")
        ax.grid(alpha=.18); ax.tick_params(labelsize=9)
        ax.legend(fontsize=10, loc="upper right", framealpha=.92)   # legend on EVERY panel
    fig.suptitle(f"Stage-1 reconstruction — {label}\n"
                 f"CURRENT config: width_base=16 · token_dim=64 · window={W}  |  "
                 f"{r['npar']/1000:.0f}k params · test nMSE={r['macro']:.3f} · "
                 f"ppl={r['ppl']:.1f} · codes {r['active']}/{r['K']}",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[recon-demo] saved -> {out}  ({dpi} dpi)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ncols", type=int, default=3)
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--outdir", default="plots/recon_demo")
    ap.add_argument("--retrain", action="store_true", help="ignore cache and retrain")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.outdir, exist_ok=True)

    for ds, ent, slug, label, col in PAIRS:
        cache = f"{args.outdir}/cache_{slug}.npz"
        if os.path.exists(cache) and not args.retrain:
            print(f"\n===== {ds} / {ent} (cache) =====", flush=True)
            z = np.load(cache)
            r = dict(W=int(z["W"]), npar=int(z["npar"]), macro=float(z["macro"]),
                     ppl=float(z["ppl"]), active=int(z["active"]), K=int(z["K"]),
                     Xc=z["Xc"], rec=z["rec"], Yte=z["Yte"])
        else:
            print(f"\n===== {ds} / {ent} (train {args.steps} steps) =====", flush=True)
            r = compute_one(ds, ent, args.steps, args.lr, args.seed, device)
            np.savez_compressed(cache, W=r["W"], npar=r["npar"], macro=r["macro"],
                                ppl=r["ppl"], active=r["active"], K=r["K"],
                                Xc=r["Xc"], rec=r["rec"], Yte=r["Yte"])
        print(f"  window={r['W']}  params={r['npar']/1000:.0f}k  test nMSE={r['macro']:.3f} "
              f"ppl={r['ppl']:.1f} active={r['active']}/{r['K']}", flush=True)
        plot_one(r, slug, label, col, args.ncols, args.dpi, f"{args.outdir}/recon_{slug}.png")


if __name__ == "__main__":
    main()
