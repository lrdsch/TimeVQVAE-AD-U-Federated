#!/usr/bin/env python3
"""Autopsy of a SINGLE 'local' stage-1 reconstruction — no federation involved.

Question: forget federation. Take one standalone TimeVQVAE-AD (own encoder + own
codebook + own decoder, trained only on its own series). Why is the reconstruction
so smoothed / poor?

We decompose the reconstruction to attribute the smoothing to its causes:

  original                     x
  full (VQ + refinement)       encoder -> VQ -> decoder -> iSTFT -> refine   (what we plot elsewhere)
  no-VQ (autoencoder only)     encoder ->      decoder -> iSTFT -> refine    (continuous latent, quantizer bypassed)
  no-VQ, no-refine             encoder ->      decoder -> iSTFT             (pure conv autoencoder)
  best moving-average          uniform box filter of x, k chosen to minimise nMSE

Reads: nMSE of each variant (macro over windows AND on the single highest-std window
that the recon plots draw); VQ perplexity / active codes; latent grid size (the
bottleneck); spectral energy retained by band; how close the model recon is to a
plain moving average. Overlays the anomaly-labelled region so we can see whether the
"bad" reconstruction is just the model correctly failing to reconstruct an anomaly
(the intended reconstruction-based-AD signal) vs a genuine capacity/training problem.

    CUDA_VISIBLE_DEVICES=1 python scripts/local_recon_autopsy.py --dataset wsd_fed --cluster c0 --seed 0
"""
from __future__ import annotations
import argparse, dataclasses, json, os, sys, warnings
import numpy as np, torch
warnings.filterwarnings("ignore")
sys.path.insert(0, "pipeline"); sys.path.insert(0, ".")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _fedpaths import conv_seed_root            # noqa: E402
from config import Config                       # noqa: E402
from stage1 import load_stage1                  # noqa: E402
from data import make_dataloaders               # noqa: E402
import matplotlib; matplotlib.use("Agg")        # noqa: E402
import matplotlib.pyplot as plt                 # noqa: E402


def _overlay(obj, d):
    for k, v in d.items():
        if not hasattr(obj, k):
            continue
        cur = getattr(obj, k)
        if dataclasses.is_dataclass(cur) and isinstance(v, dict):
            _overlay(cur, v)
        else:
            setattr(obj, k, v)


def load_cfg(ckpt):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = Config(); _overlay(cfg, ck["cfg_dict"]); return cfg


def entities_of(dataset, cluster):
    cl = json.load(open(f"data/raw/{dataset}/clusters.json"))
    return list(cl[cluster])


def get_windows(cfg, ent, device, split="test", max_windows=400):
    c = Config(); _overlay(c, dataclasses.asdict(cfg)); c.dataset.entity_id = ent
    dl = make_dataloaders(c, stage="stage1")
    loader = {"test": dl.test_loader, "val": dl.val_loader, "train": dl.train_loader}[split]
    xs, ys = [], []
    for b in loader:
        xs.append(b["inputs"]); ys.append(b["labels"])
        if sum(x.shape[0] for x in xs) >= max_windows:
            break
    X = torch.cat(xs, 0)[:max_windows]           # (N, C, W) normalized
    Y = torch.cat(ys, 0)[:max_windows]           # (N, W) int  (-1 == unlabelled)
    return X.to(device), Y


def nmse(r, x):
    return float(((r - x) ** 2).sum() / (x ** 2).sum().clamp_min(1e-12))


@torch.no_grad()
def variants(s1, X, bs=64):
    """Return dict of reconstructions: full (VQ+refine), novq (AE+refine),
    novq_noref (pure AE), vq_noref (VQ, no refine). Also VQ perplexity + latent grid."""
    outs = {k: [] for k in ("full", "novq", "novq_noref", "vq_noref")}
    ppls, active = [], set()
    latent_shape = None
    K = int(s1.quantizer._vq.codebook_size)
    for i in range(0, X.shape[0], bs):
        xb = X[i:i+bs]
        tf = s1.transform(xb)
        if hasattr(s1.quantizer, "set_groups") and getattr(s1.quantizer, "groups", None) is None:
            s1.quantizer.set_groups(tf.spec.original_channels)
        latent = s1.encoder(tf)                                  # (B, C*d, F, W')
        if latent_shape is None:
            latent_shape = tuple(int(s) for s in latent.shape)
        qout = s1.quantizer(latent)
        # ── VQ path ──
        rep_vq = s1.reconstruct_representation(qout.quantized, tf.tensor.shape)
        rec_vq = s1.transform.inverse(rep_vq, tf.spec)
        outs["vq_noref"].append(rec_vq.cpu())
        outs["full"].append(s1.refinement(rec_vq).cpu())
        # ── no-VQ path (bypass quantizer: feed continuous latent to decoder) ──
        rep_nv = s1.reconstruct_representation(latent, tf.tensor.shape)
        rec_nv = s1.transform.inverse(rep_nv, tf.spec)
        outs["novq_noref"].append(rec_nv.cpu())
        outs["novq"].append(s1.refinement(rec_nv).cpu())
        # diagnostics
        if qout.perplexity is not None:
            ppls.append(float(qout.perplexity))
        idx = qout.indices.reshape(-1)
        active.update(int(v) for v in torch.unique(idx).tolist())
    return ({k: torch.cat(v, 0) for k, v in outs.items()},
            dict(perplexity=float(np.mean(ppls)) if ppls else float("nan"),
                 active=len([a for a in active if 0 <= a < K]), K=K, latent_shape=latent_shape))


def moving_avg(X, k):
    """Centered uniform box filter along time, reflect-padded. X: (N, C, W) tensor."""
    if k <= 1:
        return X.clone()
    pad = k // 2
    ker = torch.ones(1, 1, k, device=X.device) / k
    N, C, W = X.shape
    xp = torch.nn.functional.pad(X.reshape(N * C, 1, W), (pad, pad), mode="reflect")
    y = torch.nn.functional.conv1d(xp, ker)
    return y[..., :W].reshape(N, C, W)


def spectral_retained(x, r):
    """Fraction of x's power spectrum retained by r, split low/mid/high thirds.
    x, r: (N, W) numpy (channel 0)."""
    Fx = np.abs(np.fft.rfft(x, axis=-1)) ** 2
    Fr = np.abs(np.fft.rfft(r, axis=-1)) ** 2
    Px, Pr = Fx.mean(0), Fr.mean(0)                     # (W//2+1,)
    n = len(Px); a, b = n // 3, 2 * n // 3
    def band(lo, hi):
        return float(Pr[lo:hi].sum() / max(Px[lo:hi].sum(), 1e-12))
    return dict(low=band(0, a), mid=band(a, b), high=band(b, n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="wsd_fed")
    ap.add_argument("--cluster", default="c0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="test")
    ap.add_argument("--entities", default="", help="comma list; default = all in cluster")
    ap.add_argument("--max-plot", type=int, default=4)
    ap.add_argument("--outdir", default="plots/codebook_analysis")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds, cl, seed = args.dataset, args.cluster, args.seed
    ents = args.entities.split(",") if args.entities else entities_of(ds, cl)
    root = conv_seed_root(ds, cl, seed)
    os.makedirs(args.outdir, exist_ok=True)
    ks = [2, 4, 8, 16, 32]
    per, plot_rows = {}, []
    print(f"[autopsy] {ds}/{cl} seed{seed} — local arm only — {len(ents)} clients\n")

    for ent in ents:
        d_loc = f"{root}/local/{ent}/stage1.ckpt"
        if not os.path.exists(d_loc):
            print(f"  skip {ent} (missing {d_loc})"); continue
        cfg = load_cfg(d_loc); W = int(cfg.dataset.window_length)
        ex = torch.zeros(1, 1, W)
        s1 = load_stage1(d_loc, cfg, ex, device=device)
        X, Ylab = get_windows(cfg, ent, device, split=args.split)
        recs, diag = variants(s1, X)
        Xc = X.cpu()
        nm = {k: nmse(recs[k], Xc) for k in recs}
        # best moving average (macro over all windows)
        ma_nm = {k: nmse(moving_avg(Xc, k), Xc) for k in ks}
        k_best = min(ma_nm, key=ma_nm.get)
        rec_ma = moving_avg(Xc, k_best)
        nm["ma_best"] = ma_nm[k_best]
        # how close is the model's full recon to the best moving average?
        r_full = recs["full"].reshape(len(Xc), -1).numpy()
        r_ma = rec_ma.reshape(len(Xc), -1).numpy()
        corr = float(np.mean([np.corrcoef(a, b)[0, 1] for a, b in zip(r_full, r_ma)
                              if a.std() > 1e-8 and b.std() > 1e-8]))
        # spectral energy retained (channel 0)
        sp_full = spectral_retained(Xc[:, 0].numpy(), recs["full"][:, 0].numpy())
        sp_novq = spectral_retained(Xc[:, 0].numpy(), recs["novq"][:, 0].numpy())
        # highest-std window (the one the recon plots draw) + its anomaly fraction
        j = int(Xc.reshape(len(Xc), -1).std(1).argmax().item())
        yj = Ylab[j].numpy()
        anom_frac = float((yj > 0).mean()) if (yj >= 0).any() else float("nan")
        nm_win = {k: nmse(recs[k][j:j+1], Xc[j:j+1]) for k in recs}
        nm_win["ma_best"] = nmse(rec_ma[j:j+1], Xc[j:j+1])

        per[ent] = dict(nm=nm, nm_win=nm_win, k_best=k_best, corr_full_ma=corr,
                        sp_full=sp_full, sp_novq=sp_novq, anom_frac=anom_frac, **diag)
        print(f"  {ent}: latent={diag['latent_shape']} (W={W} -> {diag['latent_shape'][-1]} latent pos)"
              f"  ppl={diag['perplexity']:.1f} active={diag['active']}/{diag['K']}")
        print(f"     nMSE  full(VQ+ref)={nm['full']:.3f}  novq(AE)={nm['novq']:.3f} "
              f" novq_noref={nm['novq_noref']:.3f}  vq_noref={nm['vq_noref']:.3f}  MA(k={k_best})={nm['ma_best']:.3f}")
        print(f"     full-vs-MA corr={corr:.3f}  |  hi-freq energy kept: full={sp_full['high']:.2f} novq={sp_novq['high']:.2f}"
              f"  |  plotted-window anom_frac={anom_frac:.2f}\n")

        if len(plot_rows) < args.max_plot:
            plot_rows.append((ent, Xc[j, 0].numpy(), recs["full"][j, 0].numpy(),
                              recs["novq"][j, 0].numpy(), rec_ma[j, 0].numpy(),
                              yj, nm_win, k_best))

    # ── macro summary ──
    def mac(sel):
        return {k: float(np.mean([per[e]["nm"][k] for e in per])) for k in sel}
    macro = mac(["full", "novq", "novq_noref", "vq_noref", "ma_best"])
    print("=== MACRO nMSE over clients (local arm) ===")
    for k in ("full", "novq", "novq_noref", "vq_noref", "ma_best"):
        print(f"   {k:12s} {macro[k]:.4f}")
    print(f"   VQ penalty  (full - novq)            = {macro['full']-macro['novq']:+.4f}")
    print(f"   refine gain (novq_noref - novq)      = {macro['novq_noref']-macro['novq']:+.4f}")
    print(f"   AE floor    (novq vs best MA)        = novq {macro['novq']:.3f} vs MA {macro['ma_best']:.3f}")

    # ── FIG: decomposition overlays ──
    n = len(plot_rows)
    if n:
        fig, axes = plt.subplots(n, 1, figsize=(12, 2.6 * n))
        axes = np.atleast_1d(axes)
        for ax, (ent, x, rf, rv, rm, y, nmw, kb) in zip(axes, plot_rows):
            t = np.arange(len(x))
            # shade anomaly region
            if (y >= 0).any() and (y > 0).any():
                ax.fill_between(t, x.min(), x.max(), where=(y > 0), color="#f2c14e",
                                alpha=0.30, label="anomaly (labelled)", zorder=0)
            ax.plot(t, x, color="#2b2f36", lw=1.7, label="original", zorder=4)
            ax.plot(t, rv, color="#2b5f9e", lw=1.3, label=f"no-VQ (AE only)  nMSE={nmw['novq']:.3f}", zorder=3)
            ax.plot(t, rf, color="#b3323f", lw=1.3, ls="--", label=f"full (VQ+refine)  nMSE={nmw['full']:.3f}", zorder=3)
            ax.plot(t, rm, color="#1c7a4b", lw=1.1, ls=":", label=f"moving-avg k={kb}  nMSE={nmw['ma_best']:.3f}", zorder=2)
            ax.set_title(f"{ent}", fontsize=10, loc="left")
            ax.legend(fontsize=7.5, loc="upper right", framealpha=.92, ncol=2)
            ax.grid(alpha=.15)
        fig.suptitle(f"Why is the LOCAL reconstruction smoothed? decomposition — {ds}/{cl} seed{seed}\n"
                     f"latent bottleneck {per[plot_rows[0][0]]['latent_shape']} · "
                     f"cb={per[plot_rows[0][0]]['K']} · d={load_cfg(f'{root}/local/{plot_rows[0][0]}/stage1.ckpt').quantizer.token_embedding_dim}",
                     fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        p1 = f"{args.outdir}/autopsy_{ds}_{cl}_s{seed}.png"; fig.savefig(p1, dpi=135); plt.close(fig)
        print(f"\n[plot] {p1}")

    json.dump({e: {k: v for k, v in per[e].items() if k != "latent_shape"} | {"latent_shape": list(per[e]["latent_shape"])}
               for e in per} | {"_macro": macro},
              open(f"{args.outdir}/autopsy_{ds}_{cl}_s{seed}.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
