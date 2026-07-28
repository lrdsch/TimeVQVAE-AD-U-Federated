#!/usr/bin/env python3
"""Old (35ep) vs New (10k-step) reconstruction, side by side on TRAIN and TEST.

Reads the converged local checkpoints and the retrain10k checkpoints (no training).
For each retrained client it draws a 2-column row: [train window | test window],
each overlaying original / old / new, and prints macro nMSE per split. The
overfitting signature = new BETTER on train, WORSE on test.

    python scripts/plot_retrain_train_test.py --dataset wsd_fed --seed 0
"""
from __future__ import annotations
import argparse, glob, os, sys, json
import numpy as np, torch
sys.path.insert(0, "pipeline"); sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import torch.nn as nn                                                  # noqa: E402
from stage1 import Stage1VQVAE                                          # noqa: E402
import local_recon_autopsy as A                                        # load_cfg, get_windows, variants, nmse
from _fedpaths import conv_root as _conv_root                          # noqa: E402
import matplotlib; matplotlib.use("Agg")                               # noqa: E402
import matplotlib.pyplot as plt                                        # noqa: E402

_BN = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def load_stage1(ckpt_path, cfg, example_inputs, device="cpu"):
    """Like stage1.load_stage1 but weights_only=False (trusted local ckpts whose
    cfg_dict carries numpy scalars that trip the weights_only=True default)."""
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model = Stage1VQVAE(cfg); model.eval()
    with torch.no_grad():
        model(example_inputs.cpu())
    model.load_state_dict(state["state_dict"], strict=True)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def recalibrate_bn(m, Xtr, bs=128, passes=5):
    """Reset BN running stats and re-estimate on train windows (cumulative avg) so
    eval-mode reflects the learned representation, not stale/miscalibrated BN stats."""
    saved = {}
    for mod in m.modules():
        if isinstance(mod, _BN):
            saved[mod] = mod.momentum
            mod.reset_running_stats(); mod.momentum = None
    m.train()
    for _ in range(passes):
        for i in range(0, len(Xtr), bs):
            m(Xtr[i:i+bs])
    for mod, mom in saved.items():
        mod.momentum = mom
    m.eval()


def retrained_clients(ds, seed, ckptdir):
    out = []
    for p in sorted(glob.glob(f"{ckptdir}/{ds}/*/seed{seed}/local/*/stage1.ckpt")):
        parts = p.split("/")
        cl, ent = parts[-5], parts[-2]
        out.append((cl, ent, p))
    return out


def full_recon(s1, X):
    recs, _ = A.variants(s1, X)
    return recs["full"], A.nmse(recs["full"], X.cpu())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="wsd_fed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckptdir", default="artifacts/fed_eval/retrain10k")
    ap.add_argument("--outdir", default="plots/codebook_analysis")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds, seed = args.dataset, args.seed
    conv_root = _conv_root(ds)
    clients = retrained_clients(ds, seed, args.ckptdir)
    if not clients:
        print(f"[plot] no retrain10k checkpoints for {ds} seed{seed} in {args.ckptdir}"); return
    os.makedirs(args.outdir, exist_ok=True)
    print(f"[plot] {ds} seed{seed} — {len(clients)} retrained clients")

    rows, plot_rows = [], []
    for cl, ent, new_ckpt in clients:
        old_ckpt = f"{conv_root}/{cl}/seed{seed}/local/{ent}/stage1.ckpt"
        if not os.path.exists(old_ckpt):
            print(f"  skip {ent}: no converged ckpt"); continue
        cfg = A.load_cfg(old_ckpt)
        ex = torch.zeros(1, 1, int(cfg.dataset.window_length))
        old = load_stage1(old_ckpt, cfg, ex, device=device)
        new = load_stage1(new_ckpt, A.load_cfg(new_ckpt), ex, device=device)

        # BN-recalibrate BOTH on the train windows so eval-mode reconstructions
        # reflect the learned representation (not stale/miscalibrated BN stats).
        Xtr_recal, _ = A.get_windows(cfg, ent, device, split="train")
        recalibrate_bn(old, Xtr_recal); recalibrate_bn(new, Xtr_recal)

        cell = {}
        for split in ("train", "test"):
            X, Ylab = A.get_windows(cfg, ent, device, split=split)
            Xc = X.cpu()
            ro, nmo = full_recon(old, X)
            rn, nmn = full_recon(new, X)
            j = int(Xc.reshape(len(Xc), -1).std(1).argmax().item())
            yj = Ylab[j].numpy()
            cell[split] = dict(nmo=nmo, nmn=nmn,
                               x=Xc[j, 0].numpy(), ro=ro[j, 0].numpy(), rn=rn[j, 0].numpy(),
                               y=yj, wnmo=A.nmse(ro[j:j+1], Xc[j:j+1]), wnmn=A.nmse(rn[j:j+1], Xc[j:j+1]))
        rows.append(dict(ent=ent, cl=cl,
                         tr_old=cell["train"]["nmo"], tr_new=cell["train"]["nmn"],
                         te_old=cell["test"]["nmo"], te_new=cell["test"]["nmn"]))
        plot_rows.append((ent, cl, cell))
        print(f"  {ent:10s}({cl}): TRAIN old={cell['train']['nmo']:.3f} new={cell['train']['nmn']:.3f} "
              f"(Δ{cell['train']['nmn']-cell['train']['nmo']:+.3f})  |  "
              f"TEST old={cell['test']['nmo']:.3f} new={cell['test']['nmn']:.3f} "
              f"(Δ{cell['test']['nmn']-cell['test']['nmo']:+.3f})")

    # ── summary ──
    def mac(k): return float(np.mean([r[k] for r in rows]))
    print("\n=== MACRO nMSE (overfitting = train↓ but test↑) ===")
    print(f"  TRAIN  old {mac('tr_old'):.4f} -> new {mac('tr_new'):.4f}  (Δ {mac('tr_new')-mac('tr_old'):+.4f})")
    print(f"  TEST   old {mac('te_old'):.4f} -> new {mac('te_new'):.4f}  (Δ {mac('te_new')-mac('te_old'):+.4f})")

    # ── FIG: rows=clients, cols=[train,test] ──
    n = len(plot_rows)
    fig, axes = plt.subplots(n, 2, figsize=(15, 2.5 * n), squeeze=False)
    for i, (ent, cl, cell) in enumerate(plot_rows):
        for j, split in enumerate(("train", "test")):
            ax = axes[i][j]; c = cell[split]
            t = np.arange(len(c["x"]))
            y = c["y"]
            if split == "test" and (y >= 0).any() and (y > 0).any():
                ax.fill_between(t, c["x"].min(), c["x"].max(), where=(y > 0),
                                color="#f2c14e", alpha=0.30, label="anomaly", zorder=0)
            ax.plot(t, c["x"], color="#2b2f36", lw=1.6, label="original", zorder=4)
            ax.plot(t, c["ro"], color="#b3323f", lw=1.2, ls="--", label=f"OLD 35ep  nMSE={c['wnmo']:.3f}", zorder=3)
            ax.plot(t, c["rn"], color="#1c7a4b", lw=1.3, label=f"NEW 10k  nMSE={c['wnmn']:.3f}", zorder=3)
            ax.set_title(f"{ent} ({cl}) — {split.upper()}  [macro nMSE old={c['nmo']:.3f} new={c['nmn']:.3f}]",
                         fontsize=9, loc="left")
            ax.legend(fontsize=7, loc="upper right", framealpha=.9, ncol=1); ax.grid(alpha=.15)
    verdict = ("more training HELPS — sharper recon (signal learnable)"
               if mac('te_new') < mac('te_old') else
               "more training HURTS — eval-mode degradation (scarce/heterogeneous data)")
    fig.suptitle(f"OLD (35ep) vs NEW (10k-step) reconstruction [BN-recalibrated, eval-mode] — TRAIN vs TEST — {ds} seed{seed}\n"
                 f"macro nMSE  TRAIN {mac('tr_old'):.3f}->{mac('tr_new'):.3f}   "
                 f"TEST {mac('te_old'):.3f}->{mac('te_new'):.3f}   →  {verdict}",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = f"{args.outdir}/retrain10k_traintest_{ds}_s{seed}.png"; fig.savefig(p, dpi=130); plt.close(fig)
    json.dump(rows, open(f"{args.outdir}/retrain10k_traintest_{ds}_s{seed}.json", "w"), indent=1, default=float)
    print(f"\n[plot] {p}")


if __name__ == "__main__":
    main()
