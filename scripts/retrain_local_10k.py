#!/usr/bin/env python3
"""Empirical test of the 'few epochs' hypothesis: retrain N random 'local' stage-1
models for a FIXED STEP budget (default 10000, no early stopping) — same exact
architecture/config as the converged 35-epoch checkpoints — and compare the
reconstruction, old vs new.

Faithful to federated_eval._train_stage1: fresh Stage1VQVAE, AdamW(lr) CONSTANT
(no schedule), same _amp()/_make_scaler() precision path, no early stopping. The
ONLY change vs converged is epoch-loop -> step-loop with n_steps steps.

    CUDA_VISIBLE_DEVICES=1 python scripts/retrain_local_10k.py --dataset wsd_fed --n-clients 4 --steps 10000
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np, torch
sys.path.insert(0, "pipeline"); sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from config import Config                                              # noqa: E402
from stage1 import Stage1VQVAE, save_stage1_checkpoint, load_stage1    # noqa: E402
from data import make_dataloaders                                      # noqa: E402
from federated import _amp, _make_scaler, _cycle                       # noqa: E402
from utils import seed_everything                                      # noqa: E402
import local_recon_autopsy as A                                        # helpers: load_cfg, get_windows, variants, nmse, moving_avg, spectral_retained
from _fedpaths import conv_root                                        # noqa: E402
import matplotlib; matplotlib.use("Agg")                               # noqa: E402
import matplotlib.pyplot as plt                                        # noqa: E402


def cluster_of(dataset):
    cl = json.load(open(f"data/raw/{dataset}/clusters.json"))
    return {e: c for c, ents in cl.items() for e in ents}


def train_steps(cfg, entity, n_steps, lr, device, seed=0):
    """Exactly federated_eval._train_stage1, but step-based for n_steps steps."""
    seed_everything(seed)
    c = Config(); A._overlay(c, __import__("dataclasses").asdict(cfg)); c.dataset.entity_id = entity
    dl = make_dataloaders(c, stage="stage1")
    loader = dl.train_loader
    ex = next(iter(loader))["inputs"][:1].cpu()
    model = Stage1VQVAE(c); model.eval()
    with torch.no_grad():
        model(ex)                                   # materialise lazy modules
    model.to(device)
    model.quantizer.collect_stats_only = False
    opt = torch.optim.AdamW(model.parameters(), lr=lr, fused=(torch.device(device).type == "cuda"))
    scaler = _make_scaler()
    model.train()
    it = _cycle(loader)
    tot, nb, trace = 0.0, 0, []
    for step in range(n_steps):
        x = next(it)["inputs"].to(device, non_blocking=True)
        with _amp():
            loss = model(x)["losses"]["loss"]
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        lv = float(loss.detach())
        if lv == lv:
            tot += lv; nb += 1
        if step < 2 or step >= n_steps - 2 or (step + 1) % 500 == 0:
            m = tot / max(nb, 1); trace.append((step + 1, m))
            print(f"    [s1 {entity}] step {step+1}/{n_steps} loss={m:.4f}", flush=True)
            tot, nb = 0.0, 0                        # windowed mean between prints
    return model, c, ex, trace


def full_nmse(s1, X):
    recs, _ = A.variants(s1, X)
    return A.nmse(recs["full"], X.cpu()), recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="wsd_fed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-clients", type=int, default=4)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pick-seed", type=int, default=0, help="RNG seed for the random client pick")
    ap.add_argument("--clients", default="", help="explicit comma list overrides random pick")
    ap.add_argument("--outdir", default="plots/codebook_analysis")
    ap.add_argument("--ckptdir", default="artifacts/fed_eval/retrain10k")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds, seed = args.dataset, args.seed
    e2c = cluster_of(ds)
    conv_root_dir = conv_root(ds)

    # clients that actually have a converged local ckpt (so we can compare)
    have = [e for e, c in e2c.items()
            if os.path.exists(f"{conv_root_dir}/{c}/seed{seed}/local/{e}/stage1.ckpt")]
    if args.clients:
        picks = [x for x in args.clients.split(",") if x in have]
    else:
        rng = np.random.RandomState(args.pick_seed)
        picks = list(rng.choice(sorted(have), size=min(args.n_clients, len(have)), replace=False))
    print(f"[retrain] {ds} seed{seed} — {args.steps} steps, no early-stop, lr={args.lr}")
    print(f"[retrain] random clients ({args.pick_seed=}): {picks}\n")

    os.makedirs(args.outdir, exist_ok=True)
    rows, plot_rows = [], []
    for ent in picks:
        cl = e2c[ent]
        old_ckpt = f"{conv_root_dir}/{cl}/seed{seed}/local/{ent}/stage1.ckpt"
        cfg = A.load_cfg(old_ckpt)
        print(f"── {ent} (cluster {cl}) ──────────────────────────────")
        # NEW: retrain from scratch for args.steps steps
        new_model, c, ex, trace = train_steps(cfg, ent, args.steps, args.lr, device, seed=seed)
        new_model.eval()
        ck_path = f"{args.ckptdir}/{ds}/{cl}/seed{seed}/local/{ent}/stage1.ckpt"
        save_stage1_checkpoint(__import__("pathlib").Path(ck_path), new_model, c,
                               step=args.steps, epoch=-1)
        # OLD: the converged 35-epoch model
        old_model = load_stage1(old_ckpt, cfg, ex, device=device)

        # eval on the SAME test windows
        X, Ylab = A.get_windows(cfg, ent, device, split="test")
        Xc = X.cpu()
        old_nm, old_recs = full_nmse(old_model, X)
        new_nm, new_recs = full_nmse(new_model, X)
        # diagnostics on the new model
        _, diag = A.variants(new_model, X)
        sp_old = A.spectral_retained(Xc[:, 0].numpy(), old_recs["full"][:, 0].numpy())
        sp_new = A.spectral_retained(Xc[:, 0].numpy(), new_recs["full"][:, 0].numpy())
        vq_pen_new = A.nmse(new_recs["full"], Xc) - A.nmse(new_recs["novq"], Xc)

        j = int(Xc.reshape(len(Xc), -1).std(1).argmax().item())
        yj = Ylab[j].numpy(); anom = float((yj > 0).mean()) if (yj >= 0).any() else float("nan")
        nm_old_w = A.nmse(old_recs["full"][j:j+1], Xc[j:j+1])
        nm_new_w = A.nmse(new_recs["full"][j:j+1], Xc[j:j+1])

        rows.append(dict(ent=ent, cl=cl, old=old_nm, new=new_nm, anom_frac=anom,
                         hi_old=sp_old["high"], hi_new=sp_new["high"],
                         ppl_new=diag["perplexity"], active_new=diag["active"], K=diag["K"],
                         final_loss=trace[-1][1] if trace else float("nan")))
        print(f"   test nMSE   OLD(35ep)={old_nm:.4f}   NEW({args.steps}step)={new_nm:.4f}   "
              f"Δ={new_nm-old_nm:+.4f}")
        print(f"   hi-freq kept  old={sp_old['high']:.3f} new={sp_new['high']:.3f}  |  "
              f"new ppl={diag['perplexity']:.1f} active={diag['active']}/{diag['K']}  |  VQ-pen(new)={vq_pen_new:+.4f}")
        print(f"   plotted window: anom_frac={anom:.2f}  nMSE old={nm_old_w:.3f} new={nm_new_w:.3f}\n")

        plot_rows.append((ent, cl, Xc[j, 0].numpy(), old_recs["full"][j, 0].numpy(),
                          new_recs["full"][j, 0].numpy(), yj, nm_old_w, nm_new_w, anom))

    # ── summary table ──
    print("=" * 74)
    print(f"{'client':16s} {'clu':4s} {'old(35ep)':>10s} {'new(10k)':>10s} {'Δ':>9s} {'hi_old':>7s} {'hi_new':>7s} {'anom%':>6s}")
    for r in rows:
        print(f"{r['ent']:16s} {r['cl']:4s} {r['old']:10.4f} {r['new']:10.4f} "
              f"{r['new']-r['old']:+9.4f} {r['hi_old']:7.2f} {r['hi_new']:7.2f} {100*r['anom_frac']:6.1f}")
    mo = float(np.mean([r['old'] for r in rows])); mn = float(np.mean([r['new'] for r in rows]))
    print("-" * 74)
    print(f"{'MACRO':16s} {'':4s} {mo:10.4f} {mn:10.4f} {mn-mo:+9.4f}")
    print("=" * 74)

    # ── FIG: old vs new reconstruction overlay ──
    n = len(plot_rows)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.6 * n)); axes = np.atleast_1d(axes)
    for ax, (ent, cl, x, ro, rn, y, nmo, nmn, af) in zip(axes, plot_rows):
        t = np.arange(len(x))
        if (y >= 0).any() and (y > 0).any():
            ax.fill_between(t, x.min(), x.max(), where=(y > 0), color="#f2c14e", alpha=0.30,
                            label="anomaly", zorder=0)
        ax.plot(t, x, color="#2b2f36", lw=1.7, label="original", zorder=4)
        ax.plot(t, ro, color="#b3323f", lw=1.3, ls="--", label=f"OLD 35ep  nMSE={nmo:.3f}", zorder=3)
        ax.plot(t, rn, color="#1c7a4b", lw=1.4, label=f"NEW {args.steps}step  nMSE={nmn:.3f}", zorder=3)
        ax.set_title(f"{ent} (cluster {cl})", fontsize=10, loc="left")
        ax.legend(fontsize=8, loc="upper right", framealpha=.92, ncol=2); ax.grid(alpha=.15)
    fig.suptitle(f"Does 10k-step retraining sharpen the LOCAL reconstruction? — {ds} seed{seed}\n"
                 f"(no early stopping · same architecture · macro nMSE {mo:.3f} -> {mn:.3f})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = f"{args.outdir}/retrain10k_{ds}_s{seed}.png"; fig.savefig(p, dpi=135); plt.close(fig)
    json.dump(rows, open(f"{args.outdir}/retrain10k_{ds}_s{seed}.json", "w"), indent=1, default=float)
    print(f"\n[plot] {p}")


if __name__ == "__main__":
    main()
