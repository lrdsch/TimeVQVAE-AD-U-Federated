#!/usr/bin/env python3
"""Test the 'latent width 4 vs 64' hypothesis (LOCAL_VS_UPSTREAM #1) on Stage 1 only.

Trains fresh Stage-1 tokenizers at three capacities, SAME budget/clients, then
compares reconstruction (BN-recalibrated, eval-mode) on train and test:

  d4     : token_embedding_dim=4,  width_base=4   (current fork — body 4/8/16, latent 4)
  d64    : token_embedding_dim=64, width_base=4   (widen ONLY latent/codebook the decoder sees)
  wide64 : token_embedding_dim=64, width_base=16  (widen the conv BODY too → 16/32/64 = upstream-like)

No Stage-2 / prior — reconstruction is pure Stage 1.

    CUDA_VISIBLE_DEVICES=1 python scripts/retrain_capacity.py --dataset wsd_fed --n-clients 3 --steps 3000
"""
from __future__ import annotations
import argparse, dataclasses, glob, json, os, sys
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, "pipeline"); sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from config import Config                                              # noqa: E402
from stage1 import Stage1VQVAE, save_stage1_checkpoint                 # noqa: E402
from data import make_dataloaders                                      # noqa: E402
from federated import _amp, _make_scaler, _cycle                       # noqa: E402
from utils import seed_everything                                      # noqa: E402
import local_recon_autopsy as A                                        # load_cfg, get_windows, variants, nmse, spectral_retained
from _fedpaths import conv_seed_root                                   # noqa: E402
import matplotlib; matplotlib.use("Agg")                               # noqa: E402
import matplotlib.pyplot as plt                                        # noqa: E402

_BN = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
ARMS = {  # name -> (token_embedding_dim, width_base)
    "d4":     (4, 4),
    "d64":    (64, 4),
    "wide64": (64, 16),
}
STYLE = {"d4": ("#b3323f", "--", 1.2), "d64": ("#d98a00", "-", 1.2), "wide64": ("#1c7a4b", "-", 1.4)}


def cfg_for(base_cfg, token_dim, width_base):
    c = Config(); A._overlay(c, dataclasses.asdict(base_cfg))
    c.quantizer.token_embedding_dim = int(token_dim)
    c.encoder.width_base = int(width_base)
    return c


def train_steps(cfg, entity, n_steps, lr, device, seed=0):
    seed_everything(seed)
    c = Config(); A._overlay(c, dataclasses.asdict(cfg)); c.dataset.entity_id = entity
    loader = make_dataloaders(c, stage="stage1").train_loader
    ex = next(iter(loader))["inputs"][:1].cpu()
    m = Stage1VQVAE(c); m.eval()
    with torch.no_grad():
        m(ex)
    m.to(device); m.quantizer.collect_stats_only = False
    n_params = sum(p.numel() for p in m.parameters())
    opt = torch.optim.AdamW(m.parameters(), lr=lr, fused=(torch.device(device).type == "cuda"))
    scaler = _make_scaler(); m.train(); it = _cycle(loader)
    tot = nb = 0
    for step in range(n_steps):
        x = next(it)["inputs"].to(device, non_blocking=True)
        with _amp():
            loss = m(x)["losses"]["loss"]
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        lv = float(loss.detach())
        if lv == lv:
            tot += lv; nb += 1
        if step < 1 or step >= n_steps - 1 or (step + 1) % 1000 == 0:
            print(f"      [{entity}] step {step+1}/{n_steps} loss={tot/max(nb,1):.4f}", flush=True); tot = nb = 0
    return m, c, ex, n_params


@torch.no_grad()
def recalibrate_bn(m, Xtr, bs=128, passes=5):
    saved = {}
    for mod in m.modules():
        if isinstance(mod, _BN):
            saved[mod] = mod.momentum; mod.reset_running_stats(); mod.momentum = None
    m.train()
    for _ in range(passes):
        for i in range(0, len(Xtr), bs):
            m(Xtr[i:i+bs])
    for mod, mom in saved.items():
        mod.momentum = mom
    m.eval()


def pick_clients(ds, seed, n, pick_seed=0):
    cl = json.load(open(f"data/raw/{ds}/clusters.json"))
    e2c = {e: c for c, ents in cl.items() for e in ents}
    have = [e for e, c in e2c.items()
            if os.path.exists(f"{conv_seed_root(ds, c, seed)}/local/{e}/stage1.ckpt")]
    picks = list(np.random.RandomState(pick_seed).choice(sorted(have), size=min(n, len(have)), replace=False))
    return [(e2c[e], e) for e in picks]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="wsd_fed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-clients", type=int, default=3)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--arms", default="d4,d64,wide64")
    ap.add_argument("--outdir", default="plots/codebook_analysis")
    ap.add_argument("--ckptdir", default="artifacts/fed_eval/capacity")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds, seed = args.dataset, args.seed
    arms = [a for a in args.arms.split(",") if a in ARMS]
    clients = pick_clients(ds, seed, args.n_clients)
    os.makedirs(args.outdir, exist_ok=True)
    print(f"[capacity] {ds} seed{seed} — {args.steps} steps — arms {arms} — clients {[e for _,e in clients]}\n")

    rows, plot_rows, params = [], [], {}
    for cl, ent in clients:
        base = A.load_cfg(f"{conv_seed_root(ds, cl, seed)}/local/{ent}/stage1.ckpt")
        print(f"── {ent} ({cl}) ──")
        Xtr, Ytr = A.get_windows(base, ent, device, split="train")
        Xte, Yte = A.get_windows(base, ent, device, split="test")
        cell = {"train": {"X": Xtr.cpu(), "Y": Ytr}, "test": {"X": Xte.cpu(), "Y": Yte}}
        r = {"ent": ent, "cl": cl}
        for arm in arms:
            td, wb = ARMS[arm]
            print(f"   [{arm}] token_dim={td} width_base={wb}")
            m, c, ex, np_ = train_steps(cfg_for(base, td, wb), ent, args.steps, args.lr, device, seed=seed)
            params[arm] = np_
            ckp = f"{args.ckptdir}/{ds}/{arm}/{cl}/seed{seed}/local/{ent}/stage1.ckpt"
            save_stage1_checkpoint(__import__("pathlib").Path(ckp), m, c, step=args.steps, epoch=-1)
            recalibrate_bn(m, Xtr)
            recs_tr, _ = A.variants(m, Xtr); recs_te, diag = A.variants(m, Xte)
            r[f"{arm}_tr"] = A.nmse(recs_tr["full"], Xtr.cpu())
            r[f"{arm}_te"] = A.nmse(recs_te["full"], Xte.cpu())
            r[f"{arm}_hi"] = A.spectral_retained(Xte.cpu()[:, 0].numpy(), recs_te["full"][:, 0].numpy())["high"]
            r[f"{arm}_ppl"] = diag["perplexity"]
            for split, recs in (("train", recs_tr), ("test", recs_te)):
                cell[split].setdefault("rec", {})[arm] = recs["full"]
            print(f"      -> nMSE train={r[f'{arm}_tr']:.3f} test={r[f'{arm}_te']:.3f} "
                  f"hi-freq={r[f'{arm}_hi']:.3f} ppl={diag['perplexity']:.1f} params={np_/1000:.0f}k")
        rows.append(r); plot_rows.append((ent, cl, cell))
        print()

    # ── table ──
    print("=" * 78)
    hdr = f"{'client':12s}{'cl':4s}" + "".join(f"{a+'_tr':>9s}{a+'_te':>9s}" for a in arms)
    print(hdr)
    for r in rows:
        print(f"{r['ent']:12s}{r['cl']:4s}" + "".join(f"{r[a+'_tr']:9.3f}{r[a+'_te']:9.3f}" for a in arms))
    print("-" * 78)
    macro = {a: {s: float(np.mean([r[f'{a}_{s}'] for r in rows])) for s in ("tr", "te")} for a in arms}
    hi = {a: float(np.mean([r[f'{a}_hi'] for r in rows])) for a in arms}
    print(f"{'MACRO':12s}{'':4s}" + "".join(f"{macro[a]['tr']:9.3f}{macro[a]['te']:9.3f}" for a in arms))
    print("=" * 78)
    for a in arms:
        print(f"  {a:7s} params={params[a]/1000:.0f}k  macro TEST nMSE={macro[a]['te']:.3f}  hi-freq kept={hi[a]:.3f}")

    # ── FIG: rows=clients, cols=[train,test], overlay arms ──
    n = len(plot_rows)
    fig, axes = plt.subplots(n, 2, figsize=(15, 2.6 * n), squeeze=False)
    for i, (ent, cl, cell) in enumerate(plot_rows):
        for j, split in enumerate(("train", "test")):
            ax = axes[i][j]; C = cell[split]
            X = C["X"]; Y = C["Y"]
            k = int(X.reshape(len(X), -1).std(1).argmax().item())
            x = X[k, 0].numpy(); y = Y[k].numpy(); t = np.arange(len(x))
            if split == "test" and (y >= 0).any() and (y > 0).any():
                ax.fill_between(t, x.min(), x.max(), where=(y > 0), color="#f2c14e", alpha=0.30, label="anomaly", zorder=0)
            ax.plot(t, x, color="#2b2f36", lw=1.7, label="original", zorder=5)
            for arm in arms:
                col, ls, lw = STYLE[arm]
                wnm = A.nmse(C["rec"][arm][k:k+1], X[k:k+1])
                ax.plot(t, C["rec"][arm][k, 0].numpy(), color=col, ls=ls, lw=lw,
                        label=f"{arm}  nMSE={wnm:.3f}", zorder=3)
            ax.set_title(f"{ent} ({cl}) — {split.upper()}", fontsize=9, loc="left")
            ax.legend(fontsize=7, loc="upper right", framealpha=.9, ncol=2); ax.grid(alpha=.15)
    pstr = " · ".join(f"{a}={params[a]/1000:.0f}k" for a in arms)
    tstr = "  ".join(f"{a} TEST {macro[a]['te']:.3f}(hf {hi[a]:.2f})" for a in arms)
    fig.suptitle(f"Latent-width test [Stage-1 only, BN-recal, {args.steps} steps] — {ds} seed{seed}\n"
                 f"params: {pstr}   |   macro {tstr}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = f"{args.outdir}/capacity_{ds}_s{seed}.png"; fig.savefig(p, dpi=130); plt.close(fig)
    json.dump({"rows": rows, "_macro": macro, "_params": params, "_hi": hi},
              open(f"{args.outdir}/capacity_{ds}_s{seed}.json", "w"), indent=1, default=float)
    print(f"\n[plot] {p}")


if __name__ == "__main__":
    main()
