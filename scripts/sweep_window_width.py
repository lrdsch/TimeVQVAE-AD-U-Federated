#!/usr/bin/env python3
"""Stage-1 ONLY 2-D sweep on wsd: window_length x conv-body width (width_base).

Why: the capacity test showed the conv BODY width (width_base) is the lever that
reduces reconstruction smoothing (token/codebook dim did nothing). This sweeps it
harder (4/16/32/64) and crosses it with SHORT windows (16/32), which reduce how
much signal must be squeezed through the latent.

CONTROL: `downsampled_width` is scaled with the window so the COMPRESSION RATIO
(~8x) and the network DEPTH (3 stages) stay constant — otherwise a short window at
the default dw=32 gives rate=1 -> depth=1 -> a degenerate 2-layer encoder with no
resnet blocks, and the width sweep would be meaningless.
  W=16 -> frames 17, dw=2 -> rate 8 -> depth 3      (body: wb, 2wb, 4wb)
  W=32 -> frames 33, dw=4 -> rate 8 -> depth 3
  (W=256 -> frames 257, dw=32 -> rate 8 -> depth 3  = the default/reference)

token_embedding_dim stays 4 (the d64 arm showed latent/codebook dim is a null).
Plot: a ~256-sample contiguous stretch rebuilt by TILING non-overlapping windows,
so window 16 vs 32 are visually comparable on the same time span.

    CUDA_VISIBLE_DEVICES=1 python scripts/sweep_window_width.py --dataset wsd_fed --n-clients 3 --steps 3000
"""
from __future__ import annotations
import argparse, dataclasses, json, os, sys
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, "pipeline"); sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from config import Config                                              # noqa: E402
from stage1 import Stage1VQVAE, save_stage1_checkpoint                 # noqa: E402
from data import make_dataloaders                                      # noqa: E402
from federated import _amp, _make_scaler, _cycle                       # noqa: E402
from utils import seed_everything                                      # noqa: E402
import local_recon_autopsy as A                                        # noqa: E402
from _fedpaths import conv_seed_root                                   # noqa: E402
import matplotlib; matplotlib.use("Agg")                               # noqa: E402
import matplotlib.pyplot as plt                                        # noqa: E402

_BN = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
WCOL = {4: "#b3323f", 16: "#d98a00", 32: "#2b5f9e", 64: "#1c7a4b"}
VIEW = 256          # samples shown in the tiled reconstruction view


def dw_for(window):
    """downsampled_width that keeps rate=round((W+1)/dw)=8 -> depth 3."""
    return max(1, round((window + 1) / 8))


def cfg_for(base, window, width_base):
    c = Config(); A._overlay(c, dataclasses.asdict(base))
    c.dataset.window_length = int(window)
    c.encoder.downsampled_width = int(dw_for(window))
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
    lat = None
    opt = torch.optim.AdamW(m.parameters(), lr=lr, fused=(torch.device(device).type == "cuda"))
    scaler = _make_scaler(); m.train(); it = _cycle(loader)
    tot = nb = 0
    for step in range(n_steps):
        x = next(it)["inputs"].to(device, non_blocking=True)
        with _amp():
            out = m(x); loss = out["losses"]["loss"]
        if lat is None:
            lat = tuple(int(s) for s in out["latent"].shape)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        lv = float(loss.detach())
        if lv == lv:
            tot += lv; nb += 1
        if step >= n_steps - 1 or (step + 1) % 1500 == 0:
            print(f"        step {step+1}/{n_steps} loss={tot/max(nb,1):.4f}", flush=True); tot = nb = 0
    return m, c, n_params, lat


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


def tile(X, rec, Y, W):
    """Stitch non-overlapping windows (stride-1 set → take every W-th) into a
    contiguous VIEW-sample stretch so different window lengths are comparable."""
    n = max(1, VIEW // W)
    idx = [i * W for i in range(n) if i * W < len(X)]
    o = np.concatenate([X[i, 0].numpy() for i in idx])
    r = np.concatenate([rec[i, 0].numpy() for i in idx])
    y = np.concatenate([Y[i].numpy() for i in idx])
    return o, r, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="wsd_fed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-clients", type=int, default=3)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--windows", default="16,32")
    ap.add_argument("--width-bases", default="4,16,32,64")
    ap.add_argument("--outdir", default="plots/codebook_analysis")
    ap.add_argument("--ckptdir", default="artifacts/fed_eval/sweep_ww")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds, seed = args.dataset, args.seed
    windows = [int(w) for w in args.windows.split(",")]
    wbs = [int(w) for w in args.width_bases.split(",")]
    clients = pick_clients(ds, seed, args.n_clients)
    os.makedirs(args.outdir, exist_ok=True)
    print(f"[sweep] {ds} seed{seed} — windows {windows} × width_base {wbs} — "
          f"clients {[e for _, e in clients]} — {args.steps} steps (Stage-1 only)\n")

    results = {}     # (W, wb) -> list of per-client dicts
    plots = {}       # W -> list of (ent, cl, stretch dict)
    for W in windows:
        print(f"########## window={W}  (downsampled_width={dw_for(W)}) ##########")
        plots[W] = []
        for cl, ent in clients:
            base = A.load_cfg(f"{conv_seed_root(ds, cl, seed)}/local/{ent}/stage1.ckpt")
            print(f"  ── {ent} ({cl}) ──")
            cW = cfg_for(base, W, wbs[0])
            Xtr, _ = A.get_windows(cW, ent, device, split="train")
            Xte, Yte = A.get_windows(cW, ent, device, split="test")
            stretch = {"X": Xte.cpu(), "Y": Yte, "rec": {}}
            for wb in wbs:
                m, c, np_, lat = train_steps(cfg_for(base, W, wb), ent, args.steps, args.lr, device, seed=seed)
                ckp = f"{args.ckptdir}/{ds}/w{W}_wb{wb}/{cl}/seed{seed}/local/{ent}/stage1.ckpt"
                save_stage1_checkpoint(__import__("pathlib").Path(ckp), m, c, step=args.steps, epoch=-1)
                recalibrate_bn(m, Xtr)
                rtr, _ = A.variants(m, Xtr); rte, diag = A.variants(m, Xte)
                d = dict(ent=ent, cl=cl, W=W, wb=wb, params=np_, latent=lat,
                         tr=A.nmse(rtr["full"], Xtr.cpu()), te=A.nmse(rte["full"], Xte.cpu()),
                         hi=A.spectral_retained(Xte.cpu()[:, 0].numpy(), rte["full"][:, 0].numpy())["high"],
                         ppl=diag["perplexity"])
                results.setdefault((W, wb), []).append(d)
                stretch["rec"][wb] = rte["full"]
                print(f"     wb={wb:3d} params={np_/1000:6.0f}k latent={lat} "
                      f"nMSE tr={d['tr']:.3f} te={d['te']:.3f} hi={d['hi']:.3f} ppl={d['ppl']:.1f}", flush=True)
            plots[W].append((ent, cl, stretch))
        print()

    # ── table ──
    print("=" * 86)
    print(f"{'window':>7s} {'wb':>4s} {'params':>9s} {'latent':>16s} {'train':>8s} {'test':>8s} {'hi-freq':>8s}")
    summ = {}
    for W in windows:
        for wb in wbs:
            rs = results[(W, wb)]
            mt = float(np.mean([r['tr'] for r in rs])); me = float(np.mean([r['te'] for r in rs]))
            mh = float(np.mean([r['hi'] for r in rs])); p = rs[0]['params']; lat = rs[0]['latent']
            summ[f"w{W}_wb{wb}"] = dict(train=mt, test=me, hi=mh, params=p, latent=list(lat))
            print(f"{W:7d} {wb:4d} {p/1000:8.0f}k {str(lat):>16s} {mt:8.3f} {me:8.3f} {mh:8.3f}")
    print("=" * 86)

    # ── FIG per window: rows=clients, tiled VIEW-sample stretch, overlay widths ──
    for W in windows:
        rows = plots[W]
        fig, axes = plt.subplots(len(rows), 1, figsize=(13, 2.7 * len(rows)), squeeze=False)
        for i, (ent, cl, S) in enumerate(rows):
            ax = axes[i][0]
            o, _, y = tile(S["X"], S["rec"][wbs[0]], S["Y"], W)
            t = np.arange(len(o))
            if (y >= 0).any() and (y > 0).any():
                ax.fill_between(t, o.min(), o.max(), where=(y > 0), color="#f2c14e", alpha=.30,
                                label="anomaly", zorder=0)
            ax.plot(t, o, color="#2b2f36", lw=1.7, label="original", zorder=5)
            for wb in wbs:
                _, r, _ = tile(S["X"], S["rec"][wb], S["Y"], W)
                me = float(np.mean([x['te'] for x in results[(W, wb)] if x['ent'] == ent]))
                ax.plot(t, r, color=WCOL.get(wb, "#666"), lw=1.2,
                        label=f"wb={wb}  test nMSE={me:.3f}", zorder=3)
            for b in range(W, len(o), W):        # window boundaries
                ax.axvline(b, color="#999", lw=0.4, alpha=.5, zorder=1)
            ax.set_title(f"{ent} ({cl}) — window={W} — {len(o)//W} tiled non-overlapping windows",
                         fontsize=9, loc="left")
            ax.legend(fontsize=7, loc="upper right", framealpha=.9, ncol=2); ax.grid(alpha=.15)
        tt = "  ".join(f"wb{wb}: {summ[f'w{W}_wb{wb}']['test']:.3f}" for wb in wbs)
        fig.suptitle(f"Stage-1 reconstruction — {ds} seed{seed} — window={W} "
                     f"(dw={dw_for(W)}, latent={summ[f'w{W}_wb{wbs[0]}']['latent']}, {args.steps} steps, BN-recal)\n"
                     f"macro TEST nMSE   {tt}", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        p = f"{args.outdir}/sweep_w{W}_{ds}_s{seed}.png"; fig.savefig(p, dpi=130); plt.close(fig)
        print(f"[plot] {p}")

    json.dump({"summary": summ,
               "rows": [d for v in results.values() for d in v]},
              open(f"{args.outdir}/sweep_ww_{ds}_s{seed}.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
