#!/usr/bin/env python3
"""DEFINITIVE width sweep: Stage-1 only, ALL clients of a dataset, paired stats.

Answers "does the conv-body width (cfg.encoder.width_base) reduce reconstruction
smoothing?" with enough clients to be conclusive, instead of the 3-client pilots.

Design:
  * window_length forced to 256 (the real config), downsampled_width 32 -> depth 3.
  * token_embedding_dim left at 4 (the d64 arm proved latent/codebook dim is a null).
  * PAIRED: every client is trained at every width from the same seed -> Wilcoxon
    signed-rank on Delta(test nMSE) and Delta(hi-freq) between the widest and
    narrowest arm. Paired design is far more powerful than cross-client unpaired.
  * Builds the Config from config.py + apply_dataset_overrides, so it works on
    datasets WITHOUT converged checkpoints (e.g. toy_fed_uni_ucrlike).
  * BN-recalibrated eval (raw eval-mode nMSE is BatchNorm-contaminated).

    CUDA_VISIBLE_DEVICES=1 python scripts/sweep_width_all_clients.py --dataset wsd_fed --n-clients 0
"""
from __future__ import annotations
import argparse, dataclasses, json, os, sys, traceback
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, "pipeline"); sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from config import Config, apply_dataset_overrides                     # noqa: E402
from stage1 import Stage1VQVAE, save_stage1_checkpoint                 # noqa: E402
from data import make_dataloaders                                      # noqa: E402
from federated import _amp, _make_scaler, _cycle                       # noqa: E402
from utils import seed_everything                                      # noqa: E402
import local_recon_autopsy as A                                        # noqa: E402
import matplotlib; matplotlib.use("Agg")                               # noqa: E402
import matplotlib.pyplot as plt                                        # noqa: E402

_BN = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def base_cfg(ds, window):
    """Config WITHOUT needing a converged checkpoint (verified: none of the fed
    datasets declare metadata 'period', so no window override fires)."""
    c = Config()
    c.dataset.name = ds
    apply_dataset_overrides(c)
    c.dataset.window_length = int(window)
    c.encoder.downsampled_width = 32 if window >= 256 else max(1, round((window + 1) / 8))
    return c


def cfg_for(ds, window, wb, entity):
    c = base_cfg(ds, window)
    c.encoder.width_base = int(wb)
    c.dataset.entity_id = entity
    return c


def train_steps(cfg, n_steps, lr, device, seed=0):
    seed_everything(seed)
    loader = make_dataloaders(cfg, stage="stage1").train_loader
    ex = next(iter(loader))["inputs"][:1].cpu()
    m = Stage1VQVAE(cfg); m.eval()
    with torch.no_grad():
        m(ex)
    m.to(device); m.quantizer.collect_stats_only = False
    n_params = sum(p.numel() for p in m.parameters())
    opt = torch.optim.AdamW(m.parameters(), lr=lr, fused=(torch.device(device).type == "cuda"))
    scaler = _make_scaler(); m.train(); it = _cycle(loader)
    for step in range(n_steps):
        x = next(it)["inputs"].to(device, non_blocking=True)
        with _amp():
            loss = m(x)["losses"]["loss"]
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    return m, n_params


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


def wilcoxon(d):
    try:
        from scipy.stats import wilcoxon as W
        if len(d) >= 6 and np.any(np.asarray(d) != 0):
            s, p = W(d)
            return float(p)
    except Exception:
        pass
    return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="wsd_fed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-clients", type=int, default=0, help="0 = ALL clients")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--width-bases", default="4,16,32")
    ap.add_argument("--outdir", default="plots/codebook_analysis")
    ap.add_argument("--ckptdir", default="artifacts/fed_eval/width_all")
    ap.add_argument("--save-ckpt", action="store_true")
    ap.add_argument("--resume-file", default="", help="per-client incremental results (auto-resume)")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds, seed = args.dataset, args.seed
    wbs = [int(w) for w in args.width_bases.split(",")]
    cl = json.load(open(f"data/raw/{ds}/clusters.json"))
    ents = [(c, e) for c, es in cl.items() for e in es]
    if args.n_clients:
        ents = ents[:args.n_clients]
    print(f"[width-all] {ds} seed{seed} — {len(ents)} clients × width {wbs} "
          f"— window {args.window} — {args.steps} steps (Stage-1 only)\n", flush=True)

    # ── resume: per-client results are saved incrementally so a kill doesn't lose hours ──
    part = args.resume_file or f"{args.outdir}/widthall_{ds}_s{seed}_partial.json"
    res, params = {}, {}
    if os.path.exists(part):
        raw = json.load(open(part))
        res = {e: {(int(k) if k.isdigit() else k): v for k, v in d.items()} for e, d in raw.get("res", {}).items()}
        params = {int(k): v for k, v in raw.get("params", {}).items()}
        print(f"[resume] loaded {len(res)} clients from {part}", flush=True)

    def _save():
        json.dump({"res": {e: {str(k): v for k, v in d.items()} for e, d in res.items()},
                   "params": {str(k): v for k, v in params.items()}},
                  open(part, "w"), indent=1, default=float)

    for k, (c, ent) in enumerate(ents, 1):
        if ent in res and all(w in res[ent] for w in wbs):
            print(f"[{k}/{len(ents)}] {ent} ({c}) — already done, skip", flush=True); continue
        print(f"[{k}/{len(ents)}] {ent} ({c})", flush=True)
        try:
            cfg0 = cfg_for(ds, args.window, wbs[0], ent)
            Xtr, _ = A.get_windows(cfg0, ent, device, split="train")
            Xte, _ = A.get_windows(cfg0, ent, device, split="test")
        except Exception as e:
            print(f"   SKIP data: {type(e).__name__}: {e}", flush=True); continue
        res[ent] = {"_cluster": c}
        for wb in wbs:
            try:
                m, np_ = train_steps(cfg_for(ds, args.window, wb, ent), args.steps, args.lr, device, seed=seed)
                params[wb] = np_
                if args.save_ckpt:
                    save_stage1_checkpoint(__import__("pathlib").Path(
                        f"{args.ckptdir}/{ds}/wb{wb}/{c}/seed{seed}/local/{ent}/stage1.ckpt"),
                        m, cfg_for(ds, args.window, wb, ent), step=args.steps, epoch=-1)
                recalibrate_bn(m, Xtr)
                rtr, _ = A.variants(m, Xtr); rte, diag = A.variants(m, Xte)
                res[ent][wb] = dict(
                    tr=A.nmse(rtr["full"], Xtr.cpu()), te=A.nmse(rte["full"], Xte.cpu()),
                    hi=A.spectral_retained(Xte.cpu()[:, 0].numpy(), rte["full"][:, 0].numpy())["high"],
                    ppl=diag["perplexity"])
                print(f"   wb={wb:3d} nMSE tr={res[ent][wb]['tr']:.3f} te={res[ent][wb]['te']:.3f} "
                      f"hi={res[ent][wb]['hi']:.3f} ppl={diag['perplexity']:.1f}", flush=True)
                del m; torch.cuda.empty_cache()
            except Exception as e:
                print(f"   wb={wb} FAILED: {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
        _save()          # checkpoint after EVERY client

    ok = [e for e in res if all(w in res[e] for w in wbs)]
    print(f"\n[width-all] complete clients: {len(ok)}/{len(ents)}")
    if not ok:
        return
    # ── macro + paired stats ──
    print("=" * 72)
    print(f"{'width':>6s} {'params':>9s} {'test nMSE (mean±sd)':>24s} {'hi-freq':>16s}")
    summ = {}
    for wb in wbs:
        te = np.array([res[e][wb]['te'] for e in ok]); hi = np.array([res[e][wb]['hi'] for e in ok])
        summ[wb] = dict(te_mean=float(te.mean()), te_sd=float(te.std()),
                        hi_mean=float(hi.mean()), params=params.get(wb))
        print(f"{wb:6d} {params.get(wb,0)/1000:8.0f}k {te.mean():14.4f} ± {te.std():.4f} {hi.mean():14.3f}")
    lo, hi_w = wbs[0], wbs[-1]
    d_te = np.array([res[e][hi_w]['te'] - res[e][lo]['te'] for e in ok])
    d_hi = np.array([res[e][hi_w]['hi'] - res[e][lo]['hi'] for e in ok])
    print("=" * 72)
    print(f"PAIRED wb{hi_w} vs wb{lo}  (n={len(ok)} clients)")
    print(f"  Δ test nMSE : median {np.median(d_te):+.4f}  mean {d_te.mean():+.4f}  "
          f"improved {int((d_te<0).sum())}/{len(ok)}  Wilcoxon p={wilcoxon(d_te):.2g}")
    print(f"  Δ hi-freq   : median {np.median(d_hi):+.4f}  mean {d_hi.mean():+.4f}  "
          f"improved {int((d_hi>0).sum())}/{len(ok)}  Wilcoxon p={wilcoxon(d_hi):.2g}")
    print("=" * 72)

    # ── FIG: width curve + paired delta ──
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
    for e in ok:
        a1.plot(wbs, [res[e][w]['te'] for w in wbs], color="#bbb", lw=0.8, alpha=.7, zorder=1)
    mean = [summ[w]['te_mean'] for w in wbs]
    sem = [np.std([res[e][w]['te'] for e in ok]) / np.sqrt(len(ok)) for w in wbs]
    a1.errorbar(wbs, mean, yerr=sem, color="#1c7a4b", lw=2.4, marker="o", capsize=4,
                label=f"mean ± sem (n={len(ok)})", zorder=3)
    a1.set_xscale("log", base=2); a1.set_xticks(wbs); a1.set_xticklabels([str(w) for w in wbs])
    a1.set_xlabel("width_base (conv body)"); a1.set_ylabel("test nMSE (BN-recal)")
    a1.set_title(f"{ds}: reconstruction error vs body width", fontsize=10)
    a1.legend(fontsize=8); a1.grid(alpha=.2)

    order = np.argsort(d_te)
    a2.bar(range(len(ok)), d_te[order],
           color=["#1c7a4b" if v < 0 else "#b3323f" for v in d_te[order]])
    a2.axhline(0, color="#333", lw=.8)
    a2.set_xlabel(f"client (sorted) — green = wb{hi_w} better")
    a2.set_ylabel(f"Δ test nMSE (wb{hi_w} − wb{lo})")
    a2.set_title(f"paired Δ: {int((d_te<0).sum())}/{len(ok)} improved, "
                 f"median {np.median(d_te):+.3f}, p={wilcoxon(d_te):.2g}", fontsize=10)
    a2.grid(axis="y", alpha=.2)
    fig.suptitle(f"Body-width sweep — {ds} — ALL {len(ok)} clients — window {args.window}, "
                 f"{args.steps} steps, Stage-1 only, BN-recal", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, .94])
    p = f"{args.outdir}/widthall_{ds}_s{seed}.png"; fig.savefig(p, dpi=130); plt.close(fig)
    json.dump({"summary": {str(k): v for k, v in summ.items()},
               "per_client": {e: {str(k): v for k, v in res[e].items()} for e in ok},
               "paired": {"lo": lo, "hi": hi_w, "n": len(ok),
                          "d_te_median": float(np.median(d_te)), "d_te_p": wilcoxon(d_te),
                          "d_hi_median": float(np.median(d_hi)), "d_hi_p": wilcoxon(d_hi)}},
              open(f"{args.outdir}/widthall_{ds}_s{seed}.json", "w"), indent=1, default=float)
    print(f"[plot] {p}")


if __name__ == "__main__":
    main()
