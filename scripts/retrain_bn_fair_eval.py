#!/usr/bin/env python3
"""Fair old-vs-new reconstruction eval that CONTROLS FOR the BatchNorm train/eval
gap. The 10k-step retrain leaves BN running-stats miscalibrated, so raw eval-mode
nMSE is contaminated. Here we recompute nMSE after recalibrating BN (reset running
stats, cumulative-average a few passes over the train windows), for BOTH models,
on train and test. The BN-recalibrated numbers isolate the LEARNED representation.

    python scripts/retrain_bn_fair_eval.py --dataset wsd_fed --seed 0
"""
from __future__ import annotations
import argparse, glob, os, sys, json
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, "pipeline"); sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from stage1 import Stage1VQVAE                                          # noqa: E402
import local_recon_autopsy as A                                        # noqa: E402
from _fedpaths import conv_root                                        # noqa: E402

BN = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def load(ckpt, cfg, ex, device="cpu"):
    st = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    m = Stage1VQVAE(cfg); m.eval()
    with torch.no_grad():
        m(ex.cpu())
    m.load_state_dict(st["state_dict"], strict=True)
    m.to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@torch.no_grad()
def nmse(m, X, bs=128):
    m.eval(); sse = sval = 0.0
    for i in range(0, len(X), bs):
        xb = X[i:i+bs]; r = m(xb)["reconstructed"]
        sse += float(((r - xb)**2).sum()); sval += float((xb**2).sum())
    return sse / sval


@torch.no_grad()
def recalibrate_bn(m, Xtr, bs=128, passes=5):
    """Reset BN running stats and re-estimate them on the train windows (cumulative
    average, momentum=None) — the standard BN-recalibration fix. Model back in eval()."""
    saved = {}
    for mod in m.modules():
        if isinstance(mod, BN):
            saved[mod] = mod.momentum
            mod.reset_running_stats(); mod.momentum = None
    m.train()
    for _ in range(passes):
        for i in range(0, len(Xtr), bs):
            m(Xtr[i:i+bs])
    for mod, mom in saved.items():
        mod.momentum = mom
    m.eval()


def clients(ds, seed, ckptdir):
    out = []
    for p in sorted(glob.glob(f"{ckptdir}/{ds}/*/seed{seed}/local/*/stage1.ckpt")):
        parts = p.split("/"); out.append((parts[-5], parts[-2], p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="wsd_fed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckptdir", default="artifacts/fed_eval/retrain10k")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds, seed = args.dataset, args.seed
    conv = conv_root(ds)
    cl = clients(ds, seed, args.ckptdir)
    if not cl:
        print(f"[bnfair] no retrain ckpts for {ds} seed{seed}"); return
    print(f"[bnfair] {ds} seed{seed} — {len(cl)} clients — BN-recalibrated old-vs-new\n")
    rows = []
    for c, ent, newck in cl:
        oldck = f"{conv}/{c}/seed{seed}/local/{ent}/stage1.ckpt"
        if not os.path.exists(oldck):
            continue
        cfg = A.load_cfg(oldck); ex = torch.zeros(1, 1, int(cfg.dataset.window_length))
        Xtr, _ = A.get_windows(cfg, ent, device, split="train")
        Xte, _ = A.get_windows(cfg, ent, device, split="test")
        old = load(oldck, cfg, ex, device); new = load(newck, A.load_cfg(newck), ex, device)
        # raw (as-deployed) eval-mode
        r = dict(ent=ent, cl=c,
                 tr_old=nmse(old, Xtr), tr_new=nmse(new, Xtr),
                 te_old=nmse(old, Xte), te_new=nmse(new, Xte))
        # recalibrate BN of BOTH on their own train, then re-measure
        recalibrate_bn(old, Xtr); recalibrate_bn(new, Xtr)
        r.update(tr_old_bn=nmse(old, Xtr), tr_new_bn=nmse(new, Xtr),
                 te_old_bn=nmse(old, Xte), te_new_bn=nmse(new, Xte))
        rows.append(r)
        print(f"  {ent:10s}({c}):")
        print(f"     RAW eval   TRAIN old={r['tr_old']:.3f} new={r['tr_new']:.3f}  | TEST old={r['te_old']:.3f} new={r['te_new']:.3f}")
        print(f"     BN-recal   TRAIN old={r['tr_old_bn']:.3f} new={r['tr_new_bn']:.3f}  | TEST old={r['te_old_bn']:.3f} new={r['te_new_bn']:.3f}"
              f"   (test Δ new-old = {r['te_new_bn']-r['te_old_bn']:+.3f})")
    m = lambda k: float(np.mean([x[k] for x in rows]))
    print("\n=== MACRO ===")
    print(f"  RAW eval  TRAIN old={m('tr_old'):.3f} new={m('tr_new'):.3f} | TEST old={m('te_old'):.3f} new={m('te_new'):.3f}")
    print(f"  BN-recal  TRAIN old={m('tr_old_bn'):.3f} new={m('tr_new_bn'):.3f} | TEST old={m('te_old_bn'):.3f} new={m('te_new_bn'):.3f}")
    print(f"\n  >> After BN fix: TRAIN new-old = {m('tr_new_bn')-m('tr_old_bn'):+.3f} (neg = new fits train better)")
    print(f"  >> After BN fix: TEST  new-old = {m('te_new_bn')-m('te_old_bn'):+.3f} (pos = genuine overfit; ~0/neg = no overfit)")
    json.dump(rows, open(f"plots/codebook_analysis/bnfair_{ds}_s{seed}.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
