#!/usr/bin/env python3
"""Codebook divergence + reconstruction quality: local (own encoder + own codebook)
vs federated_cb_only (per-client encoder + ONE shared/merged codebook).

Answers three questions on the EXISTING converged checkpoints (no training):
  (1) how different is the shared codebook from each client's local codebook?
  (2) are the shared-codebook reconstructions still good?
  (3) plots: original vs local-recon vs shared-codebook-recon.

All windows come from make_dataloaders (per_entity_standard scaling — identical to
training), so reconstructions are in the space the model actually saw.

    python scripts/codebook_recon_probe.py --dataset wsd_fed --cluster c0 --seed 0
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
    xs = []
    for b in loader:
        xs.append(b["inputs"])
        if sum(x.shape[0] for x in xs) >= max_windows:
            break
    X = torch.cat(xs, 0)[:max_windows]           # (N, C, W) normalized
    return X.to(device)


@torch.no_grad()
def recon(s1, X, codebook=None, bs=64):
    """Reconstruct X; if codebook given, temporarily swap the VQ codebook (encoder/
    decoder unchanged) to measure the pure codebook effect at fixed encoder."""
    vq = s1.quantizer._vq
    saved = None
    if codebook is not None:
        saved = vq.codebook.weight.data.clone()
        vq.codebook.weight.data.copy_(codebook.to(vq.codebook.weight.device))
    recs, idxs, sse, sval = [], [], 0.0, 0.0
    for i in range(0, X.shape[0], bs):
        xb = X[i:i+bs]
        out = s1(xb)
        r = out["reconstructed"]
        recs.append(r.cpu()); idxs.append(out["quantizer_output"].indices.reshape(xb.shape[0], -1).cpu())
        sse += float(((r - xb)**2).sum()); sval += float((xb**2).sum())
    if saved is not None:
        vq.codebook.weight.data.copy_(saved)
    return torch.cat(recs, 0), torch.cat(idxs, 0), sse / sval


def codebook_usage(idx, K):
    u = torch.bincount(idx.reshape(-1), minlength=K).float()
    p = u / u.sum().clamp_min(1); nz = p[p > 0]
    return u, int((u > 0).sum()), float(torch.exp(-(nz*nz.log()).sum()))


def hungarian_nn(A, B):
    """mean nearest-neighbour L2 from rows of A to rows of B, and Hungarian-matched
    mean L2 (row permutation). A,B: (K,D)."""
    D2 = ((A[:, None, :] - B[None, :, :])**2).sum(-1)     # (K,K)
    nn = float(np.sqrt(D2.min(1)).mean())
    try:
        from scipy.optimize import linear_sum_assignment
        ri, ci = linear_sum_assignment(D2)
        hg = float(np.sqrt(D2[ri, ci]).mean())
    except Exception:
        hg = float("nan")
    return nn, hg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="wsd_fed")
    ap.add_argument("--cluster", default="c0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="test")
    ap.add_argument("--outdir", default="plots/codebook_analysis")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds, cl, seed = args.dataset, args.cluster, args.seed
    ents = entities_of(ds, cl)
    root = conv_seed_root(ds, cl, seed)
    os.makedirs(args.outdir, exist_ok=True)
    print(f"[probe] {ds}/{cl} seed{seed} — {len(ents)} clients: {ents}")

    per = {}          # ent -> dict of results
    local_cbs, shared_cb = {}, None
    plot_windows = []  # (ent, x, rec_local, rec_cb) for a few clients

    for ent in ents:
        d_loc = f"{root}/local/{ent}/stage1.ckpt"
        d_cb  = f"{root}/federated_cb_only/{ent}/stage1.ckpt"
        if not (os.path.exists(d_loc) and os.path.exists(d_cb)):
            print(f"  skip {ent} (missing ckpt)"); continue
        cfg = load_cfg(d_loc); W = int(cfg.dataset.window_length)
        C = 1
        ex = torch.zeros(1, C, W)
        s1_loc = load_stage1(d_loc, cfg, ex, device=device)
        s1_cb  = load_stage1(d_cb,  load_cfg(d_cb), ex, device=device)
        K = int(s1_loc.quantizer._vq.codebook_size)
        cb_loc = s1_loc.quantizer._vq.codebook.weight.detach().cpu().numpy()
        cb_sh  = s1_cb.quantizer._vq.codebook.weight.detach().cpu().numpy()
        local_cbs[ent] = cb_loc
        if shared_cb is None:
            shared_cb = cb_sh

        X = get_windows(cfg, ent, device, split=args.split)
        rec_loc, idx_loc, nmse_loc = recon(s1_loc, X)
        rec_cb,  idx_cb,  nmse_cb  = recon(s1_cb,  X)
        # ideal codebook for the cb_only ENCODER (k-means refit on its own latents) —
        # pure "cost of sharing the codebook" with encoder+decoder held fixed.
        with torch.no_grad():
            lat = s1_cb.encoder(s1_cb.transform(X))                 # (N, C*d, F, W')
        d = lat.shape[1]
        pts = lat.permute(0, 2, 3, 1).reshape(-1, d).cpu().numpy()
        nmse_ideal = float("nan")
        try:
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=K, n_init=3, random_state=0).fit(
                pts[np.random.RandomState(0).choice(len(pts), min(20000, len(pts)), replace=False)])
            cb_refit = torch.from_numpy(km.cluster_centers_.astype(np.float32))
            _, _, nmse_ideal = recon(s1_cb, X, codebook=cb_refit)
        except Exception as e:
            print("  (kmeans refit skipped:", e, ")")

        u_loc, na_loc, ppl_loc = codebook_usage(idx_loc, K)
        u_cb,  na_cb,  ppl_cb  = codebook_usage(idx_cb, K)
        per[ent] = dict(nmse_loc=nmse_loc, nmse_cb=nmse_cb, nmse_ideal=nmse_ideal,
                        na_loc=na_loc, na_cb=na_cb, ppl_loc=ppl_loc, ppl_cb=ppl_cb,
                        u_cb=u_cb.numpy())
        print(f"  {ent}: nMSE local={nmse_loc:.4f} cb_only={nmse_cb:.4f} "
              f"ideal(refit)={nmse_ideal:.4f} | active loc={na_loc} cb={na_cb} | "
              f"ppl loc={ppl_loc:.1f} cb={ppl_cb:.1f}")

        if len(plot_windows) < 4:
            # pick the most 'structured' window (highest std) for a legible plot
            j = int(X.reshape(X.shape[0], -1).std(1).argmax().item())
            plot_windows.append((ent, X[j, 0].cpu().numpy(),
                                 rec_loc[j, 0].numpy(), rec_cb[j, 0].numpy(),
                                 per[ent]))

    # ── codebook divergence numbers ──────────────────────────────────────────
    ents_ok = list(local_cbs)
    ll = []
    for i in range(len(ents_ok)):
        for j in range(i+1, len(ents_ok)):
            ll.append(hungarian_nn(local_cbs[ents_ok[i]], local_cbs[ents_ok[j]])[1])
    sh = [hungarian_nn(shared_cb, local_cbs[e])[1] for e in ents_ok]
    scale = float(np.sqrt((shared_cb**2).sum(1)).mean())
    print("\n=== CODEBOOK DIVERGENCE (Hungarian-matched mean L2) ===")
    print(f"  local-vs-local (across clients):  {np.mean(ll):.3f} ± {np.std(ll):.3f}")
    print(f"  shared-vs-local (per client):     {np.mean(sh):.3f} ± {np.std(sh):.3f}")
    print(f"  (codebook vector RMS scale ≈ {scale:.3f} — divergences are large relative to this)")
    nmse_loc_all = np.mean([per[e]['nmse_loc'] for e in per])
    nmse_cb_all  = np.mean([per[e]['nmse_cb']  for e in per])
    nmse_id_all  = np.nanmean([per[e]['nmse_ideal'] for e in per])
    print("\n=== RECONSTRUCTION (macro-mean nMSE over clients) ===")
    print(f"  local (own cb)          {nmse_loc_all:.4f}")
    print(f"  cb_only (shared cb)     {nmse_cb_all:.4f}   (+{100*(nmse_cb_all-nmse_loc_all)/nmse_loc_all:.0f}% vs local)")
    print(f"  ideal (refit, same enc) {nmse_id_all:.4f}   (pure codebook-sharing cost = {nmse_cb_all-nmse_id_all:.4f})")

    # ── FIG 1: reconstruction overlays ───────────────────────────────────────
    n = len(plot_windows)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.4*n), sharex=False)
    axes = np.atleast_1d(axes)
    for ax, (ent, x, rl, rc, pe) in zip(axes, plot_windows):
        t = np.arange(len(x))
        ax.plot(t, x, color="#2b2f36", lw=1.6, label="original", zorder=3)
        ax.plot(t, rl, color="#1c7a4b", lw=1.3, label=f"local (own cb)  nMSE={pe['nmse_loc']:.3f}")
        ax.plot(t, rc, color="#b3323f", lw=1.3, ls="--", label=f"shared cb  nMSE={pe['nmse_cb']:.3f}")
        ax.set_title(f"{ent}", fontsize=10, loc="left")
        ax.legend(fontsize=8, loc="upper right", framealpha=.9)
        ax.grid(alpha=.15)
    fig.suptitle(f"Reconstruction: local vs shared-codebook stage-1 — {ds}/{cl} seed{seed}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    p1 = f"{args.outdir}/recon_{ds}_{cl}_s{seed}.png"; fig.savefig(p1, dpi=130); plt.close(fig)

    # ── FIG 2: per-client nMSE bars ──────────────────────────────────────────
    es = list(per); x = np.arange(len(es)); w = 0.27
    fig, ax = plt.subplots(figsize=(max(7, 1.1*len(es)), 4))
    ax.bar(x-w, [per[e]['nmse_loc'] for e in es], w, label="local (own cb)", color="#1c7a4b")
    ax.bar(x,   [per[e]['nmse_cb']  for e in es], w, label="cb_only (shared cb)", color="#b3323f")
    ax.bar(x+w, [per[e]['nmse_ideal'] for e in es], w, label="ideal refit (same enc)", color="#986410")
    ax.set_xticks(x); ax.set_xticklabels(es, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("reconstruction nMSE"); ax.legend(fontsize=8)
    ax.set_title(f"Per-client reconstruction error — {ds}/{cl} seed{seed}")
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout(); p2 = f"{args.outdir}/nmse_{ds}_{cl}_s{seed}.png"; fig.savefig(p2, dpi=130); plt.close(fig)

    # ── FIG 3: codebook geometry (PCA 2D) + NN-dist hist ─────────────────────
    from numpy.linalg import svd
    allcb = np.concatenate([shared_cb] + [local_cbs[e] for e in ents_ok], 0)
    mu = allcb.mean(0); U, S, Vt = svd(allcb - mu, full_matrices=False); pc = Vt[:2].T
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    e0 = ents_ok[0]
    a1.scatter(*( (local_cbs[e0]-mu)@pc ).T, s=18, c="#1c7a4b", label=f"local cb ({e0})", alpha=.8)
    a1.scatter(*( (shared_cb-mu)@pc ).T, s=30, c="#b3323f", marker="^", label="shared cb", alpha=.85)
    a1.set_title("codebook geometry (PCA-2D)"); a1.legend(fontsize=8); a1.grid(alpha=.15)
    nn_dists = np.sqrt(((shared_cb[:, None]-local_cbs[e0][None])**2).sum(-1)).min(1)
    a2.hist(nn_dists, bins=20, color="#2b5f9e", alpha=.85)
    a2.axvline(scale, color="#b3323f", ls="--", label=f"cb RMS scale {scale:.2f}")
    a2.set_title(f"shared→local nearest-neighbour L2 ({e0})"); a2.set_xlabel("L2"); a2.legend(fontsize=8)
    fig.suptitle(f"How different is the shared codebook from a local one — {ds}/{cl}", fontsize=12)
    fig.tight_layout(rect=[0,0,1,.96]); p3 = f"{args.outdir}/codebook_{ds}_{cl}_s{seed}.png"; fig.savefig(p3, dpi=130); plt.close(fig)

    json.dump({e: {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in per[e].items()}
               for e in per} | {"_div": {"local_local": float(np.mean(ll)),
               "shared_local": float(np.mean(sh)), "cb_scale": scale}},
              open(f"{args.outdir}/{ds}_{cl}_s{seed}.json", "w"), indent=1)
    print(f"\n[plots] {p1}\n        {p2}\n        {p3}")


if __name__ == "__main__":
    main()
