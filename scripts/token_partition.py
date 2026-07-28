#!/usr/bin/env python3
"""Experiment B — is the SHARED codebook a shared language, or is it implicitly
PARTITIONED per client? Distinguishes benign per-client under-utilisation from
fatal global partition.

For each (dataset, cluster), every client tokenises its OWN train windows with the
shared-codebook (federated_cb_only) stage1. We build P(token=j | client=k) and measure
  * U_global      = # codewords used by ANY client (union)
  * mean_local    = mean per-client # active codes
  * overlap       = mean pairwise Jaccard of clients' used-code sets (1=identical use)
  * I(K;J)/H(K)   = normalised mutual info between client id and token — the KEY number:
                    ~0 => clients share the vocabulary; ~1 => token identifies the client
                    (the vocabulary is partitioned, so a GLOBAL prior gets no shared language).
Then correlates I(K;J)/H(K) across clusters with the federation penalty Δ(fed−local)
and Δ(shared−local) from the converged sweep.

    python scripts/token_partition.py --dataset wsd_fed
"""
from __future__ import annotations
import argparse, csv, dataclasses, json, os, sys, warnings, itertools
import numpy as np, torch
warnings.filterwarnings("ignore")
sys.path.insert(0, "pipeline"); sys.path.insert(0, ".")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _fedpaths import conv_root, conv_seed_root  # noqa: E402
from config import Config                       # noqa: E402
from stage1 import load_stage1                  # noqa: E402
from data import make_dataloaders               # noqa: E402
import matplotlib; matplotlib.use("Agg")        # noqa: E402
import matplotlib.pyplot as plt                 # noqa: E402


def overlay(o, d):
    for k, v in d.items():
        if not hasattr(o, k):
            continue
        c = getattr(o, k)
        if dataclasses.is_dataclass(c) and isinstance(v, dict):
            overlay(c, v)
        else:
            setattr(o, k, v)


@torch.no_grad()
def client_usage(ds, cl, seed, ent, device, arm="federated_cb_only", split="train"):
    root = conv_seed_root(ds, cl, seed)
    s1c = f"{root}/{arm}/{ent}/stage1.ckpt"
    if not os.path.exists(s1c):
        return None
    ck = torch.load(s1c, map_location="cpu", weights_only=False)
    cfg = Config(); overlay(cfg, ck["cfg_dict"]); W = int(cfg.dataset.window_length)
    K = int(cfg.quantizer.codebook_size)
    s1 = load_stage1(s1c, cfg, torch.zeros(1, 1, W), device=device)
    c = Config(); overlay(c, ck["cfg_dict"]); c.dataset.entity_id = ent
    dl = make_dataloaders(c, stage="stage1")
    loader = {"train": dl.train_loader, "test": dl.test_loader}[split]
    hist = torch.zeros(K)
    for b in loader:
        _, idx, _ = s1.encode_tokens(b["inputs"].to(device))
        hist += torch.bincount(idx.reshape(-1).cpu(), minlength=K).float()
    return hist.numpy(), K


def mutual_info(P):
    """P: (n_clients, K) counts. Returns I(K;J)/H(K), U_global, mean_active, overlap."""
    n, K = P.shape
    nk = P.sum(1)                                  # windows-tokens per client
    Pk = nk / nk.sum()                             # P(client)
    Pjk = P / P.sum(1, keepdims=True).clip(1)      # P(token|client)
    Pj = (Pk[:, None] * Pjk).sum(0)                # P(token)
    I = 0.0
    for k in range(n):
        pjk = Pjk[k]; nz = pjk > 0
        I += Pk[k] * float((pjk[nz] * (np.log(pjk[nz]) - np.log(Pj[nz].clip(1e-12)))).sum())
    HK = -float((Pk * np.log(Pk.clip(1e-12))).sum())
    sets = [set(np.where(P[k] > 0)[0]) for k in range(n)]
    U = len(set().union(*sets))
    mean_active = float(np.mean([len(s) for s in sets]))
    jac = [len(sets[i] & sets[j]) / max(1, len(sets[i] | sets[j]))
           for i, j in itertools.combinations(range(n), 2)]
    return I / HK if HK > 0 else float("nan"), U, mean_active, float(np.mean(jac)) if jac else float("nan"), Pjk, K


def cluster_deltas(ds):
    """Per-cluster matched Δ(fed−local) and Δ(shared−local) VUS-PR from converged CSV."""
    path = f"{conv_root(ds)}.csv"
    rows = {}
    if not os.path.exists(path):
        return {}
    for r in csv.DictReader(open(path)):
        rows.setdefault((r["cluster"], r["seed"], r["entity"]), {})[r["arm"]] = float(r["vus_pr"])
    out = {}
    from collections import defaultdict
    fd, sd = defaultdict(list), defaultdict(list)
    for (clu, s, e), d in rows.items():
        if "local" in d and "federated" in d:
            fd[clu].append(d["federated"] - d["local"])
        if "local" in d and "federated_shared" in d:
            sd[clu].append(d["federated_shared"] - d["local"])
    for clu in fd:
        out[clu] = {"fed_local": float(np.mean(fd[clu])), "shared_local": float(np.mean(sd[clu])) if sd[clu] else float("nan")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="wsd_fed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="train")
    ap.add_argument("--outdir", default="plots/codebook_analysis")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = args.dataset
    clusters = json.load(open(f"data/raw/{ds}/clusters.json"))
    deltas = cluster_deltas(ds)
    os.makedirs(args.outdir, exist_ok=True)

    rows, heatmaps = [], {}
    for cl, ents in clusters.items():
        hs, ok = [], []
        Kv = None
        for e in ents:
            r = client_usage(ds, cl, args.seed, e, device, split=args.split)
            if r is None:
                continue
            hs.append(r[0]); ok.append(e); Kv = r[1]
        if len(hs) < 2:
            print(f"  {cl}: <2 clients with ckpts, skip"); continue
        P = np.stack(hs)
        I, U, ma, ov, Pjk, K = mutual_info(P)
        d = deltas.get(cl, {})
        rows.append((cl, len(ok), U, K, ma, ov, I, d.get("fed_local", float("nan")), d.get("shared_local", float("nan"))))
        heatmaps[cl] = (Pjk, ok)
        print(f"  {cl}: clients={len(ok)} U_global={U}/{K} mean_active={ma:.1f} "
              f"jaccard_overlap={ov:.2f} I(K;J)/H(K)={I:.3f} | Δfed-loc={d.get('fed_local',float('nan')):+.3f}")

    # summary table
    print(f"\n=== Experiment B — token partition of the SHARED codebook — {ds} seed{args.seed} ({args.split}) ===")
    print(f"{'cluster':10s}{'n':>3s}{'U_glob/K':>10s}{'mean_act':>9s}{'overlap':>8s}{'I(K;J)/H':>10s}{'Δfed-loc':>10s}{'Δshr-loc':>10s}")
    for (cl, n, U, K, ma, ov, I, fl, sl) in rows:
        print(f"{cl:10s}{n:>3d}{f'{U}/{K}':>10s}{ma:>9.1f}{ov:>8.2f}{I:>10.3f}{fl:>10.3f}{sl:>10.3f}")
    # correlation I(K;J) vs penalty
    if len(rows) >= 3:
        I = np.array([r[6] for r in rows]); fl = np.array([r[7] for r in rows]); sl = np.array([r[8] for r in rows])
        from scipy.stats import spearmanr
        m = ~np.isnan(fl)
        print(f"\nSpearman I(K;J)/H(K) vs Δ(fed−local):    ρ={spearmanr(I[m], fl[m])[0]:+.3f} (n={m.sum()})  "
              "[user hypothesis: MORE partition → MORE negative Δ ⇒ ρ<0]")
        ms = ~np.isnan(sl)
        if ms.sum() >= 3:
            print(f"Spearman I(K;J)/H(K) vs Δ(shared−local): ρ={spearmanr(I[ms], sl[ms])[0]:+.3f} (n={ms.sum()})")

    # heatmap figure (one panel per cluster)
    ncl = len(heatmaps)
    fig, axes = plt.subplots(1, ncl, figsize=(3.2*ncl, 3.6), squeeze=False)
    for ax, (cl, (Pjk, ok)) in zip(axes[0], heatmaps.items()):
        im = ax.imshow(Pjk, aspect="auto", cmap="magma", interpolation="nearest")
        ax.set_title(f"{cl}\nI(K;J)/H={[r[6] for r in rows if r[0]==cl][0]:.2f}", fontsize=9)
        ax.set_xlabel("codeword j", fontsize=8); ax.set_yticks(range(len(ok)))
        ax.set_yticklabels(ok, fontsize=6)
    fig.suptitle(f"P(token j | client k) with the SHARED codebook — {ds}  (row-normalised; "
                 "disjoint bright bands = per-client partition)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = f"{args.outdir}/partition_{ds}_s{args.seed}.png"; fig.savefig(p, dpi=130); plt.close(fig)
    json.dump([{"cluster": r[0], "n": r[1], "U_global": r[2], "K": r[3], "mean_active": r[4],
                "jaccard_overlap": r[5], "I_KJ_norm": r[6], "delta_fed_local": r[7],
                "delta_shared_local": r[8]} for r in rows],
              open(f"{args.outdir}/partition_{ds}_s{args.seed}.json", "w"), indent=1)
    print(f"\n[plot] {p}")


if __name__ == "__main__":
    main()
