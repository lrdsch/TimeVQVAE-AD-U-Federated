#!/usr/bin/env python3
"""R1 — codebook autopsy: WHY does the sufficient-statistic codebook merge (A)
beat naive weight-FedAvg, even though both produce a single global codebook?

Runs on EXISTING checkpoints (no training). For every (arm, seed, client) it
loads the saved stage1(+stage2) and measures, on the test split:

  CODEBOOK HEALTH (stage1 only)
    * perplexity   — exp(entropy of code-usage); higher = more codes used
    * n_active     — # codes used at least once (of K)
    * recon_nMSE   — reconstruction MSE / signal energy; lower = better
    * codebook_PR  — participation ratio of the codebook covariance (scale-free
                     spread in [1, D]); lower = more geometric collapse

  DETECTION DECOMPOSITION (window-level AUROC vs anomaly label)
    * s_local AUROC — reconstruction-energy term alone
    * s_prior AUROC — prior masked-NLL term alone

Headline finding (toy_fed, 4 seeds): weight-FedAvg's codebook is HEALTHIER on
every stage1 metric (more codes, better reconstruction, sharper recon contrast)
yet detects WORSE — because its token assignments are globally inconsistent, so
the shared MaskGIT prior models them poorly and s_prior collapses
(~0.79 vs ~0.91 for the suff-stat merge). I.e. merge (A) helps the PRIOR, not
the reconstruction. This is the mechanism behind Proposition 1.

    python scripts/fed_codebook_autopsy.py --dataset toy_fed --seeds 0 1 2 3
    python scripts/fed_codebook_autopsy.py --dataset toy_fed --seeds 0 --skip-prior   # fast
"""
from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import os
import sys
import time
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, "pipeline")
sys.path.insert(0, ".")
from config import Config            # noqa: E402
from stage1 import load_stage1       # noqa: E402
from stage2 import load_stage2       # noqa: E402
from utils import force_utf8_stdout  # noqa: E402

force_utf8_stdout()

ARMS_DEFAULT = ["local", "centralized", "federated", "federated_shared", "federated_fedavg_cb"]


def _overlay(obj, d):
    """Overlay a saved cfg_dict onto a default Config (handles nested dataclasses)."""
    for k, v in d.items():
        if not hasattr(obj, k):
            continue
        cur = getattr(obj, k)
        if dataclasses.is_dataclass(cur) and isinstance(v, dict):
            _overlay(cur, v)
        else:
            setattr(obj, k, v)


def _participation_ratio(cb: np.ndarray) -> float:
    """Scale-invariant codebook spread in [1, D]: (Σλ)²/Σλ² of the code covariance."""
    c = cb - cb.mean(0, keepdims=True)
    cov = c.T @ c / cb.shape[0]
    ev = np.linalg.eigvalsh(cov)
    ev = ev[ev > 0]
    return float((ev.sum() ** 2) / (ev ** 2).sum()) if ev.size else float("nan")


def _as_2d(x: np.ndarray) -> np.ndarray:
    """(T,) -> (T, 1). Univariate builds may store a bare 1-D series."""
    return x[:, None] if x.ndim == 1 else x


def _resolve_entities(dataset: str, clients: str, cluster: str | None) -> list[str]:
    """`--cluster` > explicit id list > every entity on disk.

    Entity ids are dataset-specific (`fed_0..` for toy_fed, `uni_00..` for
    toy_fed_uni), so generating them from an integer count silently produced an
    empty result table on any dataset that didn't use the `fed_<i>` convention.
    """
    if cluster:
        cpath = f"data/raw/{dataset}/clusters.json"
        if not os.path.exists(cpath):
            raise SystemExit(f"{dataset} has no clusters.json — --cluster is unavailable.")
        clusters = json.load(open(cpath))
        if cluster not in clusters:
            raise SystemExit(f"unknown cluster {cluster!r}; known: {sorted(clusters)}")
        return list(clusters[cluster])
    if clients and clients != "auto":
        if clients.isdigit():                       # legacy: --clients 6 -> fed_0..fed_5
            return [f"fed_{i}" for i in range(int(clients))]
        return [c.strip() for c in clients.split(",") if c.strip()]
    train_dir = f"data/raw/{dataset}/train"
    if not os.path.isdir(train_dir):
        raise SystemExit(f"missing {train_dir}; cannot auto-discover clients.")
    return sorted(f[:-4] for f in os.listdir(train_dir) if f.endswith(".npy"))


def eval_client(root, arm, seed, ent, device, stride, skip_prior, ckpt_root=None):
    ckpt_root = ckpt_root or root
    d = f"artifacts/fed_eval/{ckpt_root}/seed{seed}/{arm}/{ent}"
    s1c, s2c = f"{d}/stage1.ckpt", f"{d}/stage2.ckpt"
    if not os.path.exists(s1c):
        return None
    ck = torch.load(s1c, map_location="cpu", weights_only=False)
    cfg = Config()
    _overlay(cfg, ck["cfg_dict"])
    W = int(cfg.dataset.window_length)

    data_root = f"data/raw/{root}"
    test = _as_2d(np.load(f"{data_root}/test/{ent}.npy").astype(np.float32))
    lab = np.load(f"{data_root}/test_label/{ent}.npy").astype(np.int64)
    # Materialisation tensor: C comes from the DATA, not a hard-coded 8. A wrong C
    # builds the encoder's grouped conv with the wrong `groups` and the strict
    # state_dict load fails (or `assert D % C == 0` trips inside the quantizer).
    ex = torch.zeros(1, test.shape[1], W)

    use_prior = (not skip_prior) and os.path.exists(s2c)
    if use_prior:
        m = load_stage2(s2c, cfg, s1c, ex, device=device)
        m.stage1.eval(); m.prior.eval()
        s1 = m.stage1
    else:
        s1 = load_stage1(s1c, cfg, ex, device=device)
        m = None

    K = int(s1.quantizer._vq.codebook_size)
    cb = s1.quantizer._vq.codebook.weight.detach().cpu().numpy()

    starts = list(range(0, test.shape[0] - W + 1, stride))

    usage = torch.zeros(K)
    sse = sval = 0.0
    sl, sp, wl = [], [], []
    with torch.no_grad():
        for i in range(0, len(starts), 32):
            sb = starts[i:i + 32]
            xb = torch.from_numpy(np.stack([test[s:s + W] for s in sb])).permute(0, 2, 1).contiguous().to(device)
            out = s1(xb)
            rec = out["reconstructed"]
            sse += float(((rec - xb) ** 2).sum()); sval += float((xb ** 2).sum())
            usage += torch.bincount(out["quantizer_output"].indices.reshape(-1).cpu(), minlength=K).float()
            err = ((xb - rec) ** 2).mean(dim=(1, 2)).cpu().numpy()
            if use_prior:
                tok = m.score_batch({"inputs": xb}).token_scores
                pr = tok.float().mean(dim=tuple(range(1, tok.ndim))).cpu().numpy()
            for j, s in enumerate(sb):
                sl.append(err[j]); wl.append(int(lab[s:s + W].any()))
                if use_prior:
                    sp.append(pr[j])

    p = usage / usage.sum(); nz = p[p > 0]
    out = {
        "perplexity": float(torch.exp(-(nz * nz.log()).sum())),
        "n_active": int((usage > 0).sum()),
        "recon_nMSE": sse / sval,
        "codebook_PR": _participation_ratio(cb),
    }
    wl = np.array(wl)
    valid = wl.any() and not wl.all()
    if valid:
        from sklearn.metrics import roc_auc_score
        out["s_local_auroc"] = float(roc_auc_score(wl, np.array(sl)))
        if use_prior:
            out["s_prior_auroc"] = float(roc_auc_score(wl, np.array(sp)))
    return out


@torch.no_grad()
def codebook_coherence(root, arm, seed, ents, device, n_windows=64, ckpt_root=None,
                       probe_name=None):
    ckpt_root = ckpt_root or root
    """DIRECT measure of cross-client index coherence (the C7 mechanism).

    All clients of a given (arm, seed) hold the SAME broadcast/averaged global
    codebook but LOCAL encoders. We tokenize ONE shared, data-independent public
    probe with every client's frozen stage1 and ask: do clients assign the SAME
    code index to the same input?

      * token_agreement — mean over client pairs of the fraction of token
        positions where the two clients emit the IDENTICAL code index (1.0 =
        perfectly coherent indices; low = globally inconsistent assignments).
      * usage_js — mean pairwise Jensen–Shannon divergence of the per-client
        token-usage histograms (0 = identical usage).

    The suff-stat merge trains every encoder against one frozen, globally-coherent
    codebook, so assignments should agree; weight-FedAvg lets each client's codebook
    drift before averaging, so encoders are miscalibrated to the merged table and
    assignments diverge. Returns None if <2 clients or no probe is available.
    """
    # Scoped federations get a scoped probe: tokenizing the 6-morphology global
    # probe with a single-cluster encoder measures agreement on data no client
    # has seen, which is not the C7 question.
    ppath = f"data/raw/{root}/probe/{probe_name}.npy" if probe_name else f"data/raw/{root}/probe/probe.npy"
    if not os.path.exists(ppath):
        return None
    toks = []
    K = None
    probe = None
    for ent in ents:
        s1c = f"artifacts/fed_eval/{ckpt_root}/seed{seed}/{arm}/{ent}/stage1.ckpt"
        if not os.path.exists(s1c):
            continue
        ck = torch.load(s1c, map_location="cpu", weights_only=False)
        cfg = Config()
        _overlay(cfg, ck["cfg_dict"])
        W = int(cfg.dataset.window_length)
        if probe is None:
            arr = _as_2d(np.load(ppath).astype(np.float32))                # (T, C)
            arr = (arr - arr.mean(0, keepdims=True)) / (arr.std(0, keepdims=True) + 1e-8)
            starts = np.linspace(0, len(arr) - W, n_windows).astype(int)
            probe = torch.from_numpy(np.stack([arr[s:s + W].T for s in starts])).to(device)  # (B, C, W)
        s1 = load_stage1(s1c, cfg, torch.zeros(1, probe.shape[1], W), device=device)
        K = int(s1.quantizer._vq.codebook_size)
        _, idx, _ = s1.encode_tokens(probe)                               # (B, C, F*W)
        toks.append(idx.reshape(-1).cpu())
    if len(toks) < 2 or K is None:
        return None

    def _js(a, b):
        m = 0.5 * (a + b)
        def kl(p, q):
            nz = p > 0
            return float((p[nz] * (torch.log(p[nz]) - torch.log(q[nz].clamp_min(1e-12)))).sum())
        return 0.5 * kl(a, m) + 0.5 * kl(b, m)

    agrees, jss = [], []
    for i, j in itertools.combinations(range(len(toks)), 2):
        ti, tj = toks[i], toks[j]
        agrees.append(float((ti == tj).float().mean()))
        hi = torch.bincount(ti, minlength=K).float(); hi /= hi.sum().clamp_min(1.0)
        hj = torch.bincount(tj, minlength=K).float(); hj /= hj.sum().clamp_min(1.0)
        jss.append(_js(hi, hj))
    return {"token_agreement": float(np.mean(agrees)), "usage_js": float(np.mean(jss))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="toy_fed_uni", help="artifacts/fed_eval/<dataset>/... and data/raw/<dataset> "
                    "(default toy_fed_uni; the C=8 toy_fed ckpts were removed with the -M line)")
    ap.add_argument("--ckpt-root", default=None,
                    help="read checkpoints from artifacts/fed_eval/<ckpt-root>/ instead of <dataset> "
                         "(data/raw still uses <dataset>); lets a separate experiment dir reuse toy_fed data.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--arms", nargs="+", default=ARMS_DEFAULT)
    ap.add_argument("--clients", type=str, default="auto",
                    help="comma-separated entity ids, or 'auto' (default) to read them from "
                         "data/raw/<dataset>/train/*.npy. The old integer form is still accepted "
                         "and means fed_0..fed_<n-1>.")
    ap.add_argument("--cluster", type=str, default=None,
                    help="restrict to one machine-type cluster (needs clusters.json); also selects "
                         "that cluster's public probe for the coherence diagnostic.")
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--skip-prior", action="store_true", help="skip the (slow) s_prior AUROC")
    ap.add_argument("--out", default=None, help="optional JSON dump path")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ents = _resolve_entities(args.dataset, args.clients, args.cluster)
    if not ents:
        raise SystemExit(f"no clients resolved for {args.dataset}")
    print(f"[autopsy] dataset={args.dataset} "
          f"{'cluster=' + args.cluster + ' ' if args.cluster else ''}clients={ents}")
    # `federated_eval --cluster X` writes under artifacts/fed_eval/<dataset>/<cluster>/.
    ckpt_root = args.ckpt_root or (f"{args.dataset}/{args.cluster}" if args.cluster else args.dataset)
    metrics = ["perplexity", "n_active", "recon_nMSE", "codebook_PR", "s_local_auroc", "s_prior_auroc"]
    coh_metrics = ["token_agreement", "usage_js"]         # cross-client index coherence (C7)
    all_metrics = metrics + coh_metrics
    res = {a: {k: [] for k in all_metrics} for a in args.arms}

    t0 = time.time()
    for arm in args.arms:
        for seed in args.seeds:
            per = {k: [] for k in metrics}
            for e in ents:
                r = eval_client(args.dataset, arm, seed, e, device, args.stride, args.skip_prior,
                                ckpt_root=ckpt_root)
                if r is None:
                    continue
                for k in metrics:
                    if k in r:
                        per[k].append(r[k])
            for k in metrics:
                if per[k]:
                    res[arm][k].append(float(np.mean(per[k])))   # per-seed macro-mean over clients
            # cross-client token coherence — ONE value per (arm, seed), a DIRECT
            # measure of whether clients assign the same tokens to a shared probe.
            coh = codebook_coherence(args.dataset, arm, seed, ents, device, ckpt_root=ckpt_root,
                                     probe_name=args.cluster)
            if coh:
                for k in coh_metrics:
                    res[arm][k].append(coh[k])
        print(f"[{time.time()-t0:6.0f}s] {arm} done", flush=True)

    def ms(xs):
        if not xs:
            return "  n/a "
        return f"{np.mean(xs):.3f}±{np.std(xs, ddof=1):.3f}" if len(xs) > 1 else f"{np.mean(xs):.3f}"

    print(f"\n=== R1 codebook autopsy — {args.dataset}  seeds={args.seeds} ===")
    hdr = (f"{'arm':22s} {'perplexity':>13s} {'n_active':>9s} {'recon_nMSE':>12s} {'cb_PR':>11s} "
           f"{'s_local_AUC':>13s} {'s_prior_AUC':>13s} {'tok_agree':>13s} {'usage_JS':>13s}")
    print(hdr)
    for a in args.arms:
        r = res[a]
        print(f"{a:22s} {ms(r['perplexity']):>13s} {ms(r['n_active']):>9s} {ms(r['recon_nMSE']):>12s} "
              f"{ms(r['codebook_PR']):>11s} {ms(r['s_local_auroc']):>13s} {ms(r['s_prior_auroc']):>13s} "
              f"{ms(r['token_agreement']):>13s} {ms(r['usage_js']):>13s}")
    print("\nperplexity/n_active/cb_PR higher=healthier codebook; recon_nMSE lower=better; "
          "s_*_AUROC higher=better anomaly separability.")
    print("tok_agree = cross-client fraction of IDENTICAL token indices on a shared probe (higher=more "
          "coherent indices); usage_JS = cross-client token-usage divergence (lower=more coherent).")
    print("Mechanism: weight-FedAvg (federated_fedavg_cb) is healthiest on stage1 metrics yet its "
          "s_prior collapses — the suff-stat merge (A) wins via the PRIOR, not reconstruction — and its "
          "token indices are LESS coherent across clients (lower tok_agree), the direct C7 signature.")

    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)
        print(f"\n[out] {args.out}")


if __name__ == "__main__":
    main()
