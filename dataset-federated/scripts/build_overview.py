"""Build the dashboard overview for the canonical dataset (data/federated/WSD_frozen).

All 210 WSD KPIs get a full-series sparkline. Clients carry their NaN-free WINDOW
(win_start..win_end), split, anomaly spans, and their TRAIN-ONLY cluster (from the manifest).
Non-client series are assigned to the nearest train-cluster centroid (context only).
`nan_total` = missing points inside the CLIENT window (0 for this NaN-free dataset).

Output: dashboard/overview.json
"""
import os, sys, glob, json
from collections import Counter
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import day_profile, deviation_features

# Repo-relative: this file lives in <dataset-federated>/scripts/, so the corpus root
# is its parent. (Was a hardcoded Windows path — broken on Linux.)
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, "data", "WSD", "real-world")
FROZEN = os.path.join(BASE, "data", "federated", "WSD_frozen")
OUT = os.path.join(BASE, "dashboard", "overview.json")
L = 256; K = 4; P = 0.50; SPARK = 220

def zr(y):
    y = np.asarray(y, float)
    if np.isnan(y).any():
        xi = np.arange(len(y)); g = ~np.isnan(y)
        y = np.interp(xi, xi[g], y[g]) if g.sum() >= 2 else np.zeros(len(y))
    r = np.interp(np.linspace(0, 1, L), np.linspace(0, 1, len(y)), y)
    return (r - r.mean()) / (r.std() + 1e-9)

def spans(mask):
    mask = np.asarray(mask).astype(bool); out = []; i = 0; n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]: j += 1
            out.append([int(i), int(j)]); i = j + 1
        else: i += 1
    return out

def decimate(v, B):
    n = len(v)
    if n <= B: return [None if (x != x) else round(float(x), 4) for x in v]
    idx = np.linspace(0, n, B + 1).astype(int); out = []
    for a, b in zip(idx[:-1], idx[1:]):
        seg = v[a:b]; m = seg[~np.isnan(seg)]
        out.append(None if m.size == 0 else round(float(m.mean()), 4))
    return out

S = []
for f in sorted(glob.glob(os.path.join(RAW, "*.csv")), key=lambda p: int(os.path.basename(p)[:-4])):
    d = pd.read_csv(f)
    S.append((int(os.path.basename(f)[:-4]), d["value"].values.astype(np.float32),
              d["label"].fillna(0).astype(int).values, int(d["timestamp"].values[0])))

man = pd.read_csv(os.path.join(FROZEN, "manifest.csv"))
cmap = {int(r.id): r for r in man.itertuples()}
Z = np.load(os.path.join(FROZEN, "wsd_federated.npz"))
# authoritative per-series selection outcome, written by build_frozen.py
_sel = pd.read_csv(os.path.join(FROZEN, "selection.csv"))
SEL = {int(r.id): (r.status, "" if pd.isna(r.detail) else str(r.detail)) for r in _sel.itertuples()}

# centroids in the SAME space used to cluster (average-day deviation profile, see features.py).
# `profile_mean.npy` is written by build_frozen.py so the transform is bit-for-bit the same.
MEAN_PROFILE = np.load(os.path.join(FROZEN, "profile_mean.npy"))
t0_of = dict(zip(man.id, man.t0_unix))
_feat = {int(sid): deviation_features([day_profile(Z[f"{int(sid)}_train"], t0_of[sid])], MEAN_PROFILE)[0][0]
         for sid in man.id}
cent = {int(c): np.mean([_feat[int(sid)] for sid in man[man.cluster == c].id], axis=0)
        for c in sorted(man.cluster.unique())}
cids = sorted(cent); cmat = np.array([cent[c] for c in cids])

recs = []
for sid, v, lb, t0 in S:
    n = len(v); na = int((lb == 1).sum())
    rec = {"id": sid, "orig_n": n, "spark": decimate(v, SPARK)}
    if sid in cmap:
        r = cmap[sid]; a = int(r.win_start); b = int(r.win_end); sp = int(r.split_abs)
        m = np.zeros(n, bool); m[a:b] = (lb[a:b] == 1)
        rec.update(cluster=int(r.cluster), viable=True, win_start=a, win_end=b, split=sp,
                   spans=spans(m), nan_total=int(np.isnan(v[a:b]).sum()))
    else:
        f = deviation_features([day_profile(v, t0)], MEAN_PROFILE)[0][0]   # context only, not a client
        nearest = int(cids[int(np.argmin(((cmat - f) ** 2).sum(axis=1)))])
        rec.update(cluster=nearest, viable=False, reason=SEL[sid][0], reason_detail=SEL[sid][1], nan_total=0)
    recs.append(rec)

rc = Counter(r.get("reason") for r in recs if not r["viable"])
meta = {"total": len(recs), "frozen": int(sum(r["viable"] for r in recs)), "split_frac": P, "k": K,
        "per_cluster": {int(c): {"total": int(sum(1 for r in recs if r["cluster"] == c)),
                                 "frozen": int(sum(1 for r in recs if r["cluster"] == c and r["viable"]))}
                        for c in cids},
        "reasons": {k: int(v) for k, v in rc.items() if k}}
json.dump({"meta": meta, "series": recs}, open(OUT, "w"))
print(f"wrote {OUT}: {meta['frozen']} frozen / {meta['total']}  per_cluster={ {c:o['frozen'] for c,o in meta['per_cluster'].items()} }")
