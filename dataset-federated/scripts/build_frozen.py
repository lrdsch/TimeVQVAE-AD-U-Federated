"""Canonical builder for the frozen WSD federated dataset (single final version).

Selection pipeline (deterministic, seed 0):
  1. For each of the 210 WSD KPIs, find the LONGEST contiguous NaN-free window [a,b) such that no
     labelled anomaly falls in [a-GUARD, a+split): the train part is anomaly-free AND `a` sits at
     least GUARD points after the last incident. The look-back is on the ORIGINAL series, so a NaN
     gap cannot hide a recent anomaly (a KPI often goes missing *because* it is broken).
     So a KPI is not discarded just for misbehaving early, but train never starts inside a recovery.
     Split at a fixed 50%: train part = clean of anomalies & NaN, test = the rest (>=1 anomaly).
  2. Carve the LAST VAL_LEN samples of the train part off as a validation split (anomaly- & NaN-free).
     Model selection / early stopping must use `val`, never `test`. VAL_LEN is a fixed DURATION of
     exactly one seasonal period (1 day), so early stopping is scored over a whole diurnal cycle
     rather than a slice of it. Train-proper must span >= 2 periods.
  3. Deduplicate clients on their TIME-ALIGNED overlap (not on resampled shape):
     merge if |corr| > CORR_THRESH (same underlying series) OR label-Jaccard > JACC_THRESH
     (same incident). Keep the longest-window representative per group.
  4. Cluster the FINAL client set on their TRAIN portion ONLY (KMeans k=4 on the average-day
     deviation profile, see features.py) - the grouping never sees val/test.
Writes ONE dataset: data/federated/WSD_frozen.
Prints the selection funnel, per-cluster counts, intra/inter similarity, and residual correlation.

Run: python scripts/build_frozen.py     (env TimeEnvM)
"""
import os, sys, glob, shutil, itertools
from collections import Counter, defaultdict
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import day_profile, deviation_features, acf_at   # see features.py

# Repo-relative: this file lives in <dataset-federated>/scripts/, so the corpus root
# is its parent. (Was a hardcoded Windows path — broken on Linux.)
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, "data", "WSD", "real-world")
FED = os.path.join(BASE, "data", "federated")

L = 256          # resample length, kept only for the dedup shape check
K = 4            # largest k with no singleton cluster (== local-only in disguise) AND a significant
                 # same-service purity. k=4: sizes [5,6,9,11], sil 0.53, purity 88% (7/8) p<1e-4.
                 # k=5 is singleton-free only at seed 0 but a singleton under 4/5 seeds; k=6 singleton at seed 0.
P = 0.50         # train-part fraction of the client window (label-independent given the window)
MINFRAC = 0.10   # each side must hold >= 10% of the window
DT = 60          # WSD sampling interval, seconds
PERIOD = 1440    # the seasonal period, in samples = 24 h. Measured, not assumed: ACF and a zero-padded
                 # periodogram agree on 1440 +-5% for 37/38 clients where >=2 cycles are observable
                 # (median exactly 1440). See scripts/detect_periods.py.
# The model window is SHORTER than the period (128-256 samples = 2-4 h). Making it a whole period is
# not affordable: the repo rule window = 2*period needs a 2-day window and leaves 14 clients with ~15
# training windows, and down-sampling to shrink the period is ruled out by the labels (anomaly segments
# are 11 min median, 3 min min: a 5-min bin destroys 24% of them, a 10-min bin 42%).
# Two period-derived constraints survive anyway, for reasons independent of the window:
VAL_LEN = PERIOD     # `val` must span exactly one full diurnal cycle, or early stopping is scored on
                     # a slice of the day (e.g. only the night). Fixed DURATION, not a fraction.
MIN_TRAIN = 2 * PERIOD   # train must show the diurnal pattern repeat, or the average-day descriptor
                         # (features.py) that the clustering is built on has nothing to average.
GUARD = PERIOD   # 1 day: recovery gap after an early anomaly before train may start
MIN_PERIOD_ACF = 0.20    # the client must actually EXHIBIT a daily cycle. Not a window requirement:
                         # the CLUSTERING descriptor *is* the average day, so a client without a stable
                         # day has a meaningless descriptor (it was also the lone singleton cluster).
CORR_THRESH = 0.99   # |pearson| on the time-aligned overlap  -> same underlying series
JACC_THRESH = 0.50   # anomaly-label Jaccard on the overlap   -> same incident
MIN_OVERLAP = 500    # minimum aligned points to even compare two clients
STRAIGHT_MIN = 32    # >= this many samples on a perfectly constant first difference -> imputed, mask as NaN
MAX_ANOM_RATE = 0.10 # a test window that is >10% anomalous is a regime change, not an anomaly: "normal"
                     # stops being the majority class and the client is unusable for unsupervised AD.
                     # Safety rail, not a tuner: the observed rates are median 0.6%, 3rd quartile 1.3%.
SEED = 0

def zr(y):
    """resample to L points + z-normalise (shape descriptor)"""
    y = np.asarray(y, float)
    if np.isnan(y).any():
        xi = np.arange(len(y)); g = ~np.isnan(y)
        y = np.interp(xi, xi[g], y[g]) if g.sum() >= 2 else np.zeros(len(y))
    r = np.interp(np.linspace(0, 1, L), np.linspace(0, 1, len(y)), y)
    return (r - r.mean()) / (r.std() + 1e-9)

def straight_mask(v, minlen=STRAIGHT_MIN):
    """Points on a run of >= minlen samples with a perfectly constant first difference.
    WSD ships these with a real (non-NaN) value and label=0, but a >=32-minute exactly-linear
    ramp (or an exactly-flat plateau) is upstream imputation / a dead sensor, not observed
    behaviour. Left in, `train` would teach the model that a perfect straight line is normal."""
    m = np.zeros(len(v), bool)
    if len(v) < 3: return m
    d = np.diff(v)
    same = np.isclose(d[1:], d[:-1], rtol=0, atol=1e-9)   # comparisons with NaN are False -> runs break at gaps
    i = 0
    while i < len(same):
        if same[i]:
            j = i
            while j + 1 < len(same) and same[j + 1]: j += 1
            if (j + 3 - i) >= minlen: m[i:j + 3] = True   # same[i..j] collinear <=> v[i..j+2] collinear
            i = j + 1
        else: i += 1
    return m

def notnan_runs(v):
    m = ~np.isnan(v); runs = []; i = 0; n = len(m)
    while i < n:
        if m[i]:
            j = i
            while j + 1 < n and m[j + 1]: j += 1
            runs.append((i, j + 1)); i = j + 1
        else: i += 1
    return runs

def best_window(lb, runs, max_rate=MAX_ANOM_RATE):
    """Longest NaN-free window [a,b) such that no labelled anomaly falls in [a-GUARD, a+split).
    i.e. the train part is anomaly-free AND `a` sits at least GUARD points after the last
    incident (measured on the ORIGINAL series, so a NaN gap does not hide a recent anomaly).
    The test part must hold >=1 anomaly but stay below `max_rate` anomalous.
    Returns (a, split_abs, b) or None."""
    best = None
    for a0, b in runs:
        anom = np.flatnonzero(lb[:b] == 1)   # all anomalies before b: the guard looks back past a0
        cands = sorted({a0, *(int(x) + 1 + GUARD for x in anom if int(x) + 1 + GUARD > a0)})
        for a in cands:                       # ascending => first valid `a` is the longest window
            if a >= b: continue
            Lw = b - a; split = round(P * Lw)
            # the train part must hold a fixed-length `val` plus >= MIN_TRAIN of train-proper
            if split < VAL_LEN + MIN_TRAIN: continue
            if split < MINFRAC * Lw or (Lw - split) < MINFRAC * Lw: continue
            if (lb[max(0, a - GUARD):a + split] == 1).sum() != 0: continue   # guard + clean train
            n_anom = int((lb[a + split:b] == 1).sum())
            if n_anom < 1 or n_anom > max_rate * (Lw - split): continue      # >=1 anomaly, but a minority
            if best is None or Lw > (best[2] - best[0]): best = (a, a + split, b)
            break
    return best

def segments(lb):
    return int((np.diff(np.r_[0, np.asarray(lb).astype(int), 0]) == 1).sum())

# ---- 0. load raw, masking imputed straight-line segments as NaN ----
S = {}; n_masked = {}
for f in sorted(glob.glob(os.path.join(RAW, "*.csv")), key=lambda p: int(os.path.basename(p)[:-4])):
    d = pd.read_csv(f); sid = int(os.path.basename(f)[:-4])
    v = d["value"].values.astype(np.float64)
    sm = straight_mask(v); v[sm] = np.nan; n_masked[sid] = int(sm.sum())
    S[sid] = (d["timestamp"].values.astype(np.int64), v, d["label"].fillna(0).astype(int).values)
print(f"masked {sum(n_masked.values()):,} imputed points (>= {STRAIGHT_MIN}-sample constant slope) "
      f"across {sum(1 for n in n_masked.values() if n):d}/210 raw series")

# ---- 1. viable windows ----
clients = []; funnel = Counter(); status = {}   # sid -> (status, detail) for every one of the 210
for sid, (ts, v, lb) in S.items():
    runs = notnan_runs(v)
    w = best_window(lb, runs)
    if not w:
        if (lb == 1).sum() == 0: reason = "no_anomaly"
        # distinguish "no clean window at all" from "only anomaly-dominated windows exist"
        elif best_window(lb, runs, max_rate=1.0): reason = "anomaly_dominated"
        else: reason = "no_clean_window"
        funnel[reason] += 1; status[sid] = (reason, "")
        continue
    a, sp, b = w
    n_val = VAL_LEN; tr_end = sp - n_val
    # the clustering descriptor is the average day; admit only clients that actually have one
    period_acf = acf_at(v[a:tr_end], PERIOD)
    if not (period_acf >= MIN_PERIOD_ACF):
        funnel["no_period"] += 1; status[sid] = ("no_period", f"acf@{PERIOD}={period_acf:.3f}")
        continue
    clients.append(dict(id=sid, win_start=a, win_end=b, split_abs=sp, val_start_abs=tr_end,
                        period_acf=round(period_acf, 3),
                        train=v[a:tr_end].astype(np.float32), val=v[tr_end:sp].astype(np.float32),
                        test=v[sp:b].astype(np.float32), tl=(lb[sp:b] == 1).astype(np.int8),
                        train_len=tr_end - a, val_len=n_val, test_len=b - sp, win_len=b - a,
                        test_anom=int((lb[sp:b] == 1).sum()), test_segs=segments(lb[sp:b] == 1),
                        t0_unix=int(ts[a]), orig_n=len(v)))
n_viable = len(clients)

# ---- 2. dedup on the TIME-ALIGNED overlap ----
def overlap_stats(ci, cj):
    ti, vi, li = S[ci["id"]]; tj, vj, lj = S[cj["id"]]
    Ti = ti[ci["win_start"]:ci["win_end"]]; Tj = tj[cj["win_start"]:cj["win_end"]]
    com, ii, jj = np.intersect1d(Ti, Tj, return_indices=True)
    if len(com) < MIN_OVERLAP: return None
    x = vi[ci["win_start"]:ci["win_end"]][ii]; y = vj[cj["win_start"]:cj["win_end"]][jj]
    lx = li[ci["win_start"]:ci["win_end"]][ii]; ly = lj[cj["win_start"]:cj["win_end"]][jj]
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < MIN_OVERLAP: return None
    sx, sy = x[ok].std(), y[ok].std()
    r = float(abs(np.corrcoef(x[ok], y[ok])[0, 1])) if sx > 1e-9 and sy > 1e-9 else 0.0
    uni = int(((lx == 1) | (ly == 1)).sum())
    jac = float(((lx == 1) & (ly == 1)).sum() / uni) if uni else 0.0
    return int(ok.sum()), r, jac

n = len(clients); par = list(range(n))
def find(a):
    while par[a] != a: par[a] = par[par[a]]; a = par[a]
    return a
allpairs = []
for i, j in itertools.combinations(range(n), 2):
    st = overlap_stats(clients[i], clients[j])
    if st is None: continue
    ov, r, jac = st
    dup = (r > CORR_THRESH) or (jac > JACC_THRESH)
    allpairs.append((clients[i]["id"], clients[j]["id"], ov, round(r, 4), round(jac, 4), int(dup)))
    if dup: par[find(i)] = find(j)
grp = defaultdict(list)
for i in range(n): grp[find(i)].append(i)
drop = set(); merged = []
for members in grp.values():
    if len(members) == 1: continue
    keep = max(members, key=lambda i: clients[i]["win_len"])
    merged.append((clients[keep]["id"], sorted(clients[i]["id"] for i in members if i != keep)))
    drop |= {i for i in members if i != keep}
    for i in members:
        if i != keep: status[clients[i]["id"]] = ("near_duplicate", f"of {clients[keep]['id']}")
removed_ids = sorted(clients[i]["id"] for i in drop)
clients = [c for i, c in enumerate(clients) if i not in drop]

print(f"SELECTION FUNNEL: 210 KPIs -> -{funnel['no_anomaly']} no-anomaly -> {210 - funnel['no_anomaly']} "
      f"-> -{funnel['no_clean_window']} no-clean-window -> -{funnel['anomaly_dominated']} anomaly-dominated "
      f"(> {MAX_ANOM_RATE:.0%} of test) -> -{funnel['no_period']} no-period "
      f"(acf@{PERIOD} < {MIN_PERIOD_ACF}) -> {n_viable} viable "
      f"-> -{len(drop)} near-duplicate -> {len(clients)} clients")
print(f"  dedup rule: |corr|>{CORR_THRESH} (same series) OR label-Jaccard>{JACC_THRESH} (same incident),")
print(f"              on >= {MIN_OVERLAP} time-aligned points. Removed: {removed_ids}")
for keep_id, gone in merged: print(f"    keep {keep_id:>4}  drop {gone}")

# ---- 3. TRAIN-ONLY clustering, on the average-day deviation descriptor (see features.py) ----
profiles = np.array([day_profile(c["train"], c["t0_unix"]) for c in clients])
train_feat, mean_profile = deviation_features(profiles)
labels = KMeans(K, n_init=10, random_state=SEED).fit_predict(train_feat)
for c, l in zip(clients, labels): c["cluster"] = int(l)
sil = silhouette_score(train_feat, labels)
U = train_feat / np.linalg.norm(train_feat, axis=1, keepdims=True)
C = U @ U.T
iu = np.triu_indices(len(clients), 1); sm = np.array(labels)[iu[0]] == np.array(labels)[iu[1]]
print(f"\nTRAIN-ONLY clustering (k={K}, seed {SEED}) on the average-day deviation profile: "
      f"silhouette={sil:+.3f}  intra_r={C[iu][sm].mean():.2f}  inter_r={C[iu][~sm].mean():.2f}")

# independent sanity check: KPIs that are >0.9 correlated in WALL-CLOCK time are metrics of the same
# service. The clustering never sees cross-client alignment, so where they land is a real test.
_lab = {c["id"]: c["cluster"] for c in clients}
_hi = [(a, b) for a, b, ov, r, j, dup in allpairs if not dup and r > 0.9 and a in _lab and b in _lab]
if _hi:
    _same = sum(1 for a, b in _hi if _lab[a] == _lab[b])
    _sizes = np.bincount(labels, minlength=K)
    _base = (_sizes ** 2).sum() / len(labels) ** 2
    print(f"  validation: {_same}/{len(_hi)} = {_same/len(_hi):.0%} of same-service pairs (|corr|>0.9) "
          f"share a cluster, vs {_base:.0%} expected by chance")

# ---- 4. write ----
DST = os.path.join(FED, "WSD_frozen")
if os.path.isdir(DST): shutil.rmtree(DST)
os.makedirs(os.path.join(DST, "clients"), exist_ok=True)
big = {}; rows = []
for c in clients:
    sid = c["id"]
    cdir = os.path.join(DST, "clients", f"cluster_{c['cluster']}"); os.makedirs(cdir, exist_ok=True)
    np.savez_compressed(os.path.join(cdir, f"{sid}.npz"),
        train=c["train"], val=c["val"], test=c["test"], test_label=c["tl"],
        train_label=np.zeros(len(c["train"]), np.int8), val_label=np.zeros(len(c["val"]), np.int8),
        cluster=c["cluster"], series_id=sid, t0_unix=c["t0_unix"], dt_sec=DT,
        win_start=c["win_start"], win_end=c["win_end"], split_abs=c["split_abs"],
        val_start_abs=c["val_start_abs"], orig_n=c["orig_n"])
    big[f"{sid}_train"] = c["train"]; big[f"{sid}_val"] = c["val"]
    big[f"{sid}_test"] = c["test"]; big[f"{sid}_test_label"] = c["tl"]
    rows.append(dict(id=sid, cluster=c["cluster"], win_start=c["win_start"], win_end=c["win_end"],
                     val_start_abs=c["val_start_abs"], split_abs=c["split_abs"],
                     train_len=c["train_len"], val_len=c["val_len"], test_len=c["test_len"],
                     win_len=c["win_len"], test_anom=c["test_anom"], test_segs=c["test_segs"],
                     anom_rate=round(c["test_anom"] / c["test_len"], 6),
                     train_days=round(c["train_len"] / 1440, 2), test_days=round(c["test_len"] / 1440, 2),
                     t0_unix=c["t0_unix"], dt_sec=DT, orig_n=c["orig_n"],
                     period_min=PERIOD, period_acf=c["period_acf"],
                     interp_masked_orig=n_masked[sid]))
m = pd.DataFrame(rows).sort_values(["cluster", "id"])
big["ids"] = m.id.values.astype(np.int32); big["clusters"] = m.cluster.values.astype(np.int32)
big["t0_unix"] = m.t0_unix.values.astype(np.int64); big["dt_sec"] = np.int32(DT)
np.savez_compressed(os.path.join(DST, "wsd_federated.npz"), **big)
m.to_csv(os.path.join(DST, "manifest.csv"), index=False)

# per-series selection outcome for ALL 210 (the dashboard/overview reads this — never re-derive it)
for c in clients: status[c["id"]] = ("client", f"cluster {c['cluster']}")
pd.DataFrame([dict(id=s, status=st, detail=d) for s, (st, d) in sorted(status.items())]) \
  .to_csv(os.path.join(DST, "selection.csv"), index=False)
assert len(status) == 210, f"selection.csv covers {len(status)}/210 series"

# residual cross-client correlation among the FINAL clients (for the effective-n discussion)
final = set(m.id)
cor = pd.DataFrame([p for p in allpairs if p[0] in final and p[1] in final],
                   columns=["a", "b", "overlap", "abs_corr", "label_jaccard", "removed_as_dup"])
cor.sort_values("abs_corr", ascending=False).to_csv(os.path.join(DST, "correlations.csv"), index=False)

tot_anom = int(m.test_anom.sum())
open(os.path.join(DST, "README.md"), "w", encoding="utf-8").write(
    f"# WSD federated dataset - FROZEN\n\n"
    f"{len(m)} clients. Each = the longest contiguous NaN-free window `[a, b)` of a WSD KPI (one of 210\n"
    f"separate series) such that NO labelled anomaly falls in `[a - GUARD, a + split)`, split at {P:.0%}.\n\n"
    f"- `train` = first {P:.0%} minus the last {VAL_LEN} samples -> anomaly- & NaN-free, "
    f">= {MIN_TRAIN/PERIOD:.0f} periods ({MIN_TRAIN/1440:.0f} days)\n"
    f"- `val`   = last {VAL_LEN} samples = exactly one period ({VAL_LEN/1440:.0f} day) of the train part -> "
    f"anomaly- & NaN-free.\n"
    f"  **Use for model selection / early stopping. Never touch `test`.** Fixed duration, not a fraction,\n"
    f"  so early stopping is scored over a whole diurnal cycle instead of a slice of it.\n"
    f"- `test`  = remaining {1-P:.0%} -> contains >= 1 anomaly segment, but is < {MAX_ANOM_RATE:.0%} anomalous\n"
    f"  (a test that is mostly anomalous is a regime change, not an anomaly)\n\n"
    f"So the train part is anomaly-free AND the window starts at least GUARD = {GUARD/1440:.0f} day after the last\n"
    f"incident: a KPI is not discarded just for misbehaving early, but training never begins inside a\n"
    f"post-incident recovery. The GUARD look-back runs on the ORIGINAL series, not on the NaN-free run --\n"
    f"a KPI often goes missing *because* it broke, so a NaN gap must not hide a recent anomaly.\n"
    f"The split point is a fixed fraction of the window.\n\n"
    f"Before any of this, runs of >= {STRAIGHT_MIN} samples with a perfectly constant first difference are\n"
    f"masked as NaN: WSD ships them with a real value and label=0, but an exactly-linear ramp (or\n"
    f"exactly-flat plateau) lasting >= {STRAIGHT_MIN} minutes is upstream imputation / a dead sensor. Left in, they\n"
    f"would teach the model that a perfect straight line is normal. Masked: {sum(n_masked.values()):,} points over\n"
    f"{sum(1 for n in n_masked.values() if n)}/210 raw series (see `interp_masked_orig` in the manifest).\n\n"
    f"**Selection funnel:** 210 -> -{funnel['no_anomaly']} no-anomaly -> {210-funnel['no_anomaly']} -> "
    f"-{funnel['no_clean_window']} no-clean-window -> -{funnel['anomaly_dominated']} anomaly-dominated "
    f"-> -{funnel['no_period']} no-period -> {n_viable} viable -> -{len(drop)} near-duplicate "
    f"(|corr|>{CORR_THRESH} or label-Jaccard>{JACC_THRESH} on the time-aligned overlap; removed {removed_ids}) "
    f"-> {len(m)} clients.\n\n"
    f"**Clusters computed on the TRAIN portion only** (k={K}, seed {SEED}) - the grouping never sees val/test.\n"
    f"The descriptor is the client's *average day* (24 hourly bins of log1p values, wall-clock aligned via\n"
    f"`t0_unix`) minus the corpus-mean average day - see `scripts/features.py`. k={K} is the largest\n"
    f"singleton-free, seed-stable k (k=5 is a singleton under 4/5 seeds, k=6 at seed 0); a lone client is\n"
    f"just local-only in disguise.\n\n"
    f"Anomalous points: {tot_anom} total, largest single client holds {m.test_anom.max()/tot_anom:.1%}.\n"
    f"{int((m.test_segs==1).sum())}/{len(m)} clients have a single anomaly segment.\n\n"
    f"Files: `wsd_federated.npz` (`<id>_train/_val/_test/_test_label`, `ids`, `clusters`, `t0_unix`, `dt_sec`),\n"
    f"`clients/cluster_<c>/<id>.npz`, `manifest.csv`, `correlations.csv` (residual cross-client correlation).\n"
    f"Absolute time of point i of a client: `t0_unix + (i + offset) * dt_sec`.\n\n"
    f"Regenerate: `python scripts/build_frozen.py`.\n")

np.save(os.path.join(DST, "profile_mean.npy"), mean_profile)   # build_overview reuses the exact transform

print(f"\n{'cluster':8} {'n':>3} {'intra_r':>8}  {'anom%':>6}   ids")
for c in sorted(m.cluster.unique()):
    ids = list(m[m.cluster == c].id)
    F = U[[i for i, cc in enumerate(clients) if cc["id"] in ids]]
    Cc = F @ F.T; iuc = np.triu_indices(len(F), 1)
    ir = Cc[iuc].mean() if len(F) > 1 else float("nan")
    share = m[m.cluster == c].test_anom.sum() / tot_anom
    print(f"c{c:<7} {len(ids):>3} {ir:>8.2f}  {share:>6.1%}   {ids}")

print(f"\ntrain days: median {m.train_days.median():.2f}  min {m.train_days.min():.2f}  max {m.train_days.max():.2f}")
print(f"anomaly points {tot_anom} | top-1 client {m.test_anom.max()/tot_anom:.1%} | "
      f"top-3 {m.test_anom.nlargest(3).sum()/tot_anom:.1%} | rate median {m.anom_rate.median():.4f} max {m.anom_rate.max():.4f}")
print(f"single-segment clients: {int((m.test_segs==1).sum())}/{len(m)}")
res = cor[cor.removed_as_dup == 0]
print(f"residual pairs w/ overlap>={MIN_OVERLAP}: {len(res)} | |corr|>0.9: {(res.abs_corr>0.9).sum()} | "
      f"label-Jaccard>0.3: {(res.label_jaccard>0.3).sum()}  -> correlations.csv")
print(f"\nwrote single dataset -> {DST}  ({len(m)} clients)")
