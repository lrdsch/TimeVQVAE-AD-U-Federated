"""Client shape descriptor used to group WSD KPIs into similarity clusters.

Shared by build_frozen.py (which defines the clusters) and build_overview.py (which places the
non-client series next to the nearest centroid, for context only). Keep them identical.

WHY NOT the obvious thing. The first version resampled the whole train window to 256 points and
z-normalised it. That descriptor is meaningless:
  * it is indexed by *position in the window*, so it mixes clients whose train spans 1.2 days with
    clients spanning 10.8 days — 256 samples cover a different number of daily cycles each;
  * clusters built on it had silhouette +0.14, ARI 0.51 across seeds, and — measured on any OTHER
    descriptor — an intra-minus-inter cohesion gap of ~0 (+0.05 on the daily profile, +0.001 on the
    autocorrelation). They only separated data in the metric that produced them.

WHAT WE USE INSTEAD: the average day. `t0_unix` + `dt = 60 s` give each sample a wall-clock time, so
we average the train values into 24 one-hour bins by time-of-day. This is length-invariant,
phase-correct, and it *is* the thing we claim to group by ("clients with a similar normal pattern").

  1. log1p(v - min(v))  — WSD KPIs span 0..6000; log keeps the peaks from dictating the shape.
  2. average by hour-of-day (24 bins), z-normalise -> the client's typical day.
  3. subtract the corpus-mean typical day, z-normalise again.
     Every WSD KPI shares a strong diurnal cycle: without this step intra_r (0.494) and inter_r
     (0.515) are indistinguishable — one blob. Step 3 clusters on how a client's day *differs* from
     the average day, which is where the families actually live.

Validated (see MODEL.md §4): silhouette +0.53, ARI 0.956 across 5 seeds, and 88% (7 of 8) of the pairs of
KPIs that are >0.9 correlated in wall-clock time (i.e. metrics of the same service — never seen by
the clustering) land in the same cluster, vs ~27% under label permutation (p < 1e-4).
"""
import numpy as np

NBINS = 24        # one bin per hour: the shortest train is 1.17 days, finer bins get noisy
DT = 60           # WSD sampling interval, seconds
DAY = 86400

def znorm(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / (x.std() + 1e-9)

def detrend_z(v):
    """Remove linear drift, then z-normalise. Used before any autocorrelation."""
    v = np.asarray(v, float)
    t = np.arange(len(v))
    v = v - np.polyval(np.polyfit(t, v, 1), t)
    return znorm(v)

def acf_at(v, lag):
    """Autocorrelation of the detrended series at exactly `lag`. NaN if the series is too short."""
    if len(v) <= lag: return np.nan
    x = detrend_z(v); n = len(x)
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:lag + 1].real
    return float(ac[lag] / (ac[0] + 1e-12))

def day_profile(v, t0_unix, nbins=NBINS):
    """Mean of log1p(v - min v) per hour-of-day, z-normalised. Length- and phase-invariant.
    `v` may contain NaN (they are ignored)."""
    v = np.asarray(v, float)
    ok = ~np.isnan(v)
    v = v[ok]
    if len(v) == 0: return np.zeros(nbins)
    v = np.log1p(v - v.min())
    idx = np.flatnonzero(ok)
    b = ((int(t0_unix) + idx * DT) % DAY) // (DAY // nbins)
    tot = np.zeros(nbins); cnt = np.zeros(nbins)
    np.add.at(tot, b, v); np.add.at(cnt, b, 1)
    p = np.divide(tot, np.maximum(cnt, 1))
    if (cnt == 0).any():                       # circular interpolation over unobserved hours
        i = np.arange(nbins); g = cnt > 0
        p = np.interp(i, i[g], p[g], period=nbins) if g.sum() >= 2 else np.zeros(nbins)
    return znorm(p)

def deviation_features(profiles, mean_profile=None):
    """Subtract the corpus-mean typical day and re-z-normalise. Returns (features, mean_profile)."""
    profiles = np.asarray(profiles, float)
    if mean_profile is None: mean_profile = profiles.mean(axis=0)
    return np.array([znorm(p - mean_profile) for p in profiles]), mean_profile
