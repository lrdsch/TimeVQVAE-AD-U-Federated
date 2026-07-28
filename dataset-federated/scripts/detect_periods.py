"""Per-client seasonal period detection on the frozen WSD dataset.

Runs on the `train` portion ONLY (anomaly- & NaN-free), so nothing here can leak from val/test.
Two independent estimators must agree before a period is called:

  1. ACF   - autocorrelation of the linearly-detrended, z-normalised train. The peak is chosen by
             PROMINENCE, not height: on a smooth series the ACF decays slowly, so the highest local
             maximum sits at a ~30-60 min lag and reflects smoothness, not seasonality. Prominence
             ignores the decaying envelope and finds the bump that actually sticks out.
  2. FFT   - highest power in a zero-padded periodogram. Zero-padding matters: with N=4176 the raw
             frequency grid has bins at 2088 and 1392 min and nothing between, so an un-padded
             estimate cannot resolve 1440.

A third, direct check is reported: `acf_at_1440`, the plain autocorrelation at exactly one day.
That is the number to trust for "is this KPI daily?", since 1-min web KPIs have a known candidate.

A period is only *observable* if the train holds at least MIN_CYCLES full cycles, so candidate
lags are capped at train_len / MIN_CYCLES. A client whose train is 1.2 days cannot evidence a
weekly cycle, and reporting one would be an artefact.

Output: data/federated/WSD_frozen/periods.csv  +  plots/periods.png
Run: python scripts/detect_periods.py     (env TimeEnvM)

Run it AFTER build_frozen.py: that script rebuilds the frozen directory from scratch (rmtree), so it
deletes periods.csv. The manifest carries `period_min` / `period_acf` for every client regardless.
"""
import os
import numpy as np, pandas as pd
from scipy.signal import find_peaks, periodogram
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Repo-relative: this file lives in <dataset-federated>/scripts/, so the corpus root
# is its parent. (Was a hardcoded Windows path — broken on Linux.)
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(BASE, "data", "federated", "WSD_frozen")
OUT_CSV = os.path.join(D, "periods.csv")
OUT_PNG = os.path.join(BASE, "plots", "periods.png")

MIN_LAG = 30        # minutes: below this it's noise autocorrelation, not seasonality
MIN_CYCLES = 2.0    # need >= 2 full cycles in train to claim a period
TOL = 0.15          # ACF and FFT agree if they are within 15% of each other
MIN_STRENGTH = 0.20 # the ACF at the claimed period must actually be positive and non-trivial.
                    # Without this, a train shorter than one cycle yields a "peak" at ACF = -0.6:
                    # prominence finds a local max on a monotonically decaying curve.

def detrend_z(v):
    v = np.asarray(v, float)
    t = np.arange(len(v))
    v = v - np.polyval(np.polyfit(t, v, 1), t)      # remove linear drift
    return (v - v.mean()) / (v.std() + 1e-9)

def full_acf(x, max_lag):
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:max_lag + 1].real
    return ac / (ac[0] + 1e-12)

def acf_period(ac):
    """Peak chosen by PROMINENCE: the decaying envelope makes the tallest peak meaningless."""
    peaks, props = find_peaks(ac[MIN_LAG:], prominence=1e-4)
    if len(peaks) == 0: return np.nan, np.nan
    best = int(peaks[np.argmax(props["prominences"])]) + MIN_LAG
    return best, float(ac[best])

def fft_period(x, max_lag):
    nfft = 1 << int(np.ceil(np.log2(len(x) * 8)))       # zero-pad x8 to resolve the daily bin
    freqs, power = periodogram(x, fs=1.0, nfft=nfft, scaling="spectrum")
    with np.errstate(divide="ignore"):
        per = 1 / np.maximum(freqs, 1e-12)
    ok = (freqs > 0) & (per >= MIN_LAG) & (per <= max_lag)
    if not ok.any(): return np.nan, np.nan
    f = freqs[ok][np.argmax(power[ok])]
    return int(round(1 / f)), float(power[ok].max() / (power[ok].sum() + 1e-12))

def fmt(minutes):
    if not np.isfinite(minutes): return "-"
    h = minutes / 60
    return f"{int(minutes)} ({h:.1f}h)" if h < 48 else f"{int(minutes)} ({h/24:.1f}d)"

Z = np.load(os.path.join(D, "wsd_federated.npz"))
man = pd.read_csv(os.path.join(D, "manifest.csv"))
rows = []
for r in man.itertuples():
    tr = Z[f"{int(r.id)}_train"]
    x = detrend_z(tr)
    max_lag = int(len(tr) / MIN_CYCLES)
    ac = full_acf(x, max_lag)
    p_acf, strength = acf_period(ac)
    p_fft, share = fft_period(x, max_lag)
    agree = (np.isfinite(p_acf) and np.isfinite(p_fft)
             and abs(p_acf - p_fft) <= TOL * max(p_acf, p_fft)
             and np.isfinite(strength) and strength >= MIN_STRENGTH)
    # can this train even show a daily cycle? and if so, how strong is it AT exactly 1 day?
    can_daily = len(tr) >= MIN_CYCLES * 1440
    if not can_daily: agree = False          # cannot evidence a period it never completes twice
    acf_1440 = float(ac[1440]) if len(ac) > 1440 else np.nan
    rows.append(dict(id=int(r.id), cluster=int(r.cluster), train_len=int(r.train_len),
                     train_days=round(len(tr) / 1440, 2), max_detectable_min=max_lag,
                     period_acf=p_acf, acf_strength=round(strength, 3) if np.isfinite(strength) else np.nan,
                     period_fft=p_fft, fft_share=round(share, 3) if np.isfinite(share) else np.nan,
                     acf_at_1440=round(acf_1440, 3) if np.isfinite(acf_1440) else np.nan,
                     agree=bool(agree), can_show_daily=bool(can_daily),
                     period_min=int(p_acf) if agree else np.nan))
df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)

pd.set_option("display.width", 220)
show = df.copy()
show["acf"] = show.period_acf.map(fmt); show["fft"] = show.period_fft.map(fmt)
print(show[["id", "cluster", "train_days", "max_detectable_min", "acf", "acf_strength",
            "fft", "fft_share", "acf_at_1440", "agree", "can_show_daily"]].to_string(index=False))

ok = df[df.agree]
print(f"\nACF e FFT concordano su {len(ok)}/{len(df)} client")
print(f"train con >= {MIN_CYCLES} cicli giornalieri (>= {int(MIN_CYCLES*1440)} pt): {int(df.can_show_daily.sum())}/{len(df)}")
if len(ok):
    near_day = ok[(ok.period_min > 1440 * 0.85) & (ok.period_min < 1440 * 1.15)]
    print(f"entro ±15% di 1440 min: {len(near_day)}/{len(ok)} dei concordi")
    print(f"mediana dei periodi concordi: {int(ok.period_min.median())} min = {ok.period_min.median()/60:.2f} h")
d = df[df.can_show_daily]
print(f"\nTEST DIRETTO acf(lag=1440) sui {len(d)} client che possono mostrare un ciclo giornaliero:")
print(f"  mediana {d.acf_at_1440.median():.3f} | >0.5: {(d.acf_at_1440>0.5).sum()} | >0.3: {(d.acf_at_1440>0.3).sum()} | <=0: {(d.acf_at_1440<=0).sum()}")
print(f"  i piu' deboli: {d.nsmallest(5,'acf_at_1440')[['id','train_days','acf_at_1440']].to_dict('records')}")

fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
CLC = ["#4aa3df", "#e08a3c", "#5fbf6b", "#b07fd6"]
for c in sorted(df.cluster.unique()):
    s = df[(df.cluster == c) & df.agree]
    ax[0].scatter(s.train_days, s.period_min / 60, s=42, color=CLC[c % 4], label=f"c{c}", zorder=3)
bad = df[~df.agree]
ax[0].scatter(bad.train_days, np.full(len(bad), 0), marker="x", color="#c8443f", s=40, label="no agreement", zorder=3)
ax[0].axhline(24, ls="--", c="#333", lw=1, label="24 h")
ax[0].set_xlabel("train length (days)"); ax[0].set_ylabel("detected period (hours)")
ax[0].set_title("period vs train length"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.2)
ax[1].hist(ok.period_min / 60, bins=np.arange(0, 30, 0.5), color="#3f7fbf", edgecolor="#20303f")
ax[1].axvline(24, ls="--", c="#c8443f", lw=1.4)
ax[1].set_xlabel("detected period (hours)"); ax[1].set_ylabel("clients")
ax[1].set_title(f"agreed periods (n={len(ok)})"); ax[1].grid(alpha=.2)
fig.tight_layout(); fig.savefig(OUT_PNG, dpi=130); plt.close(fig)
print(f"\nwrote {OUT_CSV}\nwrote {OUT_PNG}")
