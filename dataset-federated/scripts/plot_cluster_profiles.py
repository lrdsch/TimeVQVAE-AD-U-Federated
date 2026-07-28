"""Visual proof that the clusters group clients with a similar normal day.

Left: the OLD descriptor (whole train resampled to 256 pts, z-normed) -- the clusters it produced.
Right: the frozen clusters, on the descriptor actually used (average-day deviation profile).
Each thin line is one client's average day; the thick line is the cluster centroid.
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import day_profile, deviation_features, znorm

# Repo-relative: this file lives in <dataset-federated>/scripts/, so the corpus root
# is its parent. (Was a hardcoded Windows path — broken on Linux.)
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(BASE, "data", "federated", "WSD_frozen")
OUT = os.path.join(BASE, "plots", "cluster_profiles.png")
Z = np.load(os.path.join(D, "wsd_federated.npz")); man = pd.read_csv(os.path.join(D, "manifest.csv")).reset_index(drop=True)
ids = [int(s) for s in man.id]

prof = np.array([day_profile(Z[f"{s}_train"], man.t0_unix[i]) for i, s in enumerate(ids)])
dev, _ = deviation_features(prof)

def old_feat(sid):                       # the descriptor that was in build_frozen.py before
    y = np.asarray(Z[f"{sid}_train"], float)
    return znorm(np.interp(np.linspace(0, 1, 256), np.linspace(0, 1, len(y)), y))
old = np.array([old_feat(s) for s in ids])
old_lab = KMeans(6, n_init=10, random_state=0).fit_predict(old)
new_lab = man.cluster.values

CLC = ["#4aa3df", "#e08a3c", "#5fbf6b", "#b07fd6", "#d95f5f", "#4bc3c3"]
hours = np.arange(24)
panels = [("OLD feature: zr(train) resampled to 256 pts  (k=6)", old_lab, 6),
          ("FROZEN: average-day deviation profile  (k=4)", new_lab, 4)]
fig, axes = plt.subplots(2, max(6, 4), figsize=(17, 6.8), sharex=True, sharey=True)
for row, (title, lab, k) in enumerate(panels):
    for c in range(axes.shape[1]):
        ax = axes[row, c]
        if c >= k: ax.axis("off"); continue
        mem = [i for i in range(len(ids)) if lab[i] == c]
        for i in mem: ax.plot(hours, prof[i], color=CLC[c % 6], alpha=0.32, lw=0.9)
        ax.plot(hours, prof[mem].mean(0), color="#111", lw=2.1)
        ax.set_title(f"c{c}  n={len(mem)}", fontsize=9.5)
        ax.grid(alpha=0.18); ax.set_xticks([0, 6, 12, 18])
    axes[row, 0].set_ylabel("z(avg day)", fontsize=9)
    # row banner above the row, not on top of the y-label
    axes[row, 0].annotate(title, xy=(0, 1.30), xycoords="axes fraction", ha="left", va="bottom",
                          fontsize=11, fontweight="bold")
for ax in axes[1]: ax.set_xlabel("hour of day", fontsize=9)
fig.suptitle("Each thin line = one client's average day (identical data in both rows). "
             "Only the grouping differs.", fontsize=12, y=0.995)
# manual layout: tight_layout() followed by subplots_adjust() throws the grid away
fig.subplots_adjust(left=0.045, right=0.995, top=0.825, bottom=0.085, wspace=0.16, hspace=0.52)
fig.savefig(OUT, dpi=125); plt.close(fig)
print("wrote", OUT)
