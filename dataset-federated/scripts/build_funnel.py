"""Selection funnel figure for the WSD federated dataset. Counts are read from the frozen
README (written by build_frozen.py) so the figure can never drift from the data."""
import os, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Repo-relative: this file lives in <dataset-federated>/scripts/, so the corpus root
# is its parent. (Was a hardcoded Windows path — broken on Linux.)
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "plots", "selection_funnel.png")
README = os.path.join(BASE, "data", "federated", "WSD_frozen", "README.md")
BLUE = "#3f7fbf"; DARK = "#20303f"; RED = "#c8443f"; MUT = "#5c6774"

txt = open(README, encoding="utf-8").read()
# "**Selection funnel:** 210 -> -30 no-anomaly -> 180 -> -136 no-clean-window -> -1 anomaly-dominated
#   -> -2 no-period -> 41 viable -> -10 near-duplicate (...) -> 31 clients."
FUNNEL = (r"(\d+) -> -(\d+) no-anomaly -> (\d+) -> -(\d+) no-clean-window -> -(\d+) anomaly-dominated "
          r"-> -(\d+) no-period -> (\d+) viable -> -(\d+) near-duplicate .*? -> (\d+) clients")
n0, d_anom, n1, d_win, d_dom, d_per, n_viable, d_dup, n_final = map(int, re.search(FUNNEL, txt, re.S).groups())

stages = [
    ("WSD real-world KPIs", n0, "210 separate univariate KPI series (lengths 30,737–36,471)"),
    ("with ≥1 labelled anomaly", n1, None),
    ("clean window, ≥2 daily cycles", n_viable,
     "imputed straight lines masked as NaN; train anomaly-free + 1-day guard; train ≥ 2 periods; val = 1.25 d"),
    ("deduplicated (time-aligned)", n_final, "|corr| > 0.99 (same series) or label-Jaccard > 0.5 (same incident)"),
]
drops = [None, (f"−{d_anom}", "no anomaly at all"),
         (f"−{d_win} / −{d_dom} / −{d_per}", "no clean window / test >10% anomalous / no daily period"),
         (f"−{d_dup}", "near-duplicate clients (one representative kept)")]

fig, ax = plt.subplots(figsize=(9.2, 5.4))
ax.set_xlim(0, 10); ax.set_ylim(0, len(stages)); ax.axis("off")
maxw = 8.0
for i, (name, cnt, sub) in enumerate(stages):
    y = len(stages) - 1 - i
    w = maxw * cnt / stages[0][1]
    x0 = 5 - w / 2
    shade = plt.cm.Blues(0.35 + 0.5 * cnt / stages[0][1])
    ax.add_patch(FancyBboxPatch((x0, y + 0.18), w, 0.64, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=shade, ec=DARK, lw=1.1))
    ax.text(5, y + 0.5, f"{name}", ha="center", va="center", fontsize=11, fontweight="bold", color="#0b1b2b")
    ax.text(5, y + 0.5, "", ha="center")
    ax.text(9.8, y + 0.5, f"{cnt}", ha="right", va="center", fontsize=15, fontweight="bold",
            color=DARK, family="monospace")
    if sub:
        ax.text(5, y + 0.06, sub, ha="center", va="top", fontsize=7.6, color=MUT, style="italic")
    if drops[i]:
        d, why = drops[i]
        ax.annotate("", xy=(5 - w / 2 - 0.05, y + 1.02), xytext=(5, y + 1.18),
                    arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.2))
        ax.text(0.15, y + 1.02, f"{d}  {why}", ha="left", va="center", fontsize=8.5, color=RED)

ax.text(0.15, len(stages) - 0.02, "Selection funnel — WSD federated clients", fontsize=13.5,
        fontweight="bold", color="#0b1b2b", va="bottom")
ax.text(9.85, len(stages) - 0.02, "clusters: k=4 on TRAIN only (seed 0) · split 50% · val = 1 period (1 day) · train >= 2 periods",
        fontsize=8.5, color=MUT, ha="right", va="bottom")
fig.tight_layout()
fig.savefig(OUT, dpi=130, bbox_inches="tight"); plt.close(fig)
print("wrote", OUT, round(os.path.getsize(OUT) / 1e6, 2), "MB")
