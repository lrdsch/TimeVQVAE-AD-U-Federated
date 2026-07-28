"""Stage-1 reconstruction vs width_base, on the conv-probe c3 checkpoints.

Visualizes the mechanism behind Curve 2 (VUS-PR vs width): a narrow conv body
(wb4) reconstructs like a moving average; widening sharpens it; wb64 may overfit.
Grid: rows = 2 clients (cleanest + a harder one, picked by BN-recal test nMSE at
wb16), cols = width_base {4,8,16,32,64}. Each cell overlays the original test
window (black) vs the Stage-1 reconstruction (color), with BN-recalibrated test
nMSE and fraction of high-frequency power retained in the title. A moving-average
of the same window is drawn dashed on the wb4 cell as the "is it just an MA?" ref.

All models are the LOCAL arm (one model per client, no federation).
"""
import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipeline"))
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.plot_retrain_train_test import load_stage1, recalibrate_bn
from scripts.local_recon_autopsy import load_cfg, get_windows, nmse as recon_nmse, moving_avg, spectral_retained

# Input tree of the 2026-07-17 width probe. It was deleted in the 2026-07-23 purge and NO
# script in the repo regenerates it, so this plot only runs against a freshly re-run probe:
# point RECON_WIDTH_ROOT at it (env var), e.g.
#   RECON_WIDTH_ROOT=artifacts/fed_eval/my_width_probe python scripts/plot_recon_width_c3.py
ROOT = os.environ.get("RECON_WIDTH_ROOT", "artifacts/fed_eval/conv_probe_c3_20260717")
WIDTHS = [("wb4_1x", 4), ("wb8_1x", 8), ("wb16_1x", 16), ("wb32_1x", 32), ("wb64_1x", 64)]
CLIENTS = ["kpi_086", "kpi_104", "kpi_110", "kpi_150", "kpi_170", "kpi_182"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def ckpt(run, ent):
    return f"{ROOT}/{run}/c3/seed0/local/{ent}/stage1.ckpt"


def recon_of(run, ent, Xtest, Xtrain):
    """Load width `run` for client `ent`, BN-recalibrate on its train windows,
    return (recon[N,T], macro test nMSE, hi-freq retained @0.5)."""
    cfg = load_cfg(ckpt(run, ent))
    ex = torch.from_numpy(Xtrain[:1]).float()
    m = load_stage1(ckpt(run, ent), cfg, ex, device=DEV)
    recalibrate_bn(m, torch.from_numpy(Xtrain).float().to(DEV), bs=128, passes=5)
    m.eval()
    with torch.no_grad():
        r = m(torch.from_numpy(Xtest).float().to(DEV))["reconstructed"].cpu().numpy()
    r = r.reshape(Xtest.shape)
    mse = float(np.mean((r - Xtest) ** 2) / (np.mean(Xtest ** 2) + 1e-9))
    hf = float(spectral_retained(Xtest[:, 0], r[:, 0])["high"])   # fraction of hi-freq power kept
    return r, mse, hf


def windows_np(cfg, ent, split):
    X, _Y = get_windows(cfg, ent, DEV, split=split, max_windows=400)   # (X, labels)
    return X.detach().cpu().numpy()


# 1) rank clients by wb16 test nMSE (BN-recal) → pick cleanest + a harder one
print("[rank] scoring clients at wb16 ...")
score = {}
cache = {}
for ent in CLIENTS:
    cfg = load_cfg(ckpt("wb16_1x", ent))
    Xtr = windows_np(cfg, ent, "train"); Xte = windows_np(cfg, ent, "test")
    cache[ent] = (Xtr, Xte)
    _, mse, _ = recon_of("wb16_1x", ent, Xte, Xtr)
    score[ent] = mse
    print(f"   {ent}: wb16 test nMSE = {mse:.3f}")
order = sorted(score, key=score.get)
pick = [order[0], order[len(order) // 2]]   # cleanest + median
print(f"[pick] clean={pick[0]} (nMSE {score[pick[0]]:.3f}), harder={pick[1]} (nMSE {score[pick[1]]:.3f})")

# 2) for each picked client, choose the highest-energy test window to display
fig, axes = plt.subplots(len(pick), len(WIDTHS), figsize=(3.1 * len(WIDTHS), 2.6 * len(pick)),
                         squeeze=False)
for ri, ent in enumerate(pick):
    Xtr, Xte = cache[ent]
    wi = int(np.argmax(np.var(Xte[:, 0], axis=1)))   # the window with most structure
    x = Xte[wi, 0]
    for ci, (run, wb) in enumerate(WIDTHS):
        r, mse, hf = recon_of(run, ent, Xte, Xtr)
        ax = axes[ri][ci]
        ax.plot(x, color="black", lw=1.1, label="original")
        ax.plot(r[wi, 0], color=plt.cm.viridis(ci / (len(WIDTHS) - 1)), lw=1.4, label="recon")
        if wb == 4:
            ma = moving_avg(torch.from_numpy(Xte).float(), 8).numpy()[wi, 0]
            ax.plot(ma, color="red", lw=0.9, ls="--", alpha=0.7, label="MA(8)")
            ax.legend(fontsize=6, loc="upper right")
        ax.set_title(f"wb{wb}  nMSE={mse:.3f}  hf={hf:.2f}", fontsize=8.5)
        ax.set_xticks([]); ax.set_yticks([])
    axes[ri][0].set_ylabel(f"{ent}\n(win {wi})", fontsize=8)

fig.suptitle("Stage-1 reconstruction vs width_base — wsd_fed/c3 local, BN-recal test window\n"
             "(narrow body = moving-average-like; widening sharpens; wb64 may overfit)",
             fontweight="bold", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = "plots/codebook_analysis/recon_width_c3_20260717.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=300, bbox_inches="tight")
print("saved:", out)
