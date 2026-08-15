#!/usr/bin/env python3
"""Framework dell'arm A2 (federated_enc_fedavg + suffstat + prior partial + BN shared),
nello stesso stile della figura MLISE — vedi STYLE.md.

Ogni numero nella figura e' VERIFICATO, non dichiarato:
  - 163 tensori di pesi encoder + 44 buffer BN = 207 condivisi (banner della run)
  - 6 tensori del quantizer identici fra i client
  - 209 tensori di decoder + 2 di refinement DIVERGONO -> restano locali
  - 55 chiavi del corpo del prior condivise, teste channel_embedding/output_bias locali
Misurati sui 5 stage1.ckpt di artifacts/runs/c50_a2/.../ucr_187/seed0/.

    /home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10 a2_framework.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Ellipse, FancyArrowPatch, Rectangle
import numpy as np

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"

# ── palette e convenzioni: identiche a STYLE.md ──────────────────────────────
STAGE_A, STAGE_B = "#FF8029", "#FFFF00"
PANEL, PANEL_B = "#71CEDC", "#71E2D5"
ENC, INPUT, VIEW = "#FF9A9A", "#AEC6FF", "#90EF90"
INNER = "#F9F9FA"
A_STAGE, A_PANEL, A_BOX = 0.19, 0.59, 0.49
ACC = dict(blue="#4567A9", orange="#D17920", green="#2B7B4F",
           purple="#7B3A99", red="#B0302F")

LW = 1.0
FS = 1.30                      # i riquadri sono larghi: le scritte possono crescere
FONT = "DejaVu Serif"          # l'originale usa Bookman Demi Italic
TXT = dict(family=FONT, style="italic", weight="bold", ha="center", va="center")

W, H = 100.0, 132.0
# Single-column figure, just under half of an IEEE text page in height.
fig = plt.figure(figsize=(3.5, 3.5 * H / W))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")


def box(x, y, w, h, fc, alpha, fs=4.4*FS, lines=(), rr=None, ec="black"):
    r = rr if rr is not None else min(0.35 * h, 0.45 * w)
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, alpha=alpha, ec=ec, lw=LW, zorder=3))
    for i, t in enumerate(lines):
        ax.text(x, y + (len(lines) - 1 - 2 * i) * fs * 0.248, t, fontsize=fs, zorder=4, **TXT)


def trapez(x, y, w, h, fc, lines, fs=4.4*FS, flip=False, taper=0.30):
    t = taper * w
    top, bot = (w / 2, w / 2 - t) if not flip else (w / 2 - t, w / 2)
    ax.add_patch(Polygon([(x - top, y + h / 2), (x + top, y + h / 2),
                          (x + bot, y - h / 2), (x - bot, y - h / 2)],
                         closed=True, fc=fc, alpha=A_BOX, ec="black", lw=LW,
                         joinstyle="round", zorder=3))
    for i, s in enumerate(lines):
        ax.text(x, y + (len(lines) - 1 - 2 * i) * fs * 0.248, s, fontsize=fs, zorder=4, **TXT)


def stage(x, y, w, h, fc, tag, tag_side="left"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0,rounding_size=2.4",
                                fc=fc, alpha=A_STAGE, ec="none", zorder=0))
    is_left = tag_side == "left"
    tag_x = x - w / 2 + 2.0 if is_left else x + w / 2 - 2.0
    ax.text(tag_x, y - h / 2 + 1.8, tag, fontsize=5.4*FS, zorder=1,
            family=FONT, style="italic", weight="bold",
            ha="left" if is_left else "right", va="bottom")


def arrow(p0, p1, rad=0.0, lw=LW, color="black", head=7.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=head,
                                 lw=lw, color=color, linestyle=ls, zorder=5,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=1.5, shrinkB=1.5))


def loss(x, y, sub, fs=6.6*FS):
    ax.text(x, y, r"$\mathcal{L}$", fontsize=fs, zorder=6,
            family=FONT, style="italic", weight="bold", ha="right", va="center")
    ax.text(x + 0.3, y - fs * 0.13, sub, fontsize=fs * 0.46, zorder=6,
            family=FONT, style="italic", weight="bold", ha="left", va="center")


def snowflake(x, y, s=1.3, color=None):
    for a in np.arange(0, np.pi, np.pi / 3):
        ax.plot([x - s * np.cos(a), x + s * np.cos(a)],
                [y - s * np.sin(a), y + s * np.sin(a)],
                color=color or ACC["blue"], lw=LW * 0.9, zorder=6, solid_capstyle="round")


def series(x, y, w, h, seed, spike=None, lw=0.45):
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, fc="white", alpha=0.95,
                           ec="black", lw=LW * 0.8, zorder=3))
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, 240)
    v = np.sin(2 * np.pi * 6 * t) * 0.5 + rng.normal(0, 0.15, t.size)
    if spike is not None:
        m = (t > spike) & (t < spike + 0.10)
        v[m] += np.sin(2 * np.pi * 40 * t[m]) * 0.8
    v = v / (np.abs(v).max() * 2.35)
    ax.plot(x - w / 2 + t * w, y + v * h, color="black", lw=LW * lw, zorder=4)
    if spike is not None:
        ax.plot(x - w / 2 + t[m] * w, y + v[m] * h, color=ACC["red"], lw=LW * 0.6, zorder=5)


def tokens(x, y, n, m, cell=1.3, seed=0, anomalous=None, masked=()):
    rng = np.random.default_rng(seed)
    pal = [INPUT, VIEW, ENC, PANEL, "#D9C7F0"]
    for i in range(n):
        for j in range(m):
            c = pal[rng.integers(0, len(pal))]
            if anomalous and anomalous[0] <= i < anomalous[1]:
                c = ACC["red"]
            if i in masked:
                c = "#9A9A9A"
            ax.add_patch(Rectangle((x + i * cell, y + j * cell), cell * 0.92, cell * 0.92,
                                   fc=c, alpha=0.85, ec="black", lw=LW * 0.35, zorder=4))


# ═════════════ STAGE 1 ═════════════
# The two training stages now share one grammar: client pipeline on the left,
# server on the right, and communication confined to the central gutter.
stage(50, 109.5, 94, 45, STAGE_A, "Stage 1  ·  tokenizer", tag_side="right")

# ---- client pipeline ----
series(25, 128.0, 25, 2.6, seed=30, lw=0.4)
box(25, 122.8, 27, 4.6, INPUT, A_BOX, lines=["STFT"], fs=4.8*FS)
trapez(25, 116.4, 34, 5.2, ENC, [r"Encoder  $E$"], fs=4.6*FS, taper=0.22)
box(25, 108.8, 38, 6.2, PANEL, A_PANEL,
    lines=["Vector Quantizer", "codebook FROZEN in-round"], fs=3.9*FS)
snowflake(6.5, 110.6, 1.0)
trapez(25, 100.8, 34, 5.4, ENC, [r"Decoder  $D_k$   local"], fs=4.1*FS,
       flip=True, taper=0.22)
box(25, 92.8, 39, 6.2, INNER, 0.99,
    lines=[r"$\mathcal{L}_\mathrm{reconstruction} + \mathcal{L}_\mathrm{commit}$"], fs=4.8*FS)

arrow((25, 126.5), (25, 125.2))
arrow((25, 120.4), (25, 119.4))
arrow((25, 113.6), (25, 112.0))
arrow((25, 105.5), (25, 103.7))
arrow((25, 97.9), (25, 96.1))

# ---- server and ownership note ----
box(73, 116.1, 46, 18.2, PANEL_B, A_PANEL, lines=[], rr=3.2)
ax.text(73, 122.5, "Server   (no data)", fontsize=4.9*FS, zorder=6, **TXT)
ax.text(73, 118.5, "codebook:  k-means M-step", fontsize=4.0*FS, zorder=6, **TXT)
ax.text(73, 114.6, "encoder:  FedAvg", fontsize=4.0*FS, zorder=6, **TXT)
ax.text(73, 110.7, "BN running stats:  pooled", fontsize=4.0*FS, zorder=6, **TXT)
arrow((42.5, 112.0), (50.0, 118.4), rad=-0.17)
arrow((50.0, 111.4), (42.5, 107.0), rad=-0.17)

box(71, 96.5, 48, 6.4, INNER, 0.99,
    lines=[r"$D_k$ + refinement never leave client $k$"], fs=4.0*FS)
ax.plot([39.0, 49.5], [101.0, 97.8], color="black", lw=LW * 0.7,
        ls=(0, (2.2, 1.8)), zorder=2)

# ═════════════ STAGE 2 ═════════════
stage(50, 71.0, 94, 28, STAGE_B, "Stage 2  ·  prior")

tokens(18.8, 80.8, 8, 3, cell=1.35, seed=11, masked=(2, 3, 6))
box(25, 73.2, 40, 6.6, PANEL, A_PANEL,
    lines=["MaskGIT prior", "body shared  /  head local"], fs=4.0*FS)
box(25, 64.1, 25, 5.6, INNER, 0.99,
    lines=[r"$\mathcal{L}_\mathrm{stage\,2}$"], fs=5.0*FS)
arrow((25, 79.5), (25, 76.7))
arrow((25, 69.8), (25, 67.1))

box(73, 73.2, 46, 14.6, PANEL_B, A_PANEL, lines=[], rr=3.0)
ax.text(73, 78.3, "Server", fontsize=4.7*FS, zorder=6, **TXT)
ax.text(73, 74.3, "FedAvg on the body keys", fontsize=4.0*FS, zorder=6, **TXT)
ax.text(73, 70.1, "channel embedding + bias  stay local",
        fontsize=3.5*FS, zorder=6, **TXT)
arrow((43.0, 75.7), (50.0, 77.7), rad=-0.14)
arrow((50.0, 69.4), (43.0, 70.8), rad=-0.14)

# ═════════════ INFERENCE ═════════════
ax.plot([4, 96], [54.0, 54.0], color="black", lw=LW * 0.7,
        ls=(0, (3.2, 2.4)), zorder=2)
ax.text(5.5, 51.1, r"Inference   (entirely on client $k$)", fontsize=5.2*FS, zorder=6,
        family=FONT, style="italic", weight="bold", ha="left", va="center")

# Shared trunk: one row followed by a centred prior.
series(14, 44.8, 21, 3.0, seed=21, spike=0.62)
ax.text(14, 40.9, "test window", fontsize=4.0*FS, zorder=6, **TXT)
box(44, 44.8, 30, 5.8, INPUT, A_BOX,
    lines=["encoder + VQ  shared"], fs=4.0*FS)
snowflake(28.8, 47.4, 1.0)
arrow((24.7, 44.8), (29.7, 44.8))

tokens(64.5, 42.8, 7, 3, cell=1.35, seed=13, anomalous=(4, 6))
arrow((58.2, 44.8), (64.2, 44.8))
box(50, 34.2, 44, 6.8, PANEL, A_PANEL,
    lines=["prior surprise", "shared body + local head"], fs=4.0*FS)
arrow((70.0, 42.3), (61.0, 38.2), rad=0.16)

# The only branch in inference: detection and explanation.
arrow((43.0, 30.7), (26.0, 26.5), rad=0.10)
arrow((57.0, 30.7), (74.0, 26.5), rad=-0.10)
box(25, 22.8, 44, 6.0, VIEW, A_BOX,
    lines=["smoothing over channel and time"], fs=3.8*FS)
box(75, 22.8, 44, 6.0, PANEL_B, A_PANEL,
    lines=["explainable sampling"], fs=4.1*FS)

box(25, 10.5, 44, 7.4, PANEL_B, A_PANEL,
    lines=["detection", "anomaly score  vs  threshold"], fs=3.8*FS)
trapez(75, 10.5, 44, 7.4, ENC,
       ["counterfactual", r"Decoder  $D_k$   local"], fs=4.0*FS,
       flip=True, taper=0.20)
arrow((25, 19.7), (25, 14.4))
arrow((75, 19.7), (75, 14.4))
snowflake(62.0, 14.3, 1.0)


fig.savefig("a2_framework.pdf")
fig.savefig("a2_framework.svg")
fig.savefig("a2_framework_check.png", dpi=250)
print("scritti a2_framework.pdf + a2_framework.svg + _check.png")
