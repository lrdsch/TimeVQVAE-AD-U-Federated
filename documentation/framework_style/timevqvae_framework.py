#!/usr/bin/env python3
"""Framework di TimeVQVAE-AD disegnato nello stile della figura MLISE.

Contenuto: Fig. 1 (inferenza) + Fig. 4 (stage 1 / stage 2) di 2311.12550v5, ricomposte
nella grammatica visiva estratta in STYLE.md — colori puri a tre livelli di opacita',
tratto nero uniforme, angoli al 35% dell'altezza, trapezi per gli encoder, ellissi come
contenitori, due regioni di stage affiancate con le loss che convergono.

Esce vettoriale (PDF) piu' un PNG di controllo.

    /home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10 timevqvae_framework.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Ellipse, FancyArrowPatch, Rectangle
import numpy as np

matplotlib.rcParams["pdf.fonttype"] = 42

# ── palette misurata da ClusteringFramework.png (STYLE.md §1) ────────────────
STAGE_A, STAGE_B = "#FF8029", "#FFFF00"      # sfondi di stage      alpha 0.19
PANEL,   PANEL_B = "#71CEDC", "#71E2D5"      # pannelli in evidenza alpha 0.59
ENC,     INPUT,   VIEW = "#FF9A9A", "#AEC6FF", "#90EF90"   #        alpha 0.49
INNER = "#F9F9FA"
A_STAGE, A_PANEL, A_BOX = 0.19, 0.59, 0.49
ACC = dict(blue="#4567A9", orange="#D17920", green="#2B7B4F",
           purple="#7B3A99", red="#B0302F")

LW = 1.0                    # tratto UNIFORME ~1 pt alla dimensione di resa
FONT = "DejaVu Serif"       # l'originale usa Bookman Demi Italic (qui non installato)
TXT = dict(family=FONT, style="italic", weight="bold", ha="center", va="center")

W, H = 100.0, 141.0         # stesso rapporto della figura MLISE (0.708)
fig = plt.figure(figsize=(3.5, 3.5 * H / W))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")


def box(x, y, w, h, fc, alpha, label=None, fs=4.6, lines=None, rr=None):
    """Rettangolo arrotondato: raggio = 35% dell'altezza (STYLE.md §2).

    Il rettangolo va passato per INTERO: con `pad=0` matplotlib arrotonda dentro il
    riquadro dato. Sottrarre 2r dalle dimensioni (come facevo prima) fa superare al
    raggio la meta' dell'altezza residua e produce bulbi al posto degli angoli.
    """
    r = rr if rr is not None else min(0.35 * h, 0.45 * w)
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, alpha=alpha, ec="black", lw=LW, zorder=3,
                                mutation_aspect=1))
    ls = lines or ([label] if label else [])
    for i, t in enumerate(ls):
        ax.text(x, y + (len(ls) - 1 - 2 * i) * fs * 0.248, t, fontsize=fs, zorder=4, **TXT)


def trapez(x, y, w, h, fc, lines, fs=4.6, flip=False, taper=0.30):
    """Encoder: rastremato verso il basso. flip=True -> decoder, si allarga."""
    t = taper * w
    top, bot = (w / 2, w / 2 - t) if not flip else (w / 2 - t, w / 2)
    ax.add_patch(Polygon([(x - top, y + h / 2), (x + top, y + h / 2),
                          (x + bot, y - h / 2), (x - bot, y - h / 2)],
                         closed=True, fc=fc, alpha=A_BOX, ec="black", lw=LW,
                         joinstyle="round", zorder=3))
    for i, s in enumerate(lines):
        ax.text(x, y + (len(lines) - 1 - 2 * i) * fs * 0.248, s, fontsize=fs, zorder=4, **TXT)


def bubble(x, y, rx, ry, fc=INNER, alpha=0.99):
    ax.add_patch(Ellipse((x, y), 2 * rx, 2 * ry, fc=fc, alpha=alpha,
                         ec="black", lw=LW, zorder=3))


def stage(x, y, w, h, fc, tag):
    """Sfondo di stage: grande, arrotondato, SENZA contorno. Etichetta DENTRO la regione."""
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0,rounding_size=2.4",
                                fc=fc, alpha=A_STAGE, ec="none", zorder=0))
    ax.text(x - w / 2 + 2.0, y - h / 2 + 2.0, tag, fontsize=6.0, zorder=1,
            family=FONT, style="italic", weight="bold", ha="left", va="bottom")


def arrow(p0, p1, rad=0.0, ls="-", lw=LW, color="black", head=7.0):
    """Punta triangolare piena e generosa, come i marker Arrow2 dell'originale."""
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=head,
                                 lw=lw, color=color, linestyle=ls, zorder=5,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=1.5, shrinkB=1.5))


def loss(x, y, sub, fs=7.0):
    ax.text(x, y, r"$\mathcal{L}$", fontsize=fs, zorder=6,
            family=FONT, style="italic", weight="bold", ha="right", va="center")
    ax.text(x + 0.3, y - fs * 0.13, sub, fontsize=fs * 0.48, zorder=6,
            family=FONT, style="italic", weight="bold", ha="left", va="center")


def snowflake(x, y, s=1.5):
    """Il fiocco di neve del paper: modello congelato."""
    for a in np.arange(0, np.pi, np.pi / 3):
        ax.plot([x - s * np.cos(a), x + s * np.cos(a)],
                [y - s * np.sin(a), y + s * np.sin(a)],
                color=ACC["blue"], lw=LW * 0.9, zorder=6, solid_capstyle="round")


def series(x, y, w, h, seed, spike=None):
    """Miniatura di serie temporale dentro un riquadro."""
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, fc="white", alpha=0.95,
                           ec="black", lw=LW * 0.8, zorder=3))
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, 220)
    v = np.sin(2 * np.pi * 5 * t) * 0.5 + rng.normal(0, 0.16, t.size)
    if spike is not None:
        m = (t > spike) & (t < spike + 0.10)
        v[m] += np.sin(2 * np.pi * 40 * t[m]) * 0.75
    v = v / (np.abs(v).max() * 2.35)
    ax.plot(x - w / 2 + t * w, y + v * h, color="black", lw=LW * 0.45, zorder=4)
    if spike is not None:
        ax.plot(x - w / 2 + t[m] * w, y + v[m] * h, color=ACC["red"], lw=LW * 0.6, zorder=5)


def tokens(x, y, n, m, cell=1.5, seed=0, anomalous=None):
    """La griglia di token s: due assi = tempo x frequenza, colore = similarita'."""
    rng = np.random.default_rng(seed)
    pal = [INPUT, VIEW, ENC, PANEL, "#D9C7F0"]
    for i in range(n):
        for j in range(m):
            c = pal[rng.integers(0, len(pal))]
            if anomalous and anomalous[0] <= i < anomalous[1]:
                c = ACC["red"]
            ax.add_patch(Rectangle((x + i * cell, y + j * cell), cell * 0.92, cell * 0.92,
                                   fc=c, alpha=0.85, ec="black", lw=LW * 0.35, zorder=4))


# ═══════════════════════ TRONCO COMUNE: x -> STFT -> E -> VQ -> s ═══════════════════════
ax.text(50, 138.5, "TimeVQVAE-AD", fontsize=7.2, zorder=6,
        family=FONT, style="italic", weight="bold", ha="center", va="center")

for k, (xx, sp) in enumerate([(22, None), (38, None), (54, None)]):
    series(xx, 132, 14, 3.4, seed=k, spike=sp)
ax.text(78, 132, "normal\nwindows", fontsize=4.4, zorder=6, **TXT)
arrow((62, 132), (69, 132))

box(50, 124, 30, 6, INPUT, A_BOX, lines=["STFT"], fs=5.2)
arrow((38, 129.8), (46, 127.2), rad=-0.15)

trapez(50, 114.5, 34, 8, ENC, ["Encoder  E", "(dim.-preservative)"], fs=4.3)
arrow((50, 121), (50, 118.7))

box(50, 104, 34, 6.5, PANEL, A_PANEL, lines=["Vector Quantizer"], fs=4.8)
arrow((50, 110.4), (50, 107.4))
bubble(78, 104, 9, 5)
tokens(72.5, 101.5, 4, 3, cell=1.6, seed=3)
ax.text(78, 111.0, "codebook  V", fontsize=4.0, zorder=6, **TXT)
arrow((67.2, 104), (69.1, 104))

ax.text(50, 97.2, r"$\boldsymbol{s}$", fontsize=7.5, zorder=6,
        family=FONT, style="italic", weight="bold", ha="center", va="center")
arrow((50, 100.6), (50, 99.0))
ax.text(62.5, 97.2, "tokens", fontsize=4.0, zorder=6, **TXT)

# ═══════════════════════ DUE REGIONI DI STAGE, AFFIANCATE ═══════════════════════
stage(25.5, 66, 47, 56, STAGE_A, "Stage 1")
stage(74.5, 66, 47, 56, STAGE_B, "Stage 2")

# ---- Stage 1: ricostruzione -------------------------------------------------
arrow((45, 95.8), (28, 91.5), rad=0.18)
trapez(25.5, 87, 30, 7.0, ENC, ["Decoder  D"], fs=4.6, flip=True)
box(25.5, 78.5, 24, 5.4, INPUT, A_BOX, lines=["ISTFT"], fs=4.6)
arrow((25.5, 83.4), (25.5, 81.3))
series(25.5, 72, 22, 3.2, seed=7)
arrow((25.5, 75.7), (25.5, 74.0))
ax.text(25.5, 67.6, r"$\hat{x}$    reconstruction", fontsize=4.2, zorder=6, **TXT)

box(13.0, 58.5, 18, 5.2, PANEL_B, A_PANEL, lines=["reconstruction"], fs=3.9)
box(38.0, 58.5, 18, 5.2, PANEL_B, A_PANEL, lines=["codebook + commit"], fs=3.4)
arrow((21.5, 69.8), (14.5, 61.4), rad=0.16)
arrow((29.5, 69.8), (36.5, 61.4), rad=-0.16)
loss(24.0, 48.0, "stage 1", fs=7.4)
arrow((13.0, 55.7), (20.0, 50.4), rad=-0.16)
arrow((38.0, 55.7), (28.5, 50.4), rad=0.16)

# ---- Stage 2: prior ---------------------------------------------------------
arrow((55, 95.8), (72, 91.5), rad=-0.18)
snowflake(60.0, 89.0, 1.4)
ax.text(75.5, 89.0, "encoder + VQ frozen", fontsize=4.0, zorder=6, **TXT)

tokens(66.5, 81.5, 8, 3, cell=1.5, seed=11)
for i in (2, 3, 6):                                   # i token mascherati
    ax.add_patch(Rectangle((66.5 + i * 1.5, 81.5), 1.38, 4.5, fc="#9A9A9A", alpha=0.9,
                           ec="black", lw=LW * 0.35, zorder=5))
ax.text(90.0, 83.7, r"$s_M$", fontsize=5.2, zorder=6, **TXT)
ax.text(60.5, 78.4, "random\nmasking", fontsize=4.0, zorder=6, **TXT)

box(74.5, 72.0, 40, 7.6, PANEL, A_PANEL,
    lines=["Bidirectional Transformer", r"prior  $p_\theta(s \mid s_M)$"], fs=4.2)
arrow((74.5, 80.6), (74.5, 76.2))

box(74.5, 61.0, 32, 5.4, PANEL_B, A_PANEL, lines=["masked-token NLL"], fs=4.2)
arrow((74.5, 68.2), (74.5, 63.9))
loss(73.5, 50.0, "stage 2", fs=7.4)
arrow((74.5, 58.3), (74.5, 53.0))

# ═══════════════════════ BANDA DI INFERENZA ═══════════════════════
ax.plot([4, 96], [34.5, 34.5], color="black", lw=LW * 0.7, ls=(0, (3.2, 2.4)), zorder=2)
ax.text(5.5, 31.6, "Inference", fontsize=6.0, zorder=6,
        family=FONT, style="italic", weight="bold", ha="left", va="center")

series(13, 26, 19, 3.2, seed=21, spike=0.62)
ax.text(13, 21.8, "test window", fontsize=4.0, zorder=6, **TXT)

box(38, 26, 23, 5.8, INPUT, A_BOX, lines=["STFT + E + VQ"], fs=4.2)
snowflake(28.5, 29.4, 1.2)
arrow((23.0, 26), (25.8, 26))

tokens(52.0, 23.9, 7, 3, cell=1.4, seed=13, anomalous=(4, 6))
arrow((50.0, 26), (51.8, 26))
ax.text(56.8, 30.6, "anomalous tokens", fontsize=3.8, zorder=6, **TXT)

box(80, 26, 27, 7.2, PANEL, A_PANEL,
    lines=["prior surprise", r"$a = -\log p_\theta(s \mid s_M)$"], fs=3.8)
arrow((62.0, 26), (66.3, 26))

# ── biforcazione sotto il prior: entrambi i rami NASCONO dal prior. La rilevazione
#    somma le sorprese, la spiegazione ne campiona i token normali (§4 del paper).
arrow((74, 22.3), (30, 18.4), rad=0.10)
arrow((84, 22.3), (78, 18.4), rad=-0.10)

box(30, 15.2, 30, 6.0, VIEW, A_BOX, lines=[r"smoothing over $(c,t)$"], fs=4.0)
box(78, 15.2, 30, 6.0, PANEL_B, A_PANEL, lines=["explainable sampling"], fs=4.0)
arrow((30, 12.1), (30, 9.6))
arrow((78, 12.1), (78, 9.6))

box(30, 6.4, 30, 6.0, PANEL_B, A_PANEL, lines=[r"$a^*(c,t)$    vs    $\theta$"], fs=4.2)
trapez(78, 6.4, 28, 5.4, ENC, ["Decoder  D"], fs=4.2, flip=True)
snowflake(58.5, 6.4, 1.2)

ax.text(30, 1.8, "detection", fontsize=4.6, zorder=6, **TXT)
ax.text(78, 1.8, "counterfactual: likely normal state", fontsize=3.9, zorder=6, **TXT)

fig.savefig("timevqvae_framework.pdf")
fig.savefig("timevqvae_framework_check.png", dpi=260)
print("scritti timevqvae_framework.pdf + _check.png")
