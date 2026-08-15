#!/usr/bin/env python3.10
"""Figura d'insieme: dallo split federato alla rilevazione al counterfactual.

Una sola serie, una sola configurazione, quattro fasce impilate sullo stesso
asse dei tempi:

  (a) la serie intera: il segmento di train tagliato nelle cinque fette dei
      client (10/10/20/20/30 %) piu' la coda di validazione, e a seguire il
      test condiviso con l'anomalia etichettata;
  (b) i cinque profili di punteggio, uno per client, allineati al timestamp
      del pannello (a). Su ogni riga: il punteggio sulla PROPRIA fetta di
      train (fioco, sotto il proprio blocco), il punteggio sul test condiviso,
      e la posizione restituita al protocollo dell'archivio (top-1) con la
      sua tolleranza di +-64 campioni;
  (c) lo zoom sull'anomalia con il counterfactual del client 1 sovrapposto al
      segnale osservato.

Tutto viene da file gia' su disco: nessun modello viene eseguito.
  - segnale ed etichette:  data/raw/<build>/{train,test,test_label}/<eid>.npy
  - punteggi per timestep: artifacts/runs/<tag>/ckpt/.../<eid>/scores.npz
  - counterfactual:        evidence/cf_quality/arrays/<arm>_<series>.npz
    (dump di scripts/cf_quality.py: `x` finestra normalizzata, `d` = x_cf - x)

La finestra e' normalizzata z-score, quindi per riportare il counterfactual
nelle unita' del segnale basta l'affine della finestra stessa: la z-norm per
finestra annulla esattamente lo scaler per-entita' (vedi CLAUDE.md).

    python3.10 scripts/fig_overview.py --out documentation/fig_overview.pdf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
import numpy as np                                    # noqa: E402
from matplotlib.patches import ConnectionPatch        # noqa: E402

matplotlib.rcParams["pdf.fonttype"] = 42

REPO = Path(__file__).resolve().parent.parent

# La figura del paper NON esce con i default: la serie e' ucr_001 e il dump del
# counterfactual e' quello "explainable" (maschera dalla soglia del client, 5 draws),
# non quello label-driven usato in §Counterfactual quality. Comando esatto:
#
#   python scripts/fig_overview.py --series ucr_001 \
#          --cf-npz evidence/cf_quality/arrays/explainable_A2_ucr_001.npz
#
# Con i default esce ucr_011 con un solo draw, che contraddice la didascalia.

# accenti della figura-framework (documentation/framework_style/STYLE.md);
# il rosso resta riservato all'anomalia, quindi i client prendono gli altri.
CLIENT_C = ["#4567A9", "#D17920", "#2B7B4F", "#7B3A99", "#17868F"]
ANOM_C = "#B0302F"
SIG_C = "#1A1A1A"
FS = 6.6            # corpo del testo della figura
FS_S = 5.9          # etichette piccole


def envelope(y: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Decimazione min/max: stessa silhouette, ~n*2 vertici invece di len(y)."""
    if y.size <= 2 * n:
        return np.arange(y.size, dtype=float), y
    edges = np.linspace(0, y.size, n + 1).round().astype(int)
    xs, ys = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        seg = y[a:b]
        i0, i1 = int(seg.argmin()), int(seg.argmax())
        for i in sorted((i0, i1)):
            xs.append(a + i)
            ys.append(seg[i])
    return np.asarray(xs, dtype=float), np.asarray(ys)


def spans(mask: np.ndarray) -> list[tuple[int, int]]:
    """Intervalli [a, b) contigui dove `mask` e' vera."""
    d = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def col_spans(colmask: np.ndarray, t_len: int) -> list[tuple[int, int]]:
    """Colonne latenti riscritte -> intervalli di timestep (come cf_figure.py)."""
    edges = np.linspace(0, t_len, colmask.shape[0] + 1).round().astype(int)
    out: list[tuple[int, int]] = []
    for c in np.flatnonzero(colmask):
        a, b = int(edges[c]), int(edges[c + 1])
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default="ucr_011")
    ap.add_argument("--build", default="ucr_split_w2p")
    ap.add_argument("--tag", default="zn_a2")
    ap.add_argument("--arm", default="federated_enc_fedavg_bn-shared_prior-partial")
    ap.add_argument("--cf-npz", default=None,
                    help="default: evidence/cf_quality/arrays/A2_<series>.npz")
    ap.add_argument("--cf-client", type=int, default=0)
    ap.add_argument("--cf-windows", default="3,4", help="indici nel dump")
    ap.add_argument("--zoom-pad", type=int, default=260,
                    help="contesto attorno alla regione mostrata, in campioni")
    ap.add_argument("--tolerance", type=int, default=64)
    ap.add_argument("--out", default="documentation/fig_overview.pdf")
    a = ap.parse_args()

    series, build = a.series, a.build
    raw = REPO / "data" / "raw" / build
    meta = json.loads((raw / "metadata.json").read_text())
    ents = [e for e in meta["entities"] if e["series"] == series]
    ents.sort(key=lambda e: e["partition"])
    n_cli = len(ents)

    test = np.load(raw / "test" / f"{series}_p0.npy")[:, 0].astype(float)
    labels = np.load(raw / "test_label" / f"{series}_p0.npy").astype(int)
    shards = [np.load(raw / "train" / f"{e['entity_id']}.npy")[:, 0].astype(float) for e in ents]
    train_full = np.concatenate(shards)
    n_train_shards = train_full.size
    # la coda di validazione non e' su disco come split proprio: e' il tratto
    # fra la fine dell'ultima fetta e l'inizio del test nella serie originale.
    n_val = int(ents[0]["val_length"])
    val = np.load(raw / "val" / f"{ents[0]['entity_id']}.npy")[:, 0].astype(float)
    assert val.size == n_val, (val.size, n_val)

    whole = np.concatenate([train_full, val, test])
    t0_test = n_train_shards + n_val                     # inizio del test in coordinate globali
    T = whole.size

    # ── punteggi per client ─────────────────────────────────────────────────
    base = REPO / "artifacts" / "runs" / a.tag / "ckpt" / build / series / "seed0" / a.arm
    sc, tr_sc, top1, thr, flagged = [], [], [], [], []
    for e in ents:
        d = base / e["entity_id"]
        z = np.load(d / "scores.npz")
        sc.append(z["test_scores"].astype(float))
        tr_sc.append(z["train_scores"].astype(float))
        top1.append(int(z["test_scores"].argmax()))
        # soglia del metodo originale, fittata sul solo train: e' quella con cui
        # il modello dichiara una regione anomala.
        thr.append(float(json.loads((d / "report.json").read_text())["threshold"]))
        flagged.append(spans(sc[-1] > thr[-1]))
        assert np.array_equal(z["test_labels"].astype(int), labels)
    print("frazione di test sopra soglia: " +
          "  ".join(f"p{i} {(s > t).mean():.3f}" for i, (s, t) in enumerate(zip(sc, thr))))

    gt = spans(labels > 0)
    assert gt, "nessuna anomalia etichettata"

    # ── counterfactual ──────────────────────────────────────────────────────
    cf_npz = Path(a.cf_npz) if a.cf_npz else (
        REPO / "evidence" / "cf_quality" / "arrays" / f"A2_{series}.npz")
    zc = np.load(cf_npz)
    explainable = "kind" in zc.files          # dump di scripts/cf_explainable.py
    starts, cols = zc["starts"], zc["cols"]
    xw = zc["x"]
    w_len = xw.shape[-1]
    wins = (list(range(len(starts))) if explainable
            else [int(w) for w in a.cf_windows.split(",")])
    if explainable:
        a.cf_client = int(zc["client"])
    cf_pieces = []
    for w in wins:
        s = int(starts[w])
        obs = test[s:s + w_len]
        mu, sd = obs.mean(), obs.std()        # ritorno alle unita' del segnale
        if explainable:
            draws = mu + sd * zc["cf"][:, w, 0]            # (K, T)
            piece = {"cf": np.median(draws, axis=0),
                     "lo": draws.min(axis=0), "hi": draws.max(axis=0),
                     "n_draws": draws.shape[0]}
        else:
            piece = {"cf": mu + sd * (xw[w, 0] + zc["d"][a.cf_client, w, 0]),
                     "lo": None, "hi": None, "n_draws": 1}
        piece["start"] = s
        piece["rewritten"] = [(s + p, s + q) for p, q in col_spans(cols[w], w_len)]
        cf_pieces.append(piece)

    lo = min(p["start"] for p in cf_pieces) - a.zoom_pad
    hi = max(p["start"] + w_len for p in cf_pieces) + a.zoom_pad
    lo, hi = max(0, lo), min(test.size, hi)

    # tutto in coordinate della serie originale (train seguito da test): sono
    # quelle in cui l'archivio annota l'anomalia.
    gt = [(t0_test + g0, t0_test + g1) for g0, g1 in gt]
    top1 = [t0_test + p for p in top1]
    flagged = [[(t0_test + f0, t0_test + f1) for f0, f1 in fl] for fl in flagged]
    lo, hi = t0_test + lo, t0_test + hi
    for p in cf_pieces:
        p["start"] += t0_test
        p["rewritten"] = [(r0 + t0_test, r1 + t0_test) for r0, r1 in p["rewritten"]]

    # ── figura ──────────────────────────────────────────────────────────────
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif"})
    fig = plt.figure(figsize=(7.16, 4.95))
    outer = fig.add_gridspec(3, 1, height_ratios=[1.15, 3.0, 1.85], left=0.072,
                             right=0.981, top=0.940, bottom=0.078, hspace=0.46)
    ax_s = fig.add_subplot(outer[0])
    gs_b = outer[1].subgridspec(n_cli, 1, hspace=0.0)
    axes_c = [fig.add_subplot(gs_b[i], sharex=ax_s) for i in range(n_cli)]
    ax_z = fig.add_subplot(outer[2])

    # ---- (a) la serie e lo split --------------------------------------------
    x, y = envelope(whole, 2800)
    ax_s.plot(x, y, lw=0.3, color=SIG_C, zorder=3)
    ax_s.set_ylim(whole.min() - 0.02 * np.ptp(whole), whole.max() + 0.16 * np.ptp(whole))
    off = 0
    for i, (e, sh) in enumerate(zip(ents, shards)):
        ax_s.axvspan(off, off + sh.size, color=CLIENT_C[i], alpha=0.17, lw=0, zorder=0)
        ax_s.axvline(off, color="0.6", lw=0.35, zorder=1)
        ax_s.text(off + sh.size / 2, 0.965, f"{i + 1}", transform=ax_s.get_xaxis_transform(),
                  ha="center", va="top", fontsize=FS_S, color=CLIENT_C[i])
        off += sh.size
    ax_s.axvspan(off, off + n_val, color="0.5", alpha=0.22, lw=0, zorder=0)
    ax_s.text(off + n_val / 2, 0.965, "v", transform=ax_s.get_xaxis_transform(),
              ha="center", va="top", fontsize=FS_S, color="0.35")
    ax_s.axvline(t0_test, color="0.25", lw=0.7, zorder=4)
    for g0, g1 in gt:
        ax_s.axvspan(g0, g1, color=ANOM_C, alpha=0.35, lw=0, zorder=2)

    ax_s.text(0.0, 1.05,
              "(a)  the series: five disjoint client shards (10/10/20/20/30 %), "
              "a validation tail (v), and the test segment shared by the five clients",
              transform=ax_s.transAxes, ha="left", va="bottom", fontsize=FS)
    ax_s.text(t0_test + 0.008 * T, 0.965, "test", transform=ax_s.get_xaxis_transform(),
              ha="left", va="top", fontsize=FS_S, color="0.35")
    ax_s.annotate("labelled anomaly", xy=((gt[0][0] + gt[0][1]) / 2, 0.92),
                  xycoords=("data", "axes fraction"), xytext=(14, 0), textcoords="offset points",
                  ha="left", va="center", fontsize=FS_S, color=ANOM_C,
                  arrowprops=dict(arrowstyle="-", lw=0.5, color=ANOM_C, shrinkA=0, shrinkB=1))
    ax_s.set_ylabel("signal", fontsize=FS, labelpad=2)

    # ---- (b) un profilo di punteggio per client -----------------------------
    off = 0
    for i, (ax, e, sh) in enumerate(zip(axes_c, ents, shards)):
        c = CLIENT_C[i]
        xs, ys = envelope(sc[i], 2000)
        ax.plot(t0_test + xs, ys, lw=0.4, color=c, zorder=3)
        xt, yt = envelope(tr_sc[i], 260)
        ax.plot(off + xt, yt, lw=0.35, color=c, alpha=0.5, zorder=3)
        ax.axvspan(off, off + sh.size, color=c, alpha=0.10, lw=0, zorder=0)
        top = float(max(sc[i].max(), tr_sc[i].max()))
        ax.set_ylim(0, top * 1.26)
        for f0, f1 in flagged[i]:                       # regioni dichiarate anomale
            ax.axvspan(f0, f1, color=c, alpha=0.22, lw=0, zorder=1)
        ax.axhline(thr[i], color="0.30", lw=0.45, ls=(0, (2.6, 1.6)), zorder=4)
        for g0, g1 in gt:
            ax.axvspan(g0, g1, color=ANOM_C, alpha=0.45, lw=0, zorder=2)
        ax.plot([top1[i]], [top * 1.19], marker="v", ms=2.4, color=c, zorder=5, clip_on=False)
        ax.set_ylabel(f"{i + 1}", fontsize=FS_S, color=c, rotation=0,
                      ha="right", va="center", labelpad=4)
        ax.set_yticks([])
        off += sh.size

    for ax in [ax_s] + axes_c:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xlim(0, T)
        ax.tick_params(labelsize=FS_S, length=2)
    for ax in [ax_s] + axes_c[:-1]:
        ax.tick_params(labelbottom=False)
    ax_s.set_yticks([])
    axes_c[-1].set_xlabel("time step", fontsize=FS, labelpad=1.0)
    axes_c[0].annotate(
        "(b)  anomaly score of each client, numbered as in (a): faint trace = the client's own "
        "shard, dashes = its train-fitted threshold,\nshaded = what that threshold declares "
        "anomalous, marker = the single location the client returns",
        xy=(0.0, 1.10), xycoords="axes fraction", ha="left", va="bottom", fontsize=FS,
        linespacing=1.3)
    axes_c[0].text(0.004, thr[0], "threshold", transform=axes_c[0].get_yaxis_transform(),
                   ha="left", va="bottom", fontsize=FS_S - 0.5, color="0.30")

    # ---- (c) zoom con il counterfactual -------------------------------------
    tt = np.arange(lo, hi)
    cc = CLIENT_C[a.cf_client]
    for p in cf_pieces:
        for r0, r1 in p["rewritten"]:
            ax_z.axvspan(r0, r1, color="0.90", lw=0, zorder=0)
    ax_z.plot(tt, test[lo - t0_test:hi - t0_test], lw=0.7, color=SIG_C, zorder=3,
              label="observed signal")
    first = True
    for p in cf_pieces:
        s = p["start"]
        tw = np.arange(s, s + w_len)
        inside = np.zeros(w_len, dtype=bool)
        for r0, r1 in p["rewritten"]:
            inside |= (tw >= r0) & (tw < r1)
        # contesto: il decoder ripassa anche i token NON riscritti, ed e' solo
        # ricostruzione. Tenerlo distinto e' l'unico modo di non farlo leggere
        # come parte della riparazione.
        ctx = np.where(inside, np.nan, p["cf"])
        rew = np.where(inside, p["cf"], np.nan)
        if p["lo"] is not None:
            ax_z.fill_between(tw, p["lo"], p["hi"], color=cc, alpha=0.22, lw=0, zorder=3,
                              label=f"spread of the {p['n_draws']} draws" if first else None)
        med = " (median)" if p["n_draws"] > 1 else ""
        ax_z.plot(tw, ctx, lw=0.6, color=cc, alpha=0.65, ls=(0, (2.2, 1.6)), zorder=4,
                  label="counterfactual: context, only reconstructed" if first else None)
        ax_z.plot(tw, rew, lw=1.15, color=cc, zorder=5,
                  label=f"counterfactual: rewritten tokens{med}" if first else None)
        first = False
        for b in (s, s + w_len):
            ax_z.axvline(b, color="0.62", lw=0.4, ls=(0, (2.4, 1.8)), zorder=2)
    lo_v, hi_v = ax_z.get_ylim()
    rng_v = hi_v - lo_v
    ax_z.set_ylim(lo_v - 0.16 * rng_v, hi_v + 0.98 * rng_v)
    # tre barre a quote diverse: dov'e' l'anomalia vera, dove il modello la
    # dichiara, e dove punta la singola risposta del protocollo dell'archivio.
    xt = ax_z.get_xaxis_transform()
    pad = 0.010 * (hi - lo)
    for g0, g1 in gt:
        if g1 > lo and g0 < hi:
            ga, gb = max(g0, lo), min(g1, hi)
            ax_z.plot([ga, gb], [0.975, 0.975], lw=2.6, color=ANOM_C, solid_capstyle="butt",
                      transform=xt, zorder=5, clip_on=False)
            ax_z.text(ga - pad, 0.975, "labelled anomaly", transform=xt, ha="right",
                      va="center", fontsize=FS_S, color=ANOM_C)
            for g in (ga, gb):
                ax_z.axvline(g, color=ANOM_C, lw=0.5, ls=(0, (1.4, 1.6)), alpha=0.7, zorder=2)
    fz = [(max(f0, lo), min(f1, hi)) for f0, f1 in flagged[a.cf_client] if f1 > lo and f0 < hi]
    for f0, f1 in fz:
        ax_z.plot([f0, f1], [0.885, 0.885], lw=2.6, color=cc, solid_capstyle="butt",
                  transform=xt, zorder=5, clip_on=False)
    if fz:
        ax_z.text(max(f[1] for f in fz) + pad, 0.885,
                  f"declared anomalous by client {a.cf_client + 1}", transform=xt,
                  ha="left", va="center", fontsize=FS_S, color=cc)
    p1 = int(np.median(top1))
    ax_z.plot([p1 - a.tolerance, p1 + a.tolerance], [0.795, 0.795], lw=2.0, color="0.35",
              solid_capstyle="butt", transform=xt, zorder=5, clip_on=False)
    ax_z.plot([p1], [0.795], marker="v", ms=3.0, color="0.2", transform=xt, zorder=6,
              clip_on=False)
    ax_z.text(p1 + a.tolerance + pad, 0.795,
              f"location returned, $\\pm{a.tolerance}$ tolerance",
              transform=xt, ha="left", va="center", fontsize=FS_S, color="0.35")
    ax_z.set_xlim(lo, hi)
    ax_z.set_xlabel("time step", fontsize=FS, labelpad=1.0)
    ax_z.set_ylabel("signal", fontsize=FS, labelpad=2)
    ax_z.set_yticks([])
    ax_z.tick_params(labelsize=FS_S, length=2)
    for s in ("top", "right"):
        ax_z.spines[s].set_visible(False)
    # legenda nell'angolo alto a sinistra, sotto le barre e sopra il segnale:
    # e' l'unica zona del riquadro senza dati.
    ax_z.legend(fontsize=FS_S, frameon=False, ncol=2, loc="upper left",
                bbox_to_anchor=(0.004, 0.74), handlelength=1.8, borderaxespad=0.0,
                handletextpad=0.5, labelspacing=0.35, columnspacing=1.4)
    ax_z.annotate(f"(c)  the same event at scale: the repair client {a.cf_client + 1} proposes "
                  "for it (grey = the tokens it rewrote), and what each party calls anomalous",
                  xy=(0.0, 1.05), xycoords="axes fraction", ha="left", va="bottom",
                  fontsize=FS)

    # collegamento fra la regione anomala e lo zoom
    for xa, xb in ((lo, 0.0), (hi, 1.0)):
        fig.add_artist(ConnectionPatch(
            xyA=(xa, 0.0), coordsA=axes_c[-1].get_xaxis_transform(),
            xyB=(xb, 1.0), coordsB=ax_z.transAxes,
            lw=0.4, color="0.6", ls=(0, (2.4, 1.8))))

    out = REPO / a.out if not Path(a.out).is_absolute() else Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_name(out.stem + "_check.png"), dpi=260)
    print(f"-> {out}  ({out.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
