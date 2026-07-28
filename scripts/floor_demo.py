"""FLOOR baseline — live walkthrough on one real wsd_fed client.

Prints a step-by-step numeric trace of the `ma_c` head (centered moving-average
residual, ZERO fitted parameters) travelling the exact same `detect` path the
neural arms use, then renders the five stages as a figure.

Also demonstrates, non-circularly, that sufficient-statistic federation of the
AR head is EXACT: the pooled lag-design Gram built by stacking per-entity design
matrices equals the elementwise sum of the per-client Grams.

Usage:  python scripts/floor_demo.py [--entity kpi_015] [--cluster c0]
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pipeline"))

from config import Config, apply_dataset_overrides, apply_env_overrides  # noqa: E402
import data as D_data                                                    # noqa: E402
import detect as D                                                       # noqa: E402
from federated import resolve_clients                                    # noqa: E402
from metrics_core import segments, vus_metrics                           # noqa: E402

# Validated categorical slots (dataviz reference palette, light mode)
C_SERIES, C_MA, C_STATUS, C_INK, C_MUTED = "#2a78d6", "#eb6834", "#e34948", "#0b0b0b", "#52514e"
K_MA = 10          # the head's only knob: 10 min at dt_sec=60
AR_P = 32          # lag order for the federation-exactness demo


def rule(title: str) -> None:
    print(f"\n\033[1m{'─' * 78}\n{title}\n{'─' * 78}\033[0m")


def build_cfg(ds: str) -> Config:
    cfg = Config()
    cfg.dataset.name = ds
    apply_dataset_overrides(cfg)
    apply_env_overrides(cfg)
    cfg.evaluation.save_plots = False
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# The head: three lines of maths, zero fitted parameters
# ─────────────────────────────────────────────────────────────────────────────
def ma_c_score(x: np.ndarray, k: int = K_MA) -> np.ndarray:
    """s_t = (x_t - MA_k(x)_t)**2 — MA_k is detect's own centered box filter."""
    return (x - D._moving_average_paper(x, k)) ** 2


# ─────────────────────────────────────────────────────────────────────────────
# AR head sufficient statistics (for the federation-exactness demo)
# ─────────────────────────────────────────────────────────────────────────────
def lag_design(x: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    """Z[i, j] = x[i + p - 1 - j] (columns = lag 1..p), y = x[p:]."""
    n = len(x) - p
    idx = np.arange(n)[:, None] + (p - 1 - np.arange(p))[None, :]
    return x[idx], x[p:]


def suffstats(x: np.ndarray, p: int) -> dict:
    Z, y = lag_design(x, p)
    return {"G": Z.T @ Z, "b": Z.T @ y, "q": float(y @ y), "n": len(y)}


def solve_ridge(st: dict, lam_rel: float = 1e-6) -> np.ndarray:
    G, b, n = st["G"], st["b"], st["n"]
    lam = lam_rel * np.trace(G) / G.shape[0]
    return np.linalg.solve(G + lam * np.eye(G.shape[0]), b)


# ─────────────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="wsd_fed")
    ap.add_argument("--entity", default="kpi_015")
    ap.add_argument("--cluster", default="c0")
    ap.add_argument("--out", default="plots/floor_demo")
    args = ap.parse_args()

    cfg = build_cfg(args.dataset)
    W = cfg.dataset.window_length
    S = D._resolve_eval_stride(cfg)
    tol = cfg.evaluation.paper_metrics_tolerance

    rule("STEP 0 — configuration (the conventions that must be matched)")
    print(f"  window_length           W   = {W}")
    print(f"  eval stride             S   = {S}   (detect._resolve_eval_stride)")
    print(f"  metrics tolerance       tol = {tol}   (VUS/PATE buffer, from metadata.json)")
    print(f"  impulse term                = {cfg.scoring.use_impulse_term}   (config.py:298)")
    print(f"  rolling aggregation         = {cfg.scoring.rolling_aggregation!r}  -> no coverage division")
    print(f"  threshold rule              = {cfg.threshold.name} @ q={cfg.threshold.q}")

    rule("STEP 1 — load the client (identical input to every neural arm)")
    c = copy.deepcopy(cfg)
    c.dataset.entity_id = args.entity
    tr, va, te = D_data.load_scaled_records(c)
    xtr = np.asarray(tr[0].X[:, 0], dtype=np.float64)
    xte = np.asarray(te[0].X[:, 0], dtype=np.float64)
    yte = np.asarray(te[0].y, dtype=np.int64)
    segs = segments(yte)
    print(f"  entity {args.entity}: train T={len(xtr)}  test T={len(xte)}  C=1")
    print(f"  per-entity z-score fitted on TRAIN: mean={xtr.mean():+.3e} std={xtr.std():.4f}")
    print(f"  test anomalies: {yte.sum()} points in {len(segs)} segments "
          f"(lengths {[b - a + 1 for a, b in segs]}), rate={yte.mean():.4%}")
    print(f"  FITTED PARAMETERS OF THIS HEAD: 0")

    rule(f"STEP 2 — the head: s_t = (x_t - MA_{K_MA}(x)_t)^2")
    s_te_series = ma_c_score(xte)
    s_tr_series = ma_c_score(xtr)
    print(f"  MA_{K_MA} = detect._moving_average_paper  (centered, edge-clipped kernel)")
    print(f"  test residual score: min={s_te_series.min():.3e} median={np.median(s_te_series):.3e} "
          f"max={s_te_series.max():.3e}")
    in_a = s_te_series[yte == 1]
    out_a = s_te_series[yte == 0]
    print(f"  median score INSIDE anomalies  = {np.median(in_a):.4e}")
    print(f"  median score OUTSIDE anomalies = {np.median(out_a):.4e}"
          f"   -> ratio {np.median(in_a) / max(np.median(out_a), 1e-30):.1f}x")

    rule("STEP 3 — accumulate over sliding windows, exactly as detect does")
    stages = {}
    for tag, recs, s_series in (("train", tr, s_tr_series), ("test", te, s_te_series)):
        ents = D._init_entity(recs)
        ds = D_data.SlidingWindowDataset(recs, W, S)
        nwin = 0
        for wi in ds.indices:
            e = ents[wi.record_index]
            e["channel_sum"][wi.start:wi.stop, 0] += s_series[wi.start:wi.stop]
            e["coverage"][wi.start:wi.stop] += 1.0
            nwin += 1
        for e in ents:
            e["channel_scores"] = D._assemble_rolling(
                e["channel_sum"], e["coverage"], cfg.scoring.rolling_aggregation)
        stages[tag] = (ents, nwin)
        print(f"  {tag:5s}: {nwin} windows at stride {S}; "
              f"coverage min={ents[0]['coverage'].min():.0f} max={ents[0]['coverage'].max():.0f}")
    assembled = np.asarray(stages["test"][0][0]["channel_scores"][:, 0]).copy()

    rule("STEP 4 — impulse term + channel aggregation (detect._finalize_entities)")
    D._finalize_entities(stages["train"][0], cfg)
    scores, labels = D._finalize_entities(stages["test"][0], cfg)
    raw = np.asarray(stages["test"][0][0]["channel_scores_raw"][:, 0])
    imp = np.asarray(stages["test"][0][0]["channel_scores_impulse"][:, 0])
    print(f"  a_final = 0.5*(raw + MA_{W}(raw))     <- paper Algorithm 1")
    print(f"  raw      max={raw.max():.4e}")
    print(f"  impulse  max={imp.max():.4e}")
    print(f"  combined max={scores.max():.4e}   len={len(scores)} (== test T, no truncation)")

    rule("STEP 5 — threshold from TRAIN only, then metrics")
    thr = D._fit_threshold_paper(stages["train"][0], cfg)
    preds = (scores > thr).astype(np.int64)
    print(f"  thr = quantile(train scores, {cfg.threshold.q}) = {thr:.4f}   (no test data used)")
    print(f"  flagged {preds.sum()} / {len(preds)} test points ({preds.mean():.3%})")
    rep = D._detection_metrics(labels, preds, scores, cfg)
    print(f"\n  \033[1mvus_pr = {rep['vus_pr']:.4f}\033[0m   auprc={rep['auprc']:.4f} "
          f"auroc={rep['auroc']:.4f}  f1={rep['f1']:.4f}")
    v_off = vus_metrics(labels, raw, tol)["vus_pr"]
    print(f"  same head with impulse OFF: vus_pr = {v_off:.4f}   "
          f"-> detect's post-processing is worth {rep['vus_pr'] - v_off:+.4f}")

    # ── federation exactness, non-circular ──────────────────────────────────
    rule(f"STEP 6 — federation of the AR({AR_P}) head is EXACT (independent check)")
    members = resolve_clients(cfg, None, args.cluster)
    st_k, Zs = {}, []
    for e in members:
        ce = copy.deepcopy(cfg); ce.dataset.entity_id = e
        xk = np.asarray(D_data.load_scaled_records(ce)[0][0].X[:, 0], dtype=np.float64)
        st_k[e] = suffstats(xk, AR_P)
        Zs.append(lag_design(xk, AR_P))
        print(f"  {e:10s} n={st_k[e]['n']:6d}  uplink = G({AR_P}x{AR_P})+b+q+n = "
              f"{(AR_P * AR_P + AR_P + 2) * 8 / 1024:.1f} kB")

    # (a) federated: server sums the per-client statistics
    G_fed = sum(st_k[e]["G"] for e in members)
    b_fed = sum(st_k[e]["b"] for e in members)
    w_fed = solve_ridge({"G": G_fed, "b": b_fed, "n": sum(st_k[e]["n"] for e in members)})

    # (b) centralized: INDEPENDENT path -- materialise the pooled design matrix
    Z_pool = np.vstack([z for z, _ in Zs])
    y_pool = np.concatenate([y for _, y in Zs])
    st_pool = {"G": Z_pool.T @ Z_pool, "b": Z_pool.T @ y_pool, "n": len(y_pool)}
    w_pool = solve_ridge(st_pool)

    # (c) fedavg: average the per-client SOLUTIONS
    nk = np.array([st_k[e]["n"] for e in members], dtype=np.float64); nk /= nk.sum()
    w_avg = sum(a * solve_ridge(st_k[e]) for a, e in zip(nk, members))

    dG = np.abs(G_fed - st_pool["G"]).max() / np.abs(st_pool["G"]).max()
    dw = np.abs(w_fed - w_pool).max() / np.abs(w_pool).max()
    print(f"\n  pooled design matrix Z: {Z_pool.shape}  (rows stacked PER ENTITY, never across)")
    print(f"  ||sum_k G_k - G_pool||inf / ||G_pool||inf = {dG:.2e}")
    print(f"  ||w_fed      - w_pool||inf / ||w_pool||inf = {dw:.2e}   <- float64 round-off = EXACT")
    print(f"  ||w_fedavg   - w_pool||   / ||w_pool||     = "
          f"{np.linalg.norm(w_avg - w_pool) / np.linalg.norm(w_pool):.3f}   <- averaging is NOT exact")
    A = st_pool["G"] + (1e-6 * np.trace(st_pool["G"]) / AR_P) * np.eye(AR_P)
    d = w_avg - w_pool
    print(f"  excess objective J(w_avg)-J(w*) = (w_avg-w*)' A (w_avg-w*) = {float(d @ A @ d):.4e}")

    # ── figure ──────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
        "font.size": 9, "axes.labelcolor": C_MUTED, "text.color": C_INK,
        "xtick.color": C_MUTED, "ytick.color": C_MUTED,
        "axes.edgecolor": "#dcdcd8", "axes.linewidth": 0.8,
        "grid.color": "#e8e8e4", "grid.linewidth": 0.6,
    })
    fig = plt.figure(figsize=(13.5, 12.0))
    gs = GridSpec(5, 1, height_ratios=[1.5, 1, 1, 1.3, 1.5], hspace=0.42)
    t = np.arange(len(xte))

    def gt_bands(ax, label=False):
        for i, (a, b) in enumerate(segs):
            ax.axvspan(a, b + 1, color=C_STATUS, alpha=0.16, lw=0,
                       label="anomalia (ground truth)" if (label and i == 0) else None)

    ax = fig.add_subplot(gs[0])
    gt_bands(ax, label=True)
    ax.plot(t, xte, color=C_SERIES, lw=0.9, label=f"serie test {args.entity} (z-score)")
    ax.plot(t, D._moving_average_paper(xte, K_MA), color=C_MA, lw=1.6,
            label=f"MA$_{{{K_MA}}}$ — box centrato, 0 parametri")
    ax.set_title("① Il modello: una media mobile a 10 campioni. Tutto qui.",
                 loc="left", fontsize=11, fontweight="bold")
    ax.set_ylabel("ampiezza")
    ax.legend(loc="upper left", frameon=False, fontsize=8, ncol=3)
    ax.grid(axis="y"); ax.set_xlim(0, len(t))

    ax = fig.add_subplot(gs[1])
    gt_bands(ax)
    ax.plot(t, s_te_series, color=C_SERIES, lw=0.8)
    ax.set_title(r"② Score della testa:  $s_t=(x_t-\mathrm{MA}_{10}(x)_t)^2$",
                 loc="left", fontsize=11, fontweight="bold")
    ax.set_ylabel("residuo²"); ax.set_yscale("log")
    ax.grid(axis="y"); ax.set_xlim(0, len(t))

    ax = fig.add_subplot(gs[2])
    gt_bands(ax)
    ax.plot(t, assembled, color=C_SERIES, lw=0.8)
    ax.set_title(f"③ Assemblato su {stages['test'][1]} finestre scorrevoli (W={W}, stride={S}, somma senza divisione)",
                 loc="left", fontsize=11, fontweight="bold")
    ax.set_ylabel("score accum."); ax.grid(axis="y"); ax.set_xlim(0, len(t))

    ax = fig.add_subplot(gs[3])
    gt_bands(ax, label=True)
    ax.plot(t, raw, color=C_MUTED, lw=0.7, alpha=0.55, label="grezzo")
    ax.plot(t, scores, color=C_SERIES, lw=1.0, label="finale = 0.5·(grezzo + MA$_{128}$(grezzo))")
    ax.axhline(thr, color=C_MA, ls="--", lw=1.4)
    ax.annotate(f" soglia = quantile(train, {cfg.threshold.q}) = {thr:.1f}",
                xy=(len(t) * 0.985, thr), ha="right", va="bottom",
                fontsize=8, color=C_MA, fontweight="bold")
    hit = np.flatnonzero(preds)
    ax.plot(hit, scores[hit], ".", ms=3.0, color=C_STATUS, label=f"rilevati ({len(hit)} punti)")
    ax.set_title(f"④ Impulse term + soglia da TRAIN  →  vus_pr = {rep['vus_pr']:.4f}"
                 f"   (senza impulse: {v_off:.4f})",
                 loc="left", fontsize=11, fontweight="bold")
    ax.set_ylabel("score"); ax.set_yscale("log")
    ax.legend(loc="upper left", frameon=False, fontsize=8, ncol=4)
    ax.grid(axis="y"); ax.set_xlim(0, len(t))

    ax = fig.add_subplot(gs[4])
    a0, b0 = segs[int(np.argmax([b - a for a, b in segs]))]
    lo, hi = max(0, a0 - 400), min(len(t), b0 + 400)
    ax.axvspan(a0, b0 + 1, color=C_STATUS, alpha=0.16, lw=0, label="anomalia (ground truth)")
    ax.plot(t[lo:hi], xte[lo:hi], color=C_SERIES, lw=1.3, label="serie")
    ax.plot(t[lo:hi], D._moving_average_paper(xte, K_MA)[lo:hi], color=C_MA, lw=2.0,
            label=f"MA$_{{{K_MA}}}$")
    ax.fill_between(t[lo:hi], xte[lo:hi], D._moving_average_paper(xte, K_MA)[lo:hi],
                    color=C_SERIES, alpha=0.18, lw=0, label="residuo = il segnale")
    ax.set_title("⑤ Zoom sul segmento anomalo più lungo — la media mobile non lo segue, e quello scarto È il detector",
                 loc="left", fontsize=11, fontweight="bold")
    ax.set_xlabel("timestep"); ax.set_ylabel("ampiezza")
    ax.legend(loc="upper left", frameon=False, fontsize=8, ncol=4)
    ax.grid(axis="y"); ax.set_xlim(lo, hi)

    fig.suptitle(
        f"FLOOR / testa  ma_c  —  {args.dataset} · {args.entity} · 0 parametri fittati · "
        f"percorso detect identico agli arm neurali",
        x=0.007, ha="left", fontsize=13, fontweight="bold", y=0.997)

    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    png = out / f"floor_demo_{args.dataset}_{args.entity}.png"
    fig.savefig(png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  figura -> {png.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
