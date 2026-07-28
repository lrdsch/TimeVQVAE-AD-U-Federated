#!/usr/bin/env python3
"""Live 'is the stage-2 prior learning the token grammar?' plot — NON-INVASIVE.

Parses the `[stage2] epoch= step= train/loss= val/loss=` lines that stage2.py
already prints to each per-pipeline log, and plots the masked-token cross-entropy
vs step. Reference line at ln(codebook_size) = the "no grammar" baseline
(uniform prediction over the codebook): val CE dropping BELOW it means the prior
is predicting masked tokens from context, i.e. learning the grammar. The
train–val gap flags memorization.

Re-run anytime to refresh — it just re-reads the logs.

    python scripts/stage2_grammar_live.py --logdir logs/full_3client
"""
from __future__ import annotations
import argparse, glob, math, os, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

LINE = re.compile(r"\[stage2\]\s*epoch=\s*(\d+)\s*step=\s*(\d+)\s*"
                  r"train/loss=([0-9.eE+-]+)\s*val/loss=([0-9.eE+-]+)")
COL = {"toy_fed_uni": "#1c7a4b", "ucr_pool": "#d98a00", "wsd_fed": "#b3323f"}


def parse(logfile):
    steps, tr, va = [], [], []
    with open(logfile, encoding="utf-8", errors="ignore") as fh:
        for ln in fh:
            m = LINE.search(ln)
            if m:
                steps.append(int(m.group(2))); tr.append(float(m.group(3))); va.append(float(m.group(4)))
    return steps, tr, va


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default="logs/full_3client")
    ap.add_argument("--codebook", type=int, default=64, help="codebook_size (no-grammar baseline = ln(K))")
    ap.add_argument("--out", default="plots/recon_demo/stage2_grammar_live.png")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    no_grammar = math.log(args.codebook)

    series = []
    for lf in sorted(glob.glob(f"{args.logdir}/*.log")):
        base = os.path.basename(lf)[:-4]                      # dataset_entity
        ds = next((k for k in COL if base.startswith(k)), base.split("_")[0])
        steps, tr, va = parse(lf)
        if steps:
            series.append((base, ds, steps, tr, va))
    if not series:
        print("[grammar] no stage-2 lines yet in any log — stage-2 hasn't started."); return

    n = len(series)
    fig, axes = plt.subplots(1, n, figsize=(6.4 * n, 4.6), squeeze=False)
    for j, (base, ds, steps, tr, va) in enumerate(series):
        ax = axes[0][j]; col = COL.get(ds, "#444")
        ax.axhline(no_grammar, color="#888", ls="--", lw=1.3,
                   label=f"no-grammar = ln({args.codebook}) = {no_grammar:.3f}")
        ax.plot(steps, tr, color=col, lw=1.0, alpha=0.45, label="train CE")
        ax.plot(steps, va, color=col, lw=2.0, label="val CE (masked-token)")
        last = va[-1]; gap = tr[-1] - va[-1]
        below = no_grammar - last
        ax.annotate(f"val={last:.3f}\n({below:+.3f} vs no-grammar)",
                    xy=(steps[-1], last), xytext=(0.62, 0.72), textcoords="axes fraction",
                    fontsize=9, color=col,
                    arrowprops=dict(arrowstyle="->", color=col, lw=1))
        ax.set_title(f"{base}  —  {'LEARNING' if below > 0.05 else 'flat'}  "
                     f"(train–val gap {gap:+.3f})", fontsize=10, loc="left")
        ax.set_xlabel("stage-2 step"); ax.set_ylabel("masked-token cross-entropy (nats)")
        ax.grid(alpha=.15); ax.legend(fontsize=8, loc="upper right", framealpha=.92)
        ax.set_ylim(top=no_grammar * 1.06)
    fig.suptitle("Stage-2: is the prior learning the token grammar?  "
                 "(val CE below ln(codebook) = predicting masked tokens from CONTEXT)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"[grammar] {len(series)} series plotted -> {args.out}")
    for base, ds, steps, tr, va in series:
        print(f"  {base}: {len(steps)} pts, last val CE={va[-1]:.3f} "
              f"(no-grammar={no_grammar:.3f}, below by {no_grammar-va[-1]:+.3f})")


if __name__ == "__main__":
    main()
