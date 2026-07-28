#!/usr/bin/env python3
"""tau_identity_check.py — GUARD. Does  Σ_τ score_batch(per_rate=True) == score_batch(per_rate=False)?

WHY THIS EXISTS. `detect._fit_threshold_paper` (pipeline/detect.py:539-555) builds the TRAIN
threshold from the per-τ stack summed over τ, while `mixture_eval._score_entity` scores the TEST
set through the `per_rate=False` path. If a scorer breaks that identity, the threshold is fitted
on a DIFFERENT FORMULA than the score it is applied to — and the failure is silent: the script
runs to completion, writes a plausible-looking `vus_pr`, and buries an all-zero detector in every
threshold-dependent metric (precision / recall / f1 / event_* / pate / detection_delay).

That is exactly what `fa_backoff` did until 2026-07-27: it mixed a τ-SUMMED deep NLL
(`score_tokens`, a sum over the n_τ=3 `score_window_size_rates`) against a SINGLE-τ count NLL,
a 3x inflation that both broke the identity and drove the mixture onto the count model at every
λ — collapsing four "λ variants" into one. Mixing per-τ and summing afterwards restores the
identity by construction; this script is the check that it stays restored.

Run it over EVERY combiner before any FA relaunch, not just the defaults.

    TVQ_CONVERGED_ROOT=artifacts/converge60/ckpt python scripts/tau_identity_check.py
    python scripts/tau_identity_check.py --root artifacts/converge60/ckpt --cluster c0

Exit code 0 = every scorer preserves the identity; 1 = at least one is BROKEN (or the
checkpoints needed for the probe are absent). Safe to use as a launch gate.
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pipeline"))
sys.path.insert(0, str(REPO / "scripts"))

from mixture_eval import _build_cfg, MixtureStage2  # noqa: E402
from data import make_dataloaders  # noqa: E402
from stage1 import load_stage1  # noqa: E402
from stage2 import load_stage2  # noqa: E402
from fa_backoff import BackoffStage2, _pooled_position_logp  # noqa: E402
from fa_coldstart import BackoffStage2 as ColdBackoff  # noqa: E402
from fa_dp import CountPriorScorer  # noqa: E402

TOL = 1e-3  # matches the repo's federated unit tests; the identity is exact up to fp32 noise


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.environ.get("TVQ_CONVERGED_ROOT",
                                                     "artifacts/converge60/ckpt"),
                    help="checkpoint root holding <ds>/<cluster>/seed<N>/<arm>/<entity>/")
    ap.add_argument("--dataset", default="wsd_fed")
    ap.add_argument("--cluster", default="c0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arm", default="federated_cb_only")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--n-windows", type=int, default=8, help="test windows to score")
    args = ap.parse_args()

    base = Path(args.root)
    if not base.is_absolute():
        base = REPO / base
    base = base / args.dataset / args.cluster / f"seed{args.seed}" / args.arm
    if not base.is_dir():
        print(f"[tau-check] FAIL: checkpoint dir not found: {base}", file=sys.stderr)
        return 1

    ents = sorted(d.name for d in base.iterdir() if d.is_dir())
    if len(ents) < 3:
        print(f"[tau-check] FAIL: need >=3 entities for the multi-prior combiners, found "
              f"{len(ents)} in {base}", file=sys.stderr)
        return 1

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = _build_cfg(args.dataset, args.batch)
    K = cfg.quantizer.codebook_size

    c0 = copy.deepcopy(cfg)
    c0.dataset.entity_id = ents[0]
    example = next(iter(make_dataloaders(c0, stage="eval").train_loader))["inputs"][:1].cpu()
    shared = load_stage1(base / ents[0] / "stage1.ckpt", cfg, example, device=dev)
    shared.eval()
    logp, _N = _pooled_position_logp(shared, cfg, ents[:2], dev, K)

    priors = []
    for e in ents[:3]:
        ce = copy.deepcopy(cfg)
        ce.dataset.entity_id = e
        s2 = load_stage2(base / e / "stage2.ckpt", ce, stage1_ckpt=base / e / "stage1.ckpt",
                         stage1_example_inputs=example, device=dev)
        s2.prior.eval()
        priors.append(s2.prior)

    p0 = priors[0]
    C, F_, W = int(p0.latent_channels), int(p0.latent_freq), int(p0.latent_time)

    ce = copy.deepcopy(cfg)
    ce.dataset.entity_id = ents[0]
    batch = next(iter(make_dataloaders(ce, stage="eval").test_loader))
    batch["inputs"] = batch["inputs"][:args.n_windows].to(dev)
    if "labels" in batch:
        batch["labels"] = batch["labels"][:args.n_windows]

    logn = torch.tensor([100.0, 200.0, 300.0], device=dev).log()

    # EVERY combiner, not just the defaults: `best` and `target_gate` can pass incidentally
    # when one prior dominates, and will start failing once that stops being true.
    cases = {
        "plain prior (control)":         MixtureStage2(shared, [p0], "mean_nll"),
        "MixtureStage2 mean_nll (3)":    MixtureStage2(shared, priors, "mean_nll"),
        "MixtureStage2 mixture (3)":     MixtureStage2(shared, priors, "mixture"),
        "MixtureStage2 best (3)":        MixtureStage2(shared, priors, "best"),
        "MixtureStage2 nk (3)":          MixtureStage2(shared, priors, "nk", logn=logn),
        "MixtureStage2 target_gate (3)": MixtureStage2(shared, priors, "target_gate", logn=logn),
        "fa_backoff lam=0.5":            BackoffStage2(shared, p0, logp, 0.5, C, F_, W),
        "fa_coldstart lam=0.5":          ColdBackoff(shared, p0, logp, 0.5, C, F_, W),
        "fa_dp lam=0.5":                 CountPriorScorer(shared, logp, C, F_, W, p0, lam=0.5),
        "fa_dp lam=0 (count only)":      CountPriorScorer(shared, logp, C, F_, W, p0, lam=0.0),
    }

    print(f"[tau-check] {args.dataset}/{args.cluster}/seed{args.seed}/{args.arm}  "
          f"entities={ents[:3]}  tol={TOL}")
    print(f"{'scorer':34s} {'max|Σ_τ(T) − F|':>18s} {'mean ratio':>11s}  verdict")
    broken = []
    for name, sc in cases.items():
        with torch.no_grad():
            f = sc.score_batch(batch, per_rate=False).token_scores
            t = sc.score_batch(batch, per_rate=True).token_scores.sum(0)
        d = (t - f).abs().max().item()
        r = (t.mean() / f.mean()).item()
        ok = d < TOL
        if not ok:
            broken.append((name, d))
        print(f"{name:34s} {d:18.6f} {r:11.4f}  {'OK' if ok else 'BROKEN'}")

    if broken:
        print(f"\n[tau-check] {len(broken)} scorer(s) BROKEN — their train threshold is fitted "
              f"on a different formula than their test score:", file=sys.stderr)
        for name, d in broken:
            print(f"  {name}: max deviation {d:.6f}", file=sys.stderr)
        print("  Do NOT launch: threshold-dependent metrics will be silently degenerate.",
              file=sys.stderr)
        return 1

    print("\n[tau-check] all scorers preserve the τ identity.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
