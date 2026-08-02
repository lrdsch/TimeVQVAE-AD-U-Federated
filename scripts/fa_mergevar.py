# ═══════════ RETRACTED / FROZEN — DO NOT RUN — the scripts/fa_*.py suite, 2026-07-27 ═══════════
# CAUSE      mixture_eval._load_pool loads ONE stage-1 (client have[0]'s) and tokenizes EVERY
#            client with it, while `federated_cb_only` federates only the CODEBOOK: encoders stay
#            local and diverge (cross-client token agreement measured 0.0000). Each client's prior
#            is scored on symbols it never saw. Deliberately NOT fixed here — repairing the
#            contamination is the owner's research decision, not a cleanup.
# RESULTS    NONE, ever: zero fa_* rows anywhere under artifacts/ (including the read-only
#            history artifacts/_archive_20260729/) and zero logs under logs/. No number this
#            file could print has ever been measured, so there is nothing here to cite.
# RETRACTED  The claim carried by 13 of the 14 fa_* docstrings — that these numbers sit on "the
#            same axis as the converged local / cb_only / centralized reports" — is FALSE
#            (documentation/RESEARCH_LEDGER.md, Group 4): a deployed cb_only client tokenizes
#            with its OWN encoder, so this layer measures an upper bound no deployment can
#            reach. Marked [RETRACTED] inline below wherever it occurs.
# REOPENING  needs an arm whose encoders are bit-identical across clients (`federated_enc_fedavg`);
#            see documentation/LAUNCH_RUNBOOK.md §5.2b. Entry points are guarded: this suite's
#            launcher scripts/launch_all_fa.sh refuses with exit 2 unless FA_I_KNOW_ITS_SHELVED=1.
# ═══════════════════════════════════════════════════════════════════════════════════════════════
"""
E14 — CODEBOOK MERGE VARIANTS (Federated Analytics, retraining, bounded scope).

The `federated_cb_only` arm builds a SHARED codebook by the Prop.1 sufficient-
statistic merge: every client freezes the broadcast codebook, accumulates raw
per-code (n_j, m_j) against it, and the server sets e_j = Σ_k m_j^k / smoothed(Σ_k n_j^k).
Because that is a SUM of sufficient statistics, a code j is weighted by how many
of each client's windows land on it — i.e. the merge is implicitly n_k-weighted
(a big silo dominates a heavily-used code). This experiment asks whether the
WEIGHTING of the codebook M-step matters for the deployed detector, by re-running
the federated codebook merge with three schemes and, for each, keeping the prior
LOCAL (a fresh stage2 per client on the merged, frozen codebook) exactly like
cb_only — then scoring with detect.py's real machinery.

Variants (all share the SAME federated-trained per-client encoders, so ONLY the
merge scheme differs — a clean ablation of the M-step):

  * suffstat_nk    (control): e_j = Σ_k m_j^k / smoothed(Σ_k n_j^k)   — the current
                   default, recomputed from the FINAL frozen encoders. Big silos /
                   heavily-used codes dominate. == `federated_cb_only`'s merge.
  * uniform        : per-client centroid c_j^k = m_j^k / n_j^k, then e_j = mean over
                   clients-with-support of c_j^k. Every client votes EQUALLY for a
                   code regardless of how many of its windows used it.
  * iterated_lloyd : federated k-means. Start from the suffstat codebook, then run
                   `--lloyd-iters-1` extra M-steps of (each client re-assigns its
                   OWN train windows against the CURRENT shared codebook → new
                   (n,m)) + suffstat merge. Drives the shared codebook toward the
                   pooled k-means fixed point of the (fixed) per-client encodings.

Only the CODEBOOK changes across variants; the encoders (tokenizer front-end) are
the single federated_stage1(suffstat) run, frozen. For each variant we install its
codebook, train one MaskGIT prior per client on that client's own windows (frozen
merged codebook, fully-local prior = cb_only recipe), and score. Everything
downstream (rolling assembly, paper threshold, VUS-PR/AUPRC/PATE) is the EXACT
detect.py machinery via mixture_eval._score_entity — directly comparable to the
converged `federated_cb_only` reports.
  ^^^ [RETRACTED 2026-07-27 — FALSE. See the banner at the top of this file: cb_only shares only
      the codebook, so the common tokenizer this sentence assumes does not exist.]

Decision this gates:
  * uniform ≈ suffstat_nk           → the n_k-weighting of the merge is inert;
                                       Prop.1's data-volume weighting buys nothing.
  * iterated_lloyd ≫ suffstat_nk    → one M-step under-fits; the shared codebook
                                       is worth iterating (a cheap FA-only gain, no
                                       extra weight communication).

Usage (smoke):
  CUDA_VISIBLE_DEVICES=1 python scripts/fa_mergevar.py \
      --dataset wsd_fed --clusters c3 --seeds 0 \
      --s1-rounds 2 --s2-epochs 2 --lloyd-iters 2 --batch 16
Usage (full, bounded):
  CUDA_VISIBLE_DEVICES=1 python scripts/fa_mergevar.py \
      --dataset wsd_fed --clusters c0,c3 --seeds 0,1 \
      --s1-rounds 15 --s2-epochs 15 --lloyd-iters 3 --batch 128
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pipeline"))
sys.path.insert(0, str(REPO / "scripts"))

from config import Config  # noqa: E402
from utils import seed_everything  # noqa: E402
from federated import resolve_clients, federated_stage1, _server_merge  # noqa: E402
from federated_eval import _build_train_stage2  # noqa: E402

# Reuse the tested infra from mixture_eval verbatim.
from mixture_eval import _build_cfg, _score_entity, CONVERGED  # noqa: E402

OUT_ROOT = REPO / "artifacts" / "fed_eval" / "fa_mergevar"
VARIANTS = ["suffstat_nk", "uniform", "iterated_lloyd"]


@torch.no_grad()
def _collect_client_stats(client, device):
    """Per-client raw sufficient statistics (n_j, m_j) against the client's CURRENT
    (installed) shared codebook, using the FINAL frozen encoder.

    Reuses the VQ's own suffstat path (fp32, autocast-disabled inside the quantizer)
    for exactness: encoder in eval() so no BatchNorm/dropout drift, but the inner
    SharedVectorQuantizer's `training` flag flipped on so `_accumulate_round_stats`
    fires. `collect_stats_only` is already True on the federated build, so the
    codebook is NEVER mutated by this pass (no EMA, no k-means re-seed, no dead-code
    expiry) — it only reads (n_j, m_j). Identical to what a round of
    federated_stage1(suffstat) uploads, minus the local gradient step."""
    model, vq = client.model, client.vq
    model.eval()                         # freeze encoder (no BN/EMA/dropout drift)
    assert vq.collect_stats_only, "expected the suffstat federated build (collect_stats_only=True)"
    vq.reset_round_stats()
    prev = bool(vq.training)
    vq.training = True                   # gate accumulation ON; encoder stays eval
    try:
        for batch in client.data.train_loader:
            x = batch["inputs"].to(device, non_blocking=True)
            model(x)                     # forward accumulates (n_j, m_j); codebook untouched
    finally:
        vq.training = prev
    n, m = vq.pull_round_stats()
    return n.detach().cpu(), m.detach().cpu()


@torch.no_grad()
def _collect_all(clients):
    counts, sums = [], []
    for c in clients:
        dev = next(c.model.parameters()).device
        n, m = _collect_client_stats(c, dev)
        counts.append(n); sums.append(m)
    return counts, sums


@torch.no_grad()
def _revive_dead(weight: torch.Tensor, dead: torch.Tensor) -> int:
    """Mirror _server_merge's server-side dead-code revival (data-free, privacy-clean):
    re-seed each dead code from a random LIVE centroid + tiny jitter. In place."""
    n_dead = int(dead.sum())
    if n_dead > 0:
        live = (~dead).nonzero(as_tuple=True)[0]
        if len(live) > 0:
            pick = live[torch.randint(len(live), (n_dead,))]
            weight[dead] = weight[pick] + 0.01 * torch.randn_like(weight[dead])
    return n_dead


@torch.no_grad()
def _merge_suffstat(counts, sums, eps, dead_thr, revive):
    """Control: the Prop.1 n_k-weighted M-step (== federated_cb_only's merge),
    recomputed from the final-encoder stats. _server_merge already revives dead."""
    weight, N, M, n_dead = _server_merge(counts, sums, eps, dead_thr, revive)
    return weight.cpu(), N.cpu(), n_dead


@torch.no_grad()
def _merge_uniform(counts, sums, prev_cb, revive):
    """Uniform-weighted merge: e_j = mean over clients-with-support of (m_j^k / n_j^k).
    Every client contributes ONE vote per code it uses, independent of its volume —
    the direct foil to the n_k-weighted suffstat merge. Codes no client used keep the
    previous codebook value, then get the SAME dead-code revival as suffstat."""
    K, D = prev_cb.shape
    acc = torch.zeros(K, D)
    support = torch.zeros(K)
    for n, m in zip(counts, sums):
        has = n > 0
        if has.any():
            acc[has] += m[has] / n[has].unsqueeze(1)     # this client's per-code centroid
            support[has] += 1.0
    weight = prev_cb.detach().cpu().clone()
    live = support > 0
    weight[live] = acc[live] / support[live].unsqueeze(1)
    N = torch.stack(list(counts), dim=0).sum(dim=0)
    dead = ~live                                          # no client assigned this code
    n_dead = _revive_dead(weight, dead) if revive else int(dead.sum())
    return weight, N.cpu(), n_dead


@torch.no_grad()
def _install(clients, cb: torch.Tensor):
    """Broadcast one codebook to every client's stage1 (per client device)."""
    for c in clients:
        dev = next(c.model.parameters()).device
        c.vq.set_codebook(cb.to(dev))


@torch.no_grad()
def _merge_iterated_lloyd(clients, cb_suff, counts0, sums0, eps, dead_thr, revive, n_iters):
    """Federated Lloyd: iterate assign-on-local + suffstat merge. Iteration 1 is the
    suffstat codebook (from counts0/sums0 already collected against the base codebook);
    each further iteration re-installs the current codebook, has every client RE-ASSIGN
    its own train windows against it (new n_j, m_j — assignments move as the codebook
    moves), and merges again. Converges toward the pooled k-means fixed point of the
    fixed per-client encodings."""
    cb = cb_suff.clone()                                 # iteration 1 (reuses counts0/sums0)
    for it in range(1, max(1, n_iters)):
        _install(clients, cb)
        counts, sums = _collect_all(clients)             # re-assign against current cb
        cb, _, n_dead = _merge_suffstat(counts, sums, eps, dead_thr, revive)
        print(f"    [iterated_lloyd] M-step {it + 1}/{n_iters}: revived {n_dead} dead codes")
    return cb


def _cb_only_metric(ds, cluster, seed, entity, key="vus_pr"):
    p = CONVERGED / ds / cluster / f"seed{seed}" / "federated_cb_only" / entity / "report.json"
    if not p.exists():
        return None
    try:
        return float(json.loads(p.read_text()).get(key))
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="wsd_fed")
    ap.add_argument("--clusters", default="all",
                    help="comma list or 'all' (discovered from converged tree)")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--variants", default=",".join(VARIANTS),
                    help="subset of: " + ",".join(VARIANTS))
    ap.add_argument("--s1-rounds", type=int, default=15, help="federated stage1 rounds")
    ap.add_argument("--local-epochs", type=int, default=1, help="local epochs per federated round")
    ap.add_argument("--s2-epochs", type=int, default=15, help="per-client stage2 (prior) epochs")
    ap.add_argument("--lloyd-iters", type=int, default=3, help="iterated_lloyd: total M-steps (>=1)")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = _build_cfg(args.dataset, args.batch)
    K = cfg.quantizer.codebook_size
    eps = cfg.quantizer.eps
    dead_thr = max(1, cfg.quantizer.threshold_ema_dead_code)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variants:
        if v not in VARIANTS:
            raise SystemExit(f"unknown variant {v!r}; choose from {VARIANTS}")

    if args.clusters == "all":
        clusters = sorted(p.name for p in (CONVERGED / args.dataset).iterdir() if p.is_dir())
    else:
        clusters = [c.strip() for c in args.clusters.split(",") if c.strip()]

    print(f"[fa_mergevar] dataset={args.dataset} clusters={clusters} seeds={seeds} "
          f"variants={variants} K={K} s1_rounds={args.s1_rounds} local_epochs={args.local_epochs} "
          f"s2_epochs={args.s2_epochs} lloyd_iters={args.lloyd_iters} device={device}")

    all_records = []
    for cluster in clusters:
        entities = resolve_clients(cfg, None, cluster)
        for seed in seeds:
            cb_dir = CONVERGED / args.dataset / cluster / f"seed{seed}" / "federated_cb_only"
            if not cb_dir.exists():
                print(f"[fa_mergevar] SKIP {cluster} seed{seed}: cb_only run missing")
                continue

            # ── One federated tokenizer training (suffstat) → frozen per-client encoders.
            seed_everything(seed)
            clients, _global_cb, _hist = federated_stage1(
                cfg, entities, rounds=args.s1_rounds, local_epochs=args.local_epochs,
                seed=seed, merge="suffstat")
            client_by_e = {c.entity_id: c for c in clients}
            have = [e for e in entities if e in client_by_e]

            # ── Sufficient statistics from the FINAL frozen encoders (against the
            #    installed suffstat codebook) — the shared substrate for every variant.
            counts0, sums0 = _collect_all(clients)
            base_cb = clients[0].vq.codebook.weight.detach().cpu().clone()

            # ── Build the variant codebooks (encoders held fixed).
            variant_cb = {}
            if "suffstat_nk" in variants or "iterated_lloyd" in variants:
                cb_suff, _, nd = _merge_suffstat(counts0, sums0, eps, dead_thr, revive=True)
                variant_cb["suffstat_nk"] = cb_suff
                print(f"[fa_mergevar] {cluster} s{seed}: suffstat_nk merge (revived {nd})")
            if "uniform" in variants:
                cb_uni, _, nd = _merge_uniform(counts0, sums0, base_cb, revive=True)
                variant_cb["uniform"] = cb_uni
                print(f"[fa_mergevar] {cluster} s{seed}: uniform merge (revived {nd})")
            if "iterated_lloyd" in variants:
                cb_iter = _merge_iterated_lloyd(
                    clients, variant_cb["suffstat_nk"], counts0, sums0,
                    eps, dead_thr, True, args.lloyd_iters)
                variant_cb["iterated_lloyd"] = cb_iter
                # drift vs the 1-step suffstat codebook — a scalar sanity that iteration moved it
                drift = float((cb_iter - variant_cb["suffstat_nk"]).norm() /
                              (variant_cb["suffstat_nk"].norm() + 1e-8))
                print(f"[fa_mergevar] {cluster} s{seed}: iterated_lloyd ({args.lloyd_iters} steps) "
                      f"rel-drift vs suffstat={drift:.3f}")

            # ── For each variant: install codebook, train a LOCAL prior per client, score.
            for v in variants:
                cb = variant_cb[v]
                out_dir = OUT_ROOT / args.dataset / cluster / f"seed{seed}" / v
                for e in have:
                    c = client_by_e[e]
                    dev = next(c.model.parameters()).device
                    c.vq.set_codebook(cb.to(dev))        # freeze THIS variant's shared codebook

                    ce = copy.deepcopy(cfg); ce.dataset.entity_id = e
                    ex = next(iter(c.data.train_loader))["inputs"][:1]
                    s2 = _build_train_stage2(ce, c.model, c.data.train_loader,
                                             args.s2_epochs, cfg.training.lr, dev, ex, tag=f"{v[:4]}:{e}")
                    s2.stage1.eval(); s2.prior.eval()

                    rep = _score_entity(s2.stage1, s2, cfg, e)
                    d = out_dir / e; d.mkdir(parents=True, exist_ok=True)
                    (d / "report.json").write_text(json.dumps(rep, indent=2))
                    rec = {"_arm": f"fa_mergevar_{v}", "_cluster": cluster,
                           "_seed": seed, "_entity": e,
                           **{k: float(rep[k]) for k in ("auroc", "auprc", "vus_pr", "pate_f1")
                              if isinstance(rep.get(k), (int, float)) and np.isfinite(rep.get(k))}}
                    all_records.append(rec)
                    base = _cb_only_metric(args.dataset, cluster, seed, e, "vus_pr")
                    dv = (rep.get("vus_pr", float("nan")) - base) if base is not None else float("nan")
                    print(f"  [{v} {cluster} s{seed}] {e}: "
                          f"vus_pr={rep.get('vus_pr', float('nan')):.3f} "
                          f"auprc={rep.get('auprc', float('nan')):.3f} "
                          f"pate_f1={rep.get('pate_f1', float('nan')):.3f} "
                          f"(cb_only vus={base if base is None else round(base, 3)}, Δ={dv:+.3f})")

                    del s2
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            del clients, client_by_e
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Refuse to truncate the ledger to nothing (same guard as mixture_eval:295-303). The file
    # is opened in "w", so a run that scored zero entities would OVERWRITE a good records file
    # with 0 bytes — and a 0-byte ledger is indistinguishable from "never run" for every
    # downstream reducer.
    if not all_records:
        raise SystemExit(
            "[fa_mergevar] produced 0 records — refusing to write an empty ledger.\n"
            "  Every (cluster, seed) was skipped: the per-arm checkpoints are missing.\n"
            "  Fix the inputs; do not let this overwrite an existing records file."
        )

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rec_path = OUT_ROOT / f"records_{args.dataset}.jsonl"
    with rec_path.open("w") as fh:
        for r in all_records:
            fh.write(json.dumps(r) + "\n")
    print(f"[fa_mergevar] wrote {len(all_records)} records -> {rec_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
