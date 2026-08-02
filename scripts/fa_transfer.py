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
#
# ── E7-SPECIFIC: WITHDRAWN, not merely frozen ──────────────────────────────────────────────────
# All three arms on BOTH datasets are withdrawn (documentation/RESEARCH_LEDGER.md, Group 4): the
# premise below — "token id 17 means the same latent atom for every silo" — is exactly what the
# 0.0000 cross-client token agreement refutes. The "shared codebook = shared language" bridge has
# no valid support anywhere in the repo, and the "report toy instead" mitigation fails by the
# identical mechanism. Unrecoverable without encoder federation.
"""
E7 — CROSS-SILO SHARED-VOCABULARY TRANSFER (Federated Analytics, inference).

In the `federated_cb_only` arm every client trained its OWN deep MaskGIT prior on
a SHARED (suff-stat merged) codebook. Because the codebook is shared, the discrete
tokens are COMPARABLE across clients: token id 17 means the same latent atom for
every silo. This script exploits that interoperability in two ways.

(a) VOCABULARY AGREEMENT (a pure Federated-Analytics measurement — no labels, no
    weights). Each client tokenizes its OWN train windows with the shared stage1
    and reports its token-occupancy histogram over the K codewords. From these we
    compute, per (cluster, seed):
      * per-client active-vocabulary size (# codewords ever used),
      * fraction of the union vocabulary used by ALL clients,
      * mean pairwise Jaccard overlap of active vocabularies,
      * mean pairwise Jensen-Shannon divergence (nats) of the occupancy dists.
    These quantify HOW MUCH a shared token language is actually shared.

(b) PRIOR TRANSFER. Because tokens are comparable, one client's deep prior can be
    APPLIED to another client's test tokens with no adaptation. For each entity we
    score its test data under three scorers and read off VUS-PR (real detect.py
    machinery, identical threshold fitting) to quantify transfer loss:
      * own            : the entity's OWN cb_only deep prior (control == cb_only).
      * cross_best_other: the SINGLE best OTHER client's prior. This is an ORACLE
                          upper bound (best VUS-PR over the other clients — uses
                          labels only to *select*, never to train). For reference
                          the report also stores the label-free choice a router
                          would make from part (a): the JS-nearest other client.
      * mixture_others : the uniform mixture (soft-min, logsumexp) of ALL OTHER
                          clients' priors — a label-free, deploy-today transfer
                          object that never sees the target client's own model.

  transfer_loss = own_vus_pr - transfer_vus_pr  (>0 == the target model is worth
  keeping; ~0 == a foreign / pooled prior is interchangeable == interoperability).

Everything downstream (rolling assembly, paper threshold, VUS-PR / AUPRC / PATE)
is the EXACT detect.py machinery via mixture_eval._score_entity, so the numbers are
directly comparable to the converged local / cb_only / centralized reports.
  ^^^ [RETRACTED 2026-07-27 — FALSE. See the banner at the top of this file: cb_only shares only
      the codebook, so the common tokenizer this sentence assumes does not exist.]

Usage (smoke):
  CUDA_VISIBLE_DEVICES=1 python scripts/fa_transfer.py \
      --dataset wsd_fed --clusters c3 --seeds 0
Usage (full):
  CUDA_VISIBLE_DEVICES=1 python scripts/fa_transfer.py \
      --dataset wsd_fed --clusters all --seeds 0,1,2
  CUDA_VISIBLE_DEVICES=1 python scripts/fa_transfer.py \
      --dataset toy_fed_uni --clusters all --seeds 0,1,2
"""
from __future__ import annotations

import argparse
import copy
import itertools
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
from data import make_dataloaders  # noqa: E402
from stage2 import _flatten_token_indices  # noqa: E402
from federated import resolve_clients  # noqa: E402

# Reuse the tested infra from mixture_eval verbatim.
from mixture_eval import (  # noqa: E402
    _build_cfg, _load_pool, _score_entity, MixtureStage2, CONVERGED,
)

OUT_ROOT = REPO / "artifacts" / "fed_eval" / "fa_transfer"


@torch.no_grad()
def _client_token_occupancy(shared_stage1, cfg: Config, entities: list[str], device, K: int):
    """Per client: RAW token-occupancy counts (K,) over the shared codebook, from
    that client's OWN train windows tokenized with the shared stage1. No labels, no
    gradients — a pure Federated-Analytics histogram. Returns counts (Kc, K) float."""
    rows = []
    for e in entities:
        ce = copy.deepcopy(cfg); ce.dataset.entity_id = e
        loader = make_dataloaders(ce, stage="stage2").train_loader
        counts = torch.zeros(K, device=device)
        for batch in loader:
            x = batch["inputs"].to(device, non_blocking=True)
            _, idx, _ = shared_stage1.encode_tokens(x)
            t = _flatten_token_indices(idx).long().reshape(-1)
            counts += torch.bincount(t, minlength=K).float()
        rows.append(counts)
    return torch.stack(rows, dim=0)                                   # (Kc, K)


def _agreement_metrics(counts: torch.Tensor, entities: list[str], K: int) -> dict:
    """Vocabulary-agreement Federated-Analytics summary from raw occupancy counts.
    Also returns js_nearest[i] = the entity whose occupancy dist is closest (min JS)
    to entity i, used as the label-free router baseline in part (b)."""
    c = counts.detach().cpu().numpy().astype(np.float64)             # (Kc, K)
    Kc = c.shape[0]
    used = c > 0
    vocab_size = used.sum(axis=1)                                     # (Kc,)
    union = used.any(axis=0)
    inter = used.all(axis=0)
    frac_used_by_all = float(inter.sum() / max(1, int(union.sum())))

    # Smoothed occupancy distributions (Laplace), matches _client_token_stats.
    dist = (c + 1.0) / (c.sum(axis=1, keepdims=True) + K)            # (Kc, K)

    def _js(pi, pj):
        m = 0.5 * (pi + pj)
        kl = lambda p: float(np.sum(p * np.log(p / m)))
        return 0.5 * (kl(pi) + kl(pj))

    js = np.zeros((Kc, Kc), dtype=np.float64)
    jac = np.zeros((Kc, Kc), dtype=np.float64)
    for i, j in itertools.combinations(range(Kc), 2):
        d = _js(dist[i], dist[j]); js[i, j] = js[j, i] = d
        u = int((used[i] | used[j]).sum())
        jac[i, j] = jac[j, i] = float((used[i] & used[j]).sum() / max(1, u))

    pair_idx = list(itertools.combinations(range(Kc), 2))
    mean_js = float(np.mean([js[i, j] for i, j in pair_idx])) if pair_idx else 0.0
    mean_jac = float(np.mean([jac[i, j] for i, j in pair_idx])) if pair_idx else 1.0

    # Label-free router: nearest OTHER client by occupancy JS.
    js_nearest = {}
    for i in range(Kc):
        others = [j for j in range(Kc) if j != i]
        j_star = min(others, key=lambda j: js[i, j])
        js_nearest[entities[i]] = entities[j_star]

    return {
        "n_clients": Kc,
        "K": int(K),
        "entities": list(entities),
        "vocab_size_per_client": [int(v) for v in vocab_size],
        "mean_vocab_size": float(vocab_size.mean()),
        "codebook_utilization_per_client": [float(v / K) for v in vocab_size],
        "frac_codewords_used_by_all": frac_used_by_all,
        "mean_pairwise_jaccard": mean_jac,
        "mean_pairwise_js_nats": mean_js,
        "pairwise_js_nats": js.tolist(),
        "pairwise_jaccard": jac.tolist(),
        "js_nearest_other": js_nearest,
    }


def _rec(variant: str, cluster: str, seed: int, entity: str, rep: dict) -> dict:
    return {"_arm": f"fa_transfer_{variant}", "_cluster": cluster, "_seed": seed,
            "_entity": entity,
            **{k: float(rep[k]) for k in ("auroc", "auprc", "vus_pr", "pate_f1")
               if isinstance(rep.get(k), (int, float)) and np.isfinite(rep.get(k))}}


def _write_report(out_dir: Path, entity: str, rep: dict):
    d = out_dir / entity; d.mkdir(parents=True, exist_ok=True)
    (d / "report.json").write_text(json.dumps(rep, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="wsd_fed")
    ap.add_argument("--clusters", default="all",
                    help="comma list or 'all' (discovered from converged tree)")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = _build_cfg(args.dataset, args.batch)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    K = cfg.quantizer.codebook_size

    if args.clusters == "all":
        clusters = sorted(p.name for p in (CONVERGED / args.dataset).iterdir()
                          if p.is_dir())
    else:
        clusters = [c.strip() for c in args.clusters.split(",") if c.strip()]

    print(f"[fa_transfer] dataset={args.dataset} clusters={clusters} seeds={seeds} "
          f"K={K} device={device}")

    all_records = []
    for cluster in clusters:
        entities = resolve_clients(cfg, None, cluster)
        for seed in seeds:
            shared_stage1, pool = _load_pool(args.dataset, cluster, seed, entities, cfg, device)
            if shared_stage1 is None:
                print(f"[fa_transfer] SKIP {cluster} seed{seed}: cb_only run missing/incomplete")
                continue
            priors, have = pool
            prior_by_entity = dict(zip(have, priors))
            seed_dir = OUT_ROOT / args.dataset / cluster / f"seed{seed}"
            print(f"[fa_transfer] {cluster} seed{seed}: {len(priors)} priors over {have}")

            # ---- (a) vocabulary agreement (Federated-Analytics measurement) ----
            counts = _client_token_occupancy(shared_stage1, cfg, have, device, K)
            agr = _agreement_metrics(counts, have, K)
            seed_dir.mkdir(parents=True, exist_ok=True)
            (seed_dir / "agreement.json").write_text(json.dumps(agr, indent=2))
            print(f"  [agreement] mean_vocab={agr['mean_vocab_size']:.1f}/{K} "
                  f"used_by_all={agr['frac_codewords_used_by_all']:.3f} "
                  f"jaccard={agr['mean_pairwise_jaccard']:.3f} "
                  f"JS={agr['mean_pairwise_js_nats']:.4f} nats")

            # ---- (b) prior transfer ----
            for e in have:
                own_prior = prior_by_entity[e]
                others = [o for o in have if o != e]

                # own (control == cb_only, recomputed on identical machinery)
                own_scorer = MixtureStage2(shared_stage1, [own_prior], "mixture")
                own_rep = _score_entity(shared_stage1, own_scorer, cfg, e)
                _write_report(seed_dir / "own", e, own_rep)
                all_records.append(_rec("own", cluster, seed, e, own_rep))
                own_vus = own_rep.get("vus_pr", float("nan"))

                # cross: each OTHER client's single prior applied to e's test data
                cross_by_source, cross_reps = {}, {}
                for o in others:
                    sc = MixtureStage2(shared_stage1, [prior_by_entity[o]], "mixture")
                    rep_o = _score_entity(shared_stage1, sc, cfg, e)
                    cross_reps[o] = rep_o
                    v = rep_o.get("vus_pr", float("nan"))
                    cross_by_source[o] = float(v) if np.isfinite(v) else float("nan")

                # cross_best_other = ORACLE best other (max VUS-PR)
                finite = {o: v for o, v in cross_by_source.items() if np.isfinite(v)}
                best_src = max(finite, key=finite.get) if finite else others[0]
                best_rep = dict(cross_reps[best_src])
                js_src = agr["js_nearest_other"][e]
                best_rep.update({
                    "transfer_variant": "cross_best_other",
                    "target_entity": e,
                    "source_entity": best_src,
                    "own_vus_pr": float(own_vus) if np.isfinite(own_vus) else None,
                    "transfer_loss_vus_pr": (float(own_vus - best_rep.get("vus_pr", float("nan")))
                                             if np.isfinite(own_vus) else None),
                    "cross_vus_pr_by_source": cross_by_source,
                    "js_nearest_source": js_src,
                    "cross_vus_pr_js_nearest": cross_by_source.get(js_src),
                })
                _write_report(seed_dir / "cross_best_other", e, best_rep)
                all_records.append(_rec("cross_best_other", cluster, seed, e, best_rep))

                # mixture_others = uniform mixture of ALL other priors (label-free)
                mix_scorer = MixtureStage2(
                    shared_stage1, [prior_by_entity[o] for o in others], "mixture")
                mix_rep = _score_entity(shared_stage1, mix_scorer, cfg, e)
                mix_rep = dict(mix_rep)
                mix_rep.update({
                    "transfer_variant": "mixture_others",
                    "target_entity": e,
                    "source_entities": others,
                    "own_vus_pr": float(own_vus) if np.isfinite(own_vus) else None,
                    "transfer_loss_vus_pr": (float(own_vus - mix_rep.get("vus_pr", float("nan")))
                                             if np.isfinite(own_vus) else None),
                })
                _write_report(seed_dir / "mixture_others", e, mix_rep)
                all_records.append(_rec("mixture_others", cluster, seed, e, mix_rep))

                print(f"  [{cluster} s{seed}] {e}: own={own_vus:.3f} "
                      f"cross_best({best_src})={cross_by_source[best_src]:.3f} "
                      f"mix_others={mix_rep.get('vus_pr', float('nan')):.3f}")

            del shared_stage1, priors, counts
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Refuse to truncate the ledger to nothing (same guard as mixture_eval:295-303). The file
    # is opened in "w", so a run that scored zero entities would OVERWRITE a good records file
    # with 0 bytes — and a 0-byte ledger is indistinguishable from "never run" for every
    # downstream reducer.
    if not all_records:
        raise SystemExit(
            "[fa_transfer] produced 0 records — refusing to write an empty ledger.\n"
            "  Every (cluster, seed) was skipped: the per-arm checkpoints are missing.\n"
            "  Fix the inputs; do not let this overwrite an existing records file."
        )

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rec_path = OUT_ROOT / f"records_{args.dataset}.jsonl"
    with rec_path.open("w") as fh:
        for r in all_records:
            fh.write(json.dumps(r) + "\n")
    print(f"[fa_transfer] wrote {len(all_records)} records -> {rec_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
