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
"""E10 — FEDERATED CODEBOOK-USAGE ANALYTICS (Federated Analytics, inference).

[RETRACTED 2026-07-27 — this file is the one fa_* docstring that never carried the "same axis
 as the converged reports" sentence, but the premise it opens with is the same false one: the
 clients do NOT speak the same token language, only the codebook is shared. See banner above.]

Every client's `federated_cb_only` prior speaks the SAME token language (a shared,
suff-stat–merged K=64 codebook). A pure Federated-Analytics object is the per-codeword
USAGE COUNT: how often each of the K codewords is emitted when a client's OWN train
windows are tokenized with the shared stage1. These counts are additive across clients
-> exactly mergeable, no weights, no labels.

The pathology this exploits: a codeword may be DEAD LOCALLY (a client never emits it in
training) yet ALIVE POOLED (it is a perfectly normal token for SOME other client). A
count prior built on the client's OWN usage assigns such a codeword a near-floor
probability -> it FALSE-ALARMS when that codeword shows up at test time, even though it
is globally normal. Federating the usage counts fixes this.

Scoring variants (all pure per-codeword UNIGRAM count priors; the deployed score is the
token-NLL of the observed codeword, position-independent, broadcast to the (B,C,F,W)
grid — the exact detect.py machinery via mixture_eval._score_entity):

  * local        : each client uses its OWN Laplace-smoothed usage prior. Baseline that
                   false-alarms on locally-dead-but-globally-normal codewords.
  * pooled        : the POOLED (federated) Laplace usage prior, shared by all clients. A
                   locally-unseen-but-globally-normal codeword now gets its true pooled
                   normal probability -> no false alarm.
  * usage_aware   : the pooled prior, but codewords that are DEAD POOLED (truly unused by
                   EVERY client — genuinely never-normal) are down-weighted to `dead_floor`
                   instead of the Laplace floor of 1. This is the ONLY difference from
                   `pooled`: active-but-locally-unseen codewords keep their pooled normal
                   mass (no false alarm), while a genuinely-never-seen-anywhere codeword
                   gets a much higher NLL -> a sharper anomaly signal.

Also emits, per (cluster, seed), a usage ANALYTICS json: pooled active/dead counts, per
client how many codewords are dead-locally-but-alive-pooled, and the mean pairwise
active-set Jaccard overlap across clients.

Compare vs cb_only (the deep prior; read from its converged report.json).

Usage (smoke):
  CUDA_VISIBLE_DEVICES=1 python scripts/fa_cbusage.py \
      --dataset wsd_fed --clusters c3 --seeds 0 --variants local,pooled,usage_aware
Usage (full):
  CUDA_VISIBLE_DEVICES=1 python scripts/fa_cbusage.py \
      --dataset wsd_fed --clusters all --seeds 0,1,2
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
from data import make_dataloaders  # noqa: E402
from stage2 import _flatten_token_indices  # noqa: E402
from federated import resolve_clients  # noqa: E402

# Reuse the tested infra from mixture_eval verbatim.
from mixture_eval import _build_cfg, _load_pool, _score_entity, _Out, CONVERGED  # noqa: E402

OUT_ROOT = REPO / "artifacts" / "fed_eval" / "fa_cbusage"


class CountPriorStage2:
    """Quacks like a Stage2System for detect._compute_entities_raw, but scores each
    window under a pure per-codeword UNIGRAM count prior. Tokenises with the shared
    stage1, looks up the observed codeword's log-probability under `logp` (K,), and
    lays the token-NLL out on the (B, C, F, W) grid. `prior_for_asserts` is any eval()
    deep prior — used ONLY to satisfy detect's `.prior.training is False` assert and is
    never called."""

    def __init__(self, shared_stage1, prior_for_asserts, logp: torch.Tensor,
                 C: int, F: int, W: int):
        self.stage1 = shared_stage1          # detect asserts .stage1.training is False
        self.prior = prior_for_asserts       # detect asserts .prior.training is False
        self.logp = logp                     # (K,) log unigram p(codeword)
        self._C, self._F, self._W = int(C), int(F), int(W)
        self._N = self._C * self._F * self._W

    @torch.no_grad()
    def score_batch(self, batch: dict, per_rate: bool = False) -> _Out:
        _, indices, _ = self.stage1.encode_tokens(batch["inputs"])
        tokens = _flatten_token_indices(indices).long()          # (B, N)
        B, N = tokens.shape
        assert N == self._N, f"token count {N} != C*F*W = {self._N}"
        nll = (-self.logp[tokens]).reshape(B, self._C, self._F, self._W)  # (B, C, F, W)
        if per_rate:
            # Count prior is rate-independent: a single τ. detect sums over τ, so the
            # test path (per_rate=False -> (B,C,F,W)) is on the same scale as the
            # train per-τ threshold (n_τ=1).
            return _Out(nll.unsqueeze(0))                        # (1, B, C, F, W)
        return _Out(nll)                                         # (B, C, F, W)


@torch.no_grad()
def _usage_counts(shared_stage1, cfg: Config, entities: list[str], device, K: int):
    """Per-codeword usage counts over each client's OWN train windows, tokenized with
    the shared stage1. Returns (per_client {entity -> (K,) counts}, pooled (K,), nwin
    {entity -> #windows}). Additive across clients = a pure Federated-Analytics object."""
    per_client, nwin = {}, {}
    for e in entities:
        ce = copy.deepcopy(cfg); ce.dataset.entity_id = e
        loader = make_dataloaders(ce, stage="stage2").train_loader
        counts = torch.zeros(K, device=device); n = 0
        for batch in loader:
            x = batch["inputs"].to(device, non_blocking=True)
            _, idx, _ = shared_stage1.encode_tokens(x)
            t = _flatten_token_indices(idx).long().reshape(-1)
            counts += torch.bincount(t, minlength=K).float()
            n += x.shape[0]
        per_client[e] = counts
        nwin[e] = n
    pooled = torch.stack([per_client[e] for e in entities], dim=0).sum(dim=0)  # (K,)
    return per_client, pooled, nwin


def _local_logp(counts: torch.Tensor) -> torch.Tensor:
    """Client's OWN Laplace-smoothed unigram log-prob over the K codewords."""
    w = counts + 1.0
    return (w / w.sum()).log()


def _pooled_logp(pooled: torch.Tensor) -> torch.Tensor:
    """Pooled (federated) Laplace-smoothed unigram log-prob, shared by all clients."""
    w = pooled + 1.0
    return (w / w.sum()).log()


def _usage_aware_logp(pooled: torch.Tensor, dead_floor: float) -> torch.Tensor:
    """Pooled Laplace prior, but codewords DEAD POOLED (never emitted by any client —
    genuinely never-normal) get weight `dead_floor` (<< 1) instead of the Laplace floor
    of 1, so they carry a much higher NLL. Identical to `pooled` on active codewords, so
    locally-unseen-but-globally-normal codewords keep their normal pooled mass."""
    w = pooled + 1.0
    dead = pooled == 0
    w = torch.where(dead, torch.full_like(w, float(dead_floor)), w)
    return (w / w.sum()).log()


def _usage_analytics(per_client, pooled, entities: list[str], nwin, K: int) -> dict:
    """Codebook-occupancy analytics: pooled active/dead, per-client dead-local vs
    dead-local-but-pooled-alive, and mean pairwise active-set Jaccard overlap."""
    active = {e: (per_client[e] > 0) for e in entities}
    pooled_active = pooled > 0
    stats = {
        "K": int(K),
        "n_clients": len(entities),
        "n_active_pooled": int(pooled_active.sum().item()),
        "n_dead_pooled": int((~pooled_active).sum().item()),
        "clients": {},
    }
    for e in entities:
        a = active[e]
        dead_local = ~a
        dead_local_alive_pooled = dead_local & pooled_active
        stats["clients"][e] = {
            "n_train_windows": int(nwin[e]),
            "n_active_local": int(a.sum().item()),
            "n_dead_local": int(dead_local.sum().item()),
            "n_dead_local_but_pooled_alive": int(dead_local_alive_pooled.sum().item()),
        }
    ents = list(entities)
    jacc = []
    for i in range(len(ents)):
        for j in range(i + 1, len(ents)):
            ai, aj = active[ents[i]], active[ents[j]]
            inter = float((ai & aj).sum().item())
            uni = float((ai | aj).sum().item())
            jacc.append(inter / uni if uni > 0 else 1.0)
    stats["mean_pairwise_active_jaccard"] = float(np.mean(jacc)) if jacc else 1.0
    # Federation upside: total codewords a client would false-alarm on but are normal pooled.
    stats["total_dead_local_but_pooled_alive"] = int(
        sum(c["n_dead_local_but_pooled_alive"] for c in stats["clients"].values())
    )
    return stats


def _cb_only_vus(dataset: str, cluster: str, seed: int, entity: str):
    p = CONVERGED / dataset / cluster / f"seed{seed}" / "federated_cb_only" / entity / "report.json"
    if p.exists():
        try:
            return json.loads(p.read_text()).get("vus_pr")
        except Exception:
            return None
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="wsd_fed")
    ap.add_argument("--clusters", default="all",
                    help="comma list or 'all' (discovered from converged tree)")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--variants", default="local,pooled,usage_aware",
                    help="subset of {local,pooled,usage_aware}")
    ap.add_argument("--dead_floor", type=float, default=1e-3,
                    help="weight assigned to dead-pooled codewords in usage_aware (<<1)")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = _build_cfg(args.dataset, args.batch)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    known = {"local", "pooled", "usage_aware"}
    for v in variants:
        if v not in known:
            raise SystemExit(f"unknown variant {v!r}; choose from {sorted(known)}")
    if not (0.0 < args.dead_floor < 1.0):
        raise SystemExit(f"--dead_floor must be in (0,1); got {args.dead_floor}")
    K = cfg.quantizer.codebook_size

    if args.clusters == "all":
        clusters = sorted(p.name for p in (CONVERGED / args.dataset).iterdir()
                          if p.is_dir())
    else:
        clusters = [c.strip() for c in args.clusters.split(",") if c.strip()]

    print(f"[fa_cbusage] dataset={args.dataset} clusters={clusters} seeds={seeds} "
          f"variants={variants} dead_floor={args.dead_floor:g} K={K} device={device}")

    all_records = []
    for cluster in clusters:
        entities = resolve_clients(cfg, None, cluster)
        for seed in seeds:
            shared_stage1, pool = _load_pool(args.dataset, cluster, seed, entities, cfg, device)
            if shared_stage1 is None:
                print(f"[fa_cbusage] SKIP {cluster} seed{seed}: cb_only run missing/incomplete")
                continue
            priors, have = pool
            p0 = priors[0]
            C, F_, W = int(p0.latent_channels), int(p0.latent_freq), int(p0.latent_time)
            print(f"[fa_cbusage] {cluster} seed{seed}: {len(priors)} clients {have} "
                  f"(C={C} F={F_} W={W})")

            # Federate the per-codeword usage counts (built ONCE, shared across variants).
            per_client, pooled, nwin = _usage_counts(shared_stage1, cfg, have, device, K)

            # ── Usage analytics ────────────────────────────────────────────────
            stats = _usage_analytics(per_client, pooled, have, nwin, K)
            seed_dir = OUT_ROOT / args.dataset / cluster / f"seed{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            (seed_dir / "analytics.json").write_text(json.dumps(stats, indent=2))
            print(f"  [analytics {cluster} s{seed}] active_pooled={stats['n_active_pooled']}/{K} "
                  f"dead_pooled={stats['n_dead_pooled']} "
                  f"jaccard={stats['mean_pairwise_active_jaccard']:.3f} "
                  f"dead_local_but_pooled_alive(total)={stats['total_dead_local_but_pooled_alive']}")
            for e in have:
                c = stats["clients"][e]
                print(f"    {e}: active_local={c['n_active_local']}/{K} "
                      f"dead_local={c['n_dead_local']} "
                      f"dead_local_but_pooled_alive={c['n_dead_local_but_pooled_alive']}")

            # Pooled priors (shared across entities).
            pooled_logp = _pooled_logp(pooled)
            usage_aware_logp = _usage_aware_logp(pooled, args.dead_floor)

            per_variant_vus = {v: [] for v in variants}
            cb_vus = []
            for variant in variants:
                out_dir = seed_dir / variant
                for e in have:
                    if variant == "local":
                        logp = _local_logp(per_client[e])
                    elif variant == "pooled":
                        logp = pooled_logp
                    else:  # usage_aware
                        logp = usage_aware_logp
                    scorer = CountPriorStage2(shared_stage1, p0, logp, C, F_, W)
                    rep = _score_entity(shared_stage1, scorer, cfg, e)
                    d = out_dir / e; d.mkdir(parents=True, exist_ok=True)
                    (d / "report.json").write_text(json.dumps(rep, indent=2))
                    rec = {"_arm": f"fa_cbusage_{variant}", "_cluster": cluster,
                           "_seed": seed, "_entity": e,
                           **{k: float(rep[k]) for k in ("auroc", "auprc", "vus_pr", "pate_f1")
                              if isinstance(rep.get(k), (int, float)) and np.isfinite(rep.get(k))}}
                    all_records.append(rec)
                    v = rep.get("vus_pr", float("nan"))
                    if np.isfinite(v):
                        per_variant_vus[variant].append(v)
                    print(f"  [{variant} {cluster} s{seed}] {e}: "
                          f"vus_pr={v:.3f} "
                          f"auprc={rep.get('auprc', float('nan')):.3f} "
                          f"pate_f1={rep.get('pate_f1', float('nan')):.3f}")

            # cb_only reference (deep prior) for context.
            for e in have:
                cv = _cb_only_vus(args.dataset, cluster, seed, e)
                if cv is not None and np.isfinite(cv):
                    cb_vus.append(cv)
            summ = " ".join(f"{v}={np.mean(per_variant_vus[v]):.3f}"
                            for v in variants if per_variant_vus[v])
            cb_str = f" cb_only={np.mean(cb_vus):.3f}" if cb_vus else ""
            print(f"  [mean vus_pr {cluster} s{seed}] {summ}{cb_str}")

            del shared_stage1, priors, per_client, pooled
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Refuse to truncate the ledger to nothing (same guard as mixture_eval:295-303). The file
    # is opened in "w", so a run that scored zero entities would OVERWRITE a good records file
    # with 0 bytes — and a 0-byte ledger is indistinguishable from "never run" for every
    # downstream reducer.
    if not all_records:
        raise SystemExit(
            "[fa_cbusage] produced 0 records — refusing to write an empty ledger.\n"
            "  Every (cluster, seed) was skipped: the per-arm checkpoints are missing.\n"
            "  Fix the inputs; do not let this overwrite an existing records file."
        )

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rec_path = OUT_ROOT / f"records_{args.dataset}.jsonl"
    with rec_path.open("w") as fh:
        for r in all_records:
            fh.write(json.dumps(r) + "\n")
    print(f"[fa_cbusage] wrote {len(all_records)} records -> {rec_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
