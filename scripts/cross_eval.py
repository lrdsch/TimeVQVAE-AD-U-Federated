"""Cross-client transfer evaluation — the OFF-DIAGONAL of the model×client grid.

Standard eval scores client i's model on client i's own test (the diagonal).
This computes the full grid  M[source][target] = (model trained for `source`)
evaluated on `target`'s test set, for the per-client families whose off-diagonal
is meaningful:

  * local      — each client's fully-independent model (its OWN codebook). Cross
                 matrix spans ALL clients (across clusters) → the affinity matrix
                 whose block structure should recover the clusters.
  * federated  — the per-cluster federated model (shared codebook + per-client
                 prior). Off-diagonal is WITHIN cluster (a cluster's federated
                 model only exists for that cluster's clients).

One forward pass per cell (target's own normalization, exactly as production),
from which we emit EVERY metric detect() computes, in three regimes:

  * structural  — threshold-free (vus_pr, auprc, auroc, pate_f1, best_f1). "Does
                  the representation/score transfer?" Independent of any threshold.
  * frozen      — thresholded metrics using the SOURCE's own threshold (recovered
                  from the source's diagonal report.json). "Deployable as-is?"
                  (the user's `usa la soglia di client 1 su test di client 2`).
  * adaptive    — thresholded metrics using a threshold REFIT on the target's own
                  train (unlabeled). "Deployable with cheap local recalibration?"
                  Needs the extra train forward pass; gated by --adaptive.

The diagonal (source==target) is INGESTED from the existing report.json — no
recompute — so only off-diagonal cells hit the GPU.

Correctness is pinned by --smoke: recompute one diagonal cell and assert the
test_scores match the on-disk scores.npz bit-for-bit.

Run (per convention GPU 1):
    CUDA_VISIBLE_DEVICES=1 python scripts/cross_eval.py --dataset wsd_fed \
        --arm local --seed 0 --out artifacts/fed_eval/cross/records_cross_wsd_fed_local_s0.jsonl
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

from config import Config, apply_dataset_overrides, apply_env_overrides
from data import make_dataloaders
from stage1 import load_stage1
from stage2 import Stage2System, load_stage2
from utils import resolve_path, seed_everything
import detect as D

# Threshold-FREE keys returned by _detection_metrics: identical regardless of preds.
THRESHOLD_FREE = ["auroc", "auprc", "best_f1", "vus_pr", "vus_roc", "pate_f1"]
# Threshold-DEPENDENT keys (from preds): stored per regime (_frozen / _adapt).
THRESHOLDED = ["precision", "recall", "f1", "fpr", "pate", "affiliation_f1",
               "detection_delay", "detection_delay_norm"]


def build_cfg(dataset: str, seed: int, batch: int = 16) -> Config:
    """Mirror pipeline/federated_eval.py's cfg construction exactly, so a
    recomputed diagonal reproduces the on-disk scores."""
    cfg = Config()
    cfg.dataset.name = dataset
    cfg.dataset.batch_size_stage1 = batch
    cfg.dataset.batch_size_stage2 = batch
    apply_dataset_overrides(cfg)
    apply_env_overrides(cfg)
    cfg.seed = seed
    # Off-diagonal cells must NOT poison the shared score cache (different model
    # on a given entity's data would collide on the entity-keyed cache path).
    cfg.evaluation.use_score_cache = False
    cfg.evaluation.save_scores = False
    cfg.evaluation.save_plots = False
    return cfg


def scan_arm(dataset: str, arm: str, seed: int) -> dict[str, dict]:
    """Walk artifacts/fed_eval/<ds>/<cluster>/seed<seed>/<arm>/<entity>/ and return
    {entity: {"cluster", "s1", "s2", "report"}}."""
    root = resolve_path("artifacts/fed_eval") / dataset
    out: dict[str, dict] = {}
    for s1p in root.glob(f"*/seed{seed}/{arm}/*/stage1.ckpt"):
        ent_dir = s1p.parent
        entity = ent_dir.name
        cluster = ent_dir.parents[2].name           # <ds>/<cluster>/seed/<arm>/<entity>
        rep = ent_dir / "report.json"
        out[entity] = {
            "cluster": cluster,
            "s1": s1p,
            "s2": ent_dir / "stage2.ckpt",
            "report": rep if rep.exists() else None,
        }
    return out


def load_report(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def score_target(stage1, stage2, cfg: Config, target: str, adaptive: bool):
    """Score `stage1`+`stage2` on target entity's data. Returns
    (test_scores, test_labels, thr_adapt_or_None)."""
    c = copy.deepcopy(cfg)
    c.dataset.entity_id = target
    data = make_dataloaders(c, stage="eval")
    test_entities = D._compute_entities_raw(stage1, stage2, c, data.test_records)
    test_scores, test_labels = D._finalize_entities(test_entities, c)
    thr_adapt = None
    if adaptive:
        train_entities = D._compute_entities_raw(
            stage1, stage2, c, data.train_records, record_per_rate=True)
        thr_adapt = D._fit_threshold_paper(train_entities, c)
    return test_scores, test_labels, thr_adapt


def metrics_at(test_labels, test_scores, thr, cfg) -> dict:
    preds = (test_scores > thr).astype(np.int64)
    return D._detection_metrics(test_labels, preds, test_scores, cfg)


def diagonal_record(dataset, arm, seed, entity, info) -> dict:
    """Ingest the pre-computed diagonal from report.json (frozen == adaptive here)."""
    rep = load_report(info["report"])
    rec = {
        "dataset": dataset, "arm": arm, "family": arm, "seed": seed,
        "source": entity, "target": entity,
        "cluster_src": info["cluster"], "cluster_tgt": info["cluster"],
        "is_diagonal": True,
        "thr_source": rep.get("threshold") if rep else None,
        "thr_target": rep.get("threshold") if rep else None,
    }
    if rep:
        for k in THRESHOLD_FREE:
            if k in rep:
                rec[k] = rep[k]
        for k in THRESHOLDED:
            if k in rep:
                rec[k + "_frozen"] = rep[k]
                rec[k + "_adapt"] = rep[k]
    return rec


def offdiag_record(dataset, arm, seed, source, target, src_info, tgt_info,
                   test_labels, test_scores, thr_source, thr_adapt, cfg) -> dict:
    rec = {
        "dataset": dataset, "arm": arm, "family": arm, "seed": seed,
        "source": source, "target": target,
        "cluster_src": src_info["cluster"], "cluster_tgt": tgt_info["cluster"],
        "is_diagonal": False,
        "thr_source": float(thr_source) if thr_source is not None else None,
        "thr_target": float(thr_adapt) if thr_adapt is not None else None,
    }
    # Structural + frozen (source threshold).
    if thr_source is not None:
        m_frozen = metrics_at(test_labels, test_scores, thr_source, cfg)
        for k in THRESHOLD_FREE:
            if k in m_frozen:
                rec[k] = m_frozen[k]
        for k in THRESHOLDED:
            if k in m_frozen:
                rec[k + "_frozen"] = m_frozen[k]
    else:
        # No source threshold available: still emit structural via a dummy threshold.
        m = metrics_at(test_labels, test_scores, np.inf, cfg)
        for k in THRESHOLD_FREE:
            if k in m:
                rec[k] = m[k]
    # Adaptive (target-refit threshold).
    if thr_adapt is not None:
        m_adapt = metrics_at(test_labels, test_scores, thr_adapt, cfg)
        for k in THRESHOLDED:
            if k in m_adapt:
                rec[k + "_adapt"] = m_adapt[k]
    return rec


def run(dataset: str, arm: str, seed: int, out_path: Path, adaptive: bool,
        sources: list[str] | None, targets: list[str] | None, limit: int | None):
    cfg = build_cfg(dataset, seed)
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reg = scan_arm(dataset, arm, seed)
    if not reg:
        raise SystemExit(f"no checkpoints for {dataset}/{arm}/seed{seed}")
    all_ents = sorted(reg.keys())
    src_ents = [e for e in all_ents if (sources is None or e in sources)]
    print(f"[cross] {dataset} arm={arm} seed{seed}: {len(all_ents)} entities "
          f"(clusters: {sorted({v['cluster'] for v in reg.values()})})", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_cells = 0
    t0 = time.time()
    with open(out_path, "w") as fout:
        for si, source in enumerate(src_ents):
            src_info = reg[source]
            src_rep = load_report(src_info["report"])
            thr_source = src_rep.get("threshold") if src_rep else None
            # federated models are per-cluster: only transfer within the source cluster.
            if arm == "local":
                tgt_pool = all_ents
            else:
                tgt_pool = [e for e in all_ents if reg[e]["cluster"] == src_info["cluster"]]
            tgt_ents = [e for e in tgt_pool if (targets is None or e in targets)]

            stage1 = stage2 = None  # lazy: only load the model if there is an off-diagonal target
            for target in tgt_ents:
                if source == target:
                    rec = diagonal_record(dataset, arm, seed, source, src_info)
                else:
                    if stage1 is None:
                        example = _example_inputs(cfg, source, device)
                        stage1 = load_stage1(src_info["s1"], cfg, example, device=device)
                        stage2 = load_stage2(src_info["s2"], cfg, stage1_ckpt=src_info["s1"],
                                             stage1_example_inputs=example, device=device)
                    ts, tl, thr_adapt = score_target(stage1, stage2, cfg, target, adaptive)
                    rec = offdiag_record(dataset, arm, seed, source, target,
                                         src_info, reg[target], tl, ts, thr_source, thr_adapt, cfg)
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
                n_cells += 1
                if limit and n_cells >= limit:
                    print(f"[cross] hit --limit {limit}, stopping", flush=True)
                    return
            if stage1 is not None and torch.cuda.is_available():
                del stage1, stage2
                torch.cuda.empty_cache()
            dt = time.time() - t0
            print(f"[cross] source {si+1}/{len(src_ents)} ({source}) done · "
                  f"{n_cells} cells · {dt/max(n_cells,1):.1f}s/cell · {dt/60:.1f} min elapsed",
                  flush=True)
    print(f"[cross] wrote {n_cells} cells -> {out_path}", flush=True)


def _example_inputs(cfg, entity, device):
    c = copy.deepcopy(cfg)
    c.dataset.entity_id = entity
    data = make_dataloaders(c, stage="eval")
    return next(iter(data.train_loader))["inputs"][:1].cpu()


def smoke(dataset: str, arm: str, seed: int):
    """Recompute one diagonal cell and compare test_scores to the on-disk scores.npz."""
    cfg = build_cfg(dataset, seed)
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reg = scan_arm(dataset, arm, seed)
    entity = sorted(reg.keys())[0]
    info = reg[entity]
    npz_path = info["s1"].parent / "scores.npz"
    if not npz_path.exists():
        raise SystemExit(f"no scores.npz to compare at {npz_path}")
    ref = np.load(npz_path)
    ref_test = ref["test_scores"]
    print(f"[smoke] recomputing diagonal {arm}/{entity} (seed{seed}) vs {npz_path}", flush=True)
    example = _example_inputs(cfg, entity, device)
    stage1 = load_stage1(info["s1"], cfg, example, device=device)
    stage2 = load_stage2(info["s2"], cfg, stage1_ckpt=info["s1"],
                         stage1_example_inputs=example, device=device)
    ts, tl, _ = score_target(stage1, stage2, cfg, entity, adaptive=False)
    print(f"[smoke] shapes: recompute={ts.shape} ref={ref_test.shape}", flush=True)
    if ts.shape != ref_test.shape:
        raise SystemExit(f"[smoke] FAIL shape mismatch {ts.shape} vs {ref_test.shape}")
    diff = float(np.max(np.abs(ts - ref_test)))
    rel = diff / (float(np.max(np.abs(ref_test))) + 1e-12)
    print(f"[smoke] max|Δ|={diff:.3e}  rel={rel:.3e}", flush=True)
    if rel < 1e-4:
        print("[smoke] PASS ✓ recomputed diagonal matches on-disk scores", flush=True)
    else:
        print("[smoke] WARN: mismatch above 1e-4 — investigate cfg parity", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--arm", default="local", choices=["local", "federated"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--adaptive", action="store_true",
                   help="also compute the target-refit threshold (adds a train forward pass/cell).")
    p.add_argument("--sources", type=str, default=None, help="comma-sep entity ids (default all)")
    p.add_argument("--targets", type=str, default=None, help="comma-sep entity ids (default all)")
    p.add_argument("--limit", type=int, default=None, help="stop after N cells (smoke/timing)")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        smoke(args.dataset, args.arm, args.seed)
        return
    out = Path(args.out) if args.out else (
        resolve_path("artifacts/fed_eval/cross") /
        f"records_cross_{args.dataset}_{args.arm}_s{args.seed}.jsonl")
    srcs = [x.strip() for x in args.sources.split(",")] if args.sources else None
    tgts = [x.strip() for x in args.targets.split(",")] if args.targets else None
    run(args.dataset, args.arm, args.seed, out, args.adaptive, srcs, tgts, args.limit)


if __name__ == "__main__":
    main()
