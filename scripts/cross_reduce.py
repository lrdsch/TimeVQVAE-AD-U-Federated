"""Reduce the cross-client grid into the transfer/comparison metric set.

Ingests, for ONE dataset + seed:
  * the recomputed cross grid (scripts/cross_eval.py output) for the per-client
    families `local` and `federated` — full M[source][target], diagonal + off;
  * the pooled/`centralized` diagonal from artifacts/fed_eval/<ds>/<cl>/seed*/centralized/*/report.json;
  * every Federated-Analytics family from artifacts/fed_eval/<fam>/records_<ds>.jsonl
    (flare / prism / halo / flare_via / mixture / fa_transfer / fa_onboard),
    each a diagonal-equivalent row on the common target axis.

Emits, on the primary metric (default vus_pr, threshold-free so comparable across
ALL families), per target client j:
  self, foreign_mean/oracle/worst (all + within-cluster), and the deltas
  τ (transfer gap), κ (collaboration ceiling), φ_fed (federation value),
  π (pooling premium), portability_gain (does federation make models more
  transferable?). Plus a family leaderboard and the within- vs across-cluster
  transfer split that validates the per-cluster thesis.

    python scripts/cross_reduce.py --dataset wsd_fed --seed 0
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FED = ROOT / "artifacts" / "fed_eval"
CROSS = FED / "cross"
FA_FAMILIES = ["flare", "prism", "halo", "flare_via", "mixture", "fa_transfer", "fa_onboard"]
STRUCTURAL = ["vus_pr", "auprc", "auroc", "pate_f1"]


def load_cross(dataset: str, arm: str, seed: int) -> list[dict]:
    f = CROSS / f"records_cross_{dataset}_{arm}_s{seed}.jsonl"
    if not f.exists():
        return []
    return [json.loads(l) for l in open(f) if l.strip()]


def load_centralized(dataset: str, seed: int) -> dict[str, dict]:
    """{entity: report} for the pooled model (per-cluster single-shared)."""
    out = {}
    for rep in (FED / dataset).glob(f"*/seed{seed}/centralized/*/report.json"):
        d = json.load(open(rep))
        e = d.get("entity_id") or rep.parent.name
        out[e] = d
    return out


def load_fa(dataset: str, seed: int) -> dict[str, dict[str, dict]]:
    """{family_arm: {entity: metrics}} across all FA record files (seed-filtered)."""
    fams: dict[str, dict[str, dict]] = defaultdict(dict)
    for fam in FA_FAMILIES:
        f = FED / fam / f"records_{dataset}.jsonl"
        if not f.exists():
            continue
        for line in open(f):
            r = json.loads(line)
            if int(r.get("_seed", 0)) != seed:
                continue
            arm = r.get("_arm", fam)
            ent = r.get("_entity")
            if ent is None:
                continue
            key = f"{fam}:{arm}"
            fams[key][ent] = {k: r[k] for k in STRUCTURAL if k in r}
    return fams


def build_matrix(records: list[dict], metric: str) -> tuple[dict, dict, dict]:
    """Return (M[(src,tgt)]=val, entity->cluster, diagonal[entity]=val)."""
    M, cluster, diag = {}, {}, {}
    for r in records:
        s, t = r["source"], r["target"]
        cluster[s] = r["cluster_src"]
        cluster[t] = r["cluster_tgt"]
        v = r.get(metric)
        if v is None:
            continue
        M[(s, t)] = float(v)
        if r["is_diagonal"]:
            diag[t] = float(v)
    return M, cluster, diag


def foreign_stats(M, cluster, entities, target, within: bool):
    vals = []
    for s in entities:
        if s == target:
            continue
        if within and cluster.get(s) != cluster.get(target):
            continue
        v = M.get((s, target))
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    return {"mean": float(np.mean(vals)), "oracle": float(np.max(vals)),
            "worst": float(np.min(vals)), "n": len(vals)}


def agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    m = float(np.mean(vals))
    sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    ci = 1.96 * sd / (len(vals) ** 0.5) if len(vals) > 1 else 0.0
    return {"mean": m, "median": float(np.median(vals)), "ci95": ci, "n": len(vals)}


def winrate(pairs):
    """fraction where a>b, over (a,b) with both present."""
    ok = [(a, b) for a, b in pairs if a is not None and b is not None]
    if not ok:
        return None
    return sum(1 for a, b in ok if a > b) / len(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--metric", default="vus_pr", choices=STRUCTURAL)
    args = ap.parse_args()
    ds, seed, metric = args.dataset, args.seed, args.metric

    loc = load_cross(ds, "local", seed)
    fed = load_cross(ds, "federated", seed)
    if not loc:
        raise SystemExit(f"no local cross records for {ds} seed{seed} yet")

    Ml, cluster, loc_diag = build_matrix(loc, metric)
    Mf, _, fed_diag = build_matrix(fed, metric)
    entities = sorted(cluster.keys())
    pooled = load_centralized(ds, seed)
    fa = load_fa(ds, seed)

    # ---- per-target derived table ----
    rows = []
    for j in entities:
        self_l = loc_diag.get(j)
        fa_all = foreign_stats(Ml, cluster, entities, j, within=False)
        fa_win = foreign_stats(Ml, cluster, entities, j, within=True)
        ffed = foreign_stats(Mf, cluster, entities, j, within=True)
        fed_self = fed_diag.get(j)
        pool = (pooled.get(j) or {}).get(metric)
        row = {
            "target": j, "cluster": cluster[j],
            "self_local": self_l,
            "foreign_mean_all": fa_all and fa_all["mean"],
            "foreign_oracle_all": fa_all and fa_all["oracle"],
            "foreign_worst_all": fa_all and fa_all["worst"],
            "foreign_mean_within": fa_win and fa_win["mean"],
            "foreign_oracle_within": fa_win and fa_win["oracle"],
            "self_fed": fed_self,
            "fed_foreign_mean_within": ffed and ffed["mean"],
            "pooled": pool,
        }
        # deltas
        row["tau_all"] = _sub(self_l, row["foreign_mean_all"])
        row["tau_within"] = _sub(self_l, row["foreign_mean_within"])
        row["kappa_all"] = _sub(row["foreign_oracle_all"], self_l)
        row["kappa_within"] = _sub(row["foreign_oracle_within"], self_l)
        row["phi_fed"] = _sub(fed_self, self_l)
        row["pi_pool"] = _sub(pool, self_l)
        pen_loc = _sub(self_l, row["foreign_mean_within"])
        pen_fed = _sub(fed_self, row["fed_foreign_mean_within"])
        row["portability_gain"] = _sub(pen_loc, pen_fed)
        rows.append(row)

    # ---- aggregates ----
    def col(k):
        return agg([r[k] for r in rows])

    summary = {
        "dataset": ds, "seed": seed, "metric": metric,
        "n_clients": len(entities),
        "clusters": sorted(set(cluster.values())),
        "aggregates": {k: col(k) for k in [
            "self_local", "foreign_mean_all", "foreign_oracle_all", "foreign_mean_within",
            "self_fed", "pooled", "tau_all", "tau_within", "kappa_all", "kappa_within",
            "phi_fed", "pi_pool", "portability_gain"]},
        "winrates": {
            "oracle_all>self (collaboration exists)":
                winrate([(r["foreign_oracle_all"], r["self_local"]) for r in rows]),
            "oracle_within>self":
                winrate([(r["foreign_oracle_within"], r["self_local"]) for r in rows]),
            "fed>self (federation helps)":
                winrate([(r["self_fed"], r["self_local"]) for r in rows]),
            "pooled>self (pooling helps)":
                winrate([(r["pooled"], r["self_local"]) for r in rows]),
            "portability_gain>0 (fed more transferable)":
                winrate([(r["portability_gain"], 0.0) for r in rows]),
        },
    }

    # ---- within vs across-cluster transfer (validates per-cluster thesis) ----
    within, across = [], []
    for (s, t), v in Ml.items():
        if s == t:
            continue
        (within if cluster.get(s) == cluster.get(t) else across).append(v)
    summary["cluster_structure"] = {
        "within_cluster_transfer_mean": float(np.mean(within)) if within else None,
        "across_cluster_transfer_mean": float(np.mean(across)) if across else None,
        "block_gap": (float(np.mean(within)) - float(np.mean(across))) if within and across else None,
        "n_within": len(within), "n_across": len(across),
    }

    # ---- family leaderboard (diagonal / own-target score, threshold-free) ----
    board = {}
    board["local"] = agg([loc_diag.get(e) for e in entities])
    board["federated"] = agg([fed_diag.get(e) for e in entities])
    board["centralized_pooled"] = agg([(pooled.get(e) or {}).get(metric) for e in entities])
    for famkey, byent in fa.items():
        board[famkey] = agg([byent.get(e, {}).get(metric) for e in entities])
    summary["leaderboard"] = dict(sorted(
        board.items(), key=lambda kv: (kv[1]["median"] if kv[1] else -1), reverse=True))

    # ---- write + print ----
    outdir = CROSS / "analysis"
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / f"analysis_{ds}_s{seed}_{metric}.json", "w") as f:
        json.dump({"summary": summary, "per_target": rows}, f, indent=1)
    # matrix csv
    with open(outdir / f"matrix_{ds}_local_{metric}_s{seed}.csv", "w") as f:
        f.write("source\\target," + ",".join(entities) + "\n")
        for s in entities:
            f.write(s + "," + ",".join(
                f"{Ml.get((s,t),''):.4f}" if (s, t) in Ml else "" for t in entities) + "\n")

    _print(summary)


def _sub(a, b):
    return (a - b) if (a is not None and b is not None) else None


def _fmt(d):
    if not d:
        return "  n/a"
    return f"mean={d['mean']:+.4f} median={d['median']:+.4f} ±{d['ci95']:.4f} (n={d['n']})"


def _print(s):
    print(f"\n{'='*70}\n  CROSS-CLIENT TRANSFER — {s['dataset']} seed{s['seed']} · metric={s['metric']}")
    print(f"  {s['n_clients']} clients · clusters {s['clusters']}\n{'='*70}")
    print("\n-- Reference levels (per target) --")
    for k in ["self_local", "foreign_mean_all", "foreign_oracle_all", "foreign_mean_within",
              "self_fed", "pooled"]:
        print(f"  {k:22s} {_fmt(s['aggregates'][k])}")
    print("\n-- Comparison deltas (per target) --")
    for k in ["tau_all", "tau_within", "kappa_all", "kappa_within", "phi_fed", "pi_pool",
              "portability_gain"]:
        print(f"  {k:22s} {_fmt(s['aggregates'][k])}")
    print("\n-- Win-rates --")
    for k, v in s["winrates"].items():
        print(f"  {k:45s} {('%.1f%%'%(100*v)) if v is not None else 'n/a'}")
    cs = s["cluster_structure"]
    print("\n-- Cluster structure (local transfer) --")
    print(f"  within-cluster mean : {cs['within_cluster_transfer_mean']}")
    print(f"  across-cluster mean : {cs['across_cluster_transfer_mean']}")
    print(f"  block gap (within-across): {cs['block_gap']}  (n_within={cs['n_within']} n_across={cs['n_across']})")
    print("\n-- Family leaderboard (diagonal, threshold-free) --")
    for fam, d in s["leaderboard"].items():
        print(f"  {fam:26s} {_fmt(d)}")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
