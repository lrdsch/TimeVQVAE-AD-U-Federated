"""Aggregate converged arms + the three new experiments onto the true axis
(VUS-PR / AUPRC / PATE-F1 / AUROC). Micro-mean over (cluster, seed, entity) with n,
plus a per-cluster VUS-PR breakdown for the headline arms."""
import json, glob, os, sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _fedpaths import conv_root                    # noqa: E402

REPO = "/home/leonardo/PhD/TimeVQVAE-AD-U-Federated"
MET = ["auroc", "auprc", "vus_pr", "pate_f1"]


def add(store, arm, cluster, rep):
    for m in MET:
        v = rep.get(m)
        if isinstance(v, (int, float)) and v == v:
            store[arm][m].append(v)
            store[arm]["_by_cluster"].setdefault(cluster, defaultdict(list))[m].append(v)


def new_store():
    return defaultdict(lambda: defaultdict(list, {"_by_cluster": {}}))


def collect(ds):
    S = new_store()
    # converged arms
    for rep in glob.glob(f"{conv_root(ds, f'{REPO}/artifacts/fed_eval')}/*/seed*/*/*/report.json"):
        p = rep.split("/"); arm = p[-3]; cluster = p[-5]
        try: r = json.load(open(rep))
        except Exception: continue
        add(S, arm, cluster, r)
    # protoprior (FedProto at prior level)
    for rep in glob.glob(f"{REPO}/artifacts/fed_eval/protoprior/{ds}/*/seed*/federated_protoprior/*/report.json"):
        p = rep.split("/"); cluster = p[-5]
        try: r = json.load(open(rep))
        except Exception: continue
        add(S, "protoprior", cluster, r)
    # mixture + feddf records
    for kind in ("mixture", "feddf"):
        f = f"{REPO}/artifacts/fed_eval/{kind}/records_{ds}.jsonl"
        try: lines = open(f).read().splitlines()
        except FileNotFoundError: continue
        for ln in lines:
            r = json.loads(ln); add(S, r["_arm"], r.get("_cluster", "?"), r)
    # encoder-federation trio (scripts/run_enc_algo_sweep.sh writes OUTSIDE conv_root, so the
    # loop above never sees it). The arm directory carries the swept knobs, so a mu/lambda
    # point shows up as its own row (federated_enc_fedprox_mu0.1) rather than being pooled
    # with every other point of the same arm — which is what you want in a sweep table.
    for rep in glob.glob(f"{REPO}/artifacts/encalgo_sweep/ckpt/{ds}/*/seed*/*/*/report.json"):
        p = rep.split("/"); arm = p[-3]; cluster = p[-5]
        try: r = json.load(open(rep))
        except Exception: continue
        add(S, arm, cluster, r)
    # fedsgd per tau
    for rep in glob.glob(f"{REPO}/artifacts/fed_eval/fedsgd/{ds}/tau*/*/seed*/federated_fedsgd/*/report.json"):
        p = rep.split("/"); tau = p[-6]; cluster = p[-4]
        try: r = json.load(open(rep))
        except Exception: continue
        add(S, f"fedsgd_{tau}", cluster, r)
    return S


def main():
    for ds in ["toy_fed_uni", "wsd_fed"]:
        S = collect(ds)
        print(f"\n================= {ds} — micro-mean over (cluster,seed,entity) =================")
        order = ["centralized", "local", "federated_cb_only",
                 # encoder-federation trio + its null control, kept adjacent to cb_only:
                 # cb_only is their reference (same codebook merge, no encoder federation)
                 # and commoninit is the "same init, no federation" floor.
                 "federated_enc_commoninit", "federated_enc_fedavg",
                 "federated_enc_fedprox", "federated_enc_fedproto",
                 "mixture_best", "mixture_mixture", "mixture_mean_nll", "feddf", "protoprior",
                 "fedsgd_tau1", "fedsgd_tau2", "fedsgd_tau4", "fedsgd_tau8", "fedsgd_tau16",
                 "federated", "federated_fedavg_cb", "federated_shared"]
        arms = [a for a in order if a in S] + [a for a in S if a not in order]
        print(f'{"arm":<22}{"auroc":>8}{"auprc":>8}{"vus_pr":>8}{"pate_f1":>8}    n  clusters')
        for a in arms:
            n = len(S[a].get("vus_pr", []))
            def mm(m):
                v = S[a].get(m, []); return f"{np.mean(v):.3f}" if v else "  -  "
            ncl = len(S[a]["_by_cluster"])
            print(f'{a:<22}{mm("auroc"):>8}{mm("auprc"):>8}{mm("vus_pr"):>8}{mm("pate_f1"):>8}  {n:>4}  {ncl}')

        # per-cluster VUS-PR for headline arms (matched comparison)
        head = [a for a in ["centralized", "local", "federated_cb_only",
                            "mixture_best", "mixture_mixture", "feddf"] if a in S]
        clusters = sorted({c for a in head for c in S[a]["_by_cluster"]})
        print(f"\n  per-cluster VUS-PR:")
        print("    " + "cluster".ljust(14) + "".join(a.replace("mixture_", "mix.").replace("federated_", "")[:10].rjust(11) for a in head))
        for c in clusters:
            row = "    " + c.ljust(14)
            for a in head:
                v = S[a]["_by_cluster"].get(c, {}).get("vus_pr", [])
                row += (f"{np.mean(v):.3f}" if v else "  -  ").rjust(11)
            print(row)


if __name__ == "__main__":
    main()
