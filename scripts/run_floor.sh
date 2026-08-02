#!/usr/bin/env bash
# FLOOR — THE baseline pipeline. Paper scope, frozen 2026-07-30.
#
# What this runs is the REPORTED baseline and nothing else. The runner used to carry
# every batch the design document describes; the triage of 2026-07-30 cut the ones that
# cannot earn their space in a workshop paper. The cut items are listed at the BOTTOM of
# this file with the reason, so nobody has to re-derive the decision. Selecting a cut
# mode by hand still works — floor_eval prints an OUT OF PAPER SCOPE banner — but it is
# an ablation, not part of the table.
#
#   bash scripts/run_floor.sh smoke        # ~10 min: 1 cluster/dataset, every kept cell
#   bash scripts/run_floor.sh wsd          # the wsd_fed matrix        (~1-2 h)
#   bash scripts/run_floor.sh ucr          # ucr_split_w2p, 180 series (~3-5 h)
#   bash scripts/run_floor.sh consolidate  # envelope + table + stats  (minutes)
#   bash scripts/run_floor.sh all          # wsd -> ucr -> consolidate
#
#   FLOOR_JOBS=12 bash scripts/run_floor.sh all       # free CPU
#
# RESUMABLE. Every batch skips (arm, entity) pairs already on disk at the current
# schema, so re-running after an interruption costs only what is missing. No batch
# passes --force: batches A and F used to, which meant an interrupted headline or UCR
# run restarted from zero — the two longest batches were the two that could not resume.
#
# Batches run SEQUENTIALLY on purpose. Rule 1 is one process per dataset (build_cfg
# resolves the config once at start-up and wsd_fed carries metrics_tolerance=14 against
# 64 everywhere else), and floor_eval already caps itself at 4 workers and nice 19
# whenever federated_eval.py is alive.
#
# Cost note: `--suite on` adds the nested metrics_core.evaluate_scores block and costs
# ~10x the flat block (~110 s/client; PATE is ~5.5 s per call and the suite calls it ~8
# times). It is ON for the two arms that go into the paper table and OFF everywhere
# else, which still carries the COMPLETE flat set: detection + event + paper top-1/3/5.
set -u
set -o pipefail        # so the envelope gate below sees the real exit code, not tee's
cd "$(dirname "$0")/.."
PY=${PY:-/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10}
J=${FLOOR_JOBS:-8}
LOG=logs/floor
OUT=${FLOOR_OUT:-artifacts/floor}
DERIVED="$OUT/_derived"
mkdir -p "$LOG"
# The target is REQUIRED and VALIDATED. It used to default to `all`, so a bare
# `bash scripts/run_floor.sh` silently launched the ~4-6 h target — the most expensive one
# was the one you got by forgetting the argument, while the runbook says "always smoke
# first". And an unrecognised word reached no branch at all: it ran the gate, printed
# "done ->" and exited 0, so a typo (`usr` for `ucr`) was indistinguishable from a completed
# run — worst of all under nohup, where you only ever see the tail of the log.
WHAT=${1:-}
case "$WHAT" in
  smoke|wsd|ucr|consolidate|all) ;;
  *) echo "usage: $0 {smoke|wsd|ucr|consolidate|all}" >&2
     echo "  smoke        ~10 min, writes to artifacts/floor_smoke/ — run this first" >&2
     echo "  wsd          the wsd_fed matrix        (~1 h)" >&2
     echo "  ucr          ucr_split_w2p, 180 series (~3-5 h)" >&2
     echo "  consolidate  envelope + table + stats  (minutes)" >&2
     echo "  all          wsd -> ucr -> consolidate" >&2
     exit 2;;
esac

run() {  # run <tag> <args...>
  local tag=$1; shift
  echo "=== [$(date +%H:%M:%S)] $tag"
  $PY scripts/floor_eval.py --jobs "$J" --out-dir "$OUT" "$@" 2>&1 \
    | tee -a "$LOG/$tag.log" | tail -3
}

# ─────────────────────────────────────────────────────────────────────────────
#  0 — the gate. Nothing downstream is trustworthy if these do not pass.
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$WHAT" != "consolidate" ]]; then
  echo "=== [$(date +%H:%M:%S)] 0_selftest"
  $PY scripts/floor_heads.py > "$LOG/0_selftest.log" 2>&1 \
    || { echo "  floor_heads selftest FAILED -> $LOG/0_selftest.log"; exit 1; }
  $PY scripts/floor_eval.py --selftest >> "$LOG/0_selftest.log" 2>&1 \
    || { echo "  floor_eval selftest FAILED -> $LOG/0_selftest.log"; exit 1; }
  # 19 distinct invariants: 17 standalone in floor_heads + 2 contracts against the repo.
  # floor_eval --selftest RE-RUNS the floor_heads suite, so a naive grep -c over the combined
  # log counts the 17 twice and reports 36.
  printf "    %s invariants PASS (17 standalone + 2 repo contracts)\n" \
    "$(grep -c '\[PASS\]' "$LOG/0_selftest.log" | awk '{print $1 - 17}')"
fi

# ─────────────────────────────────────────────────────────────────────────────
#  SMOKE — one cluster per dataset, every kept (head x mode) cell, suite mostly off.
#  Exercises every code path the real batches use, in minutes instead of hours.
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$WHAT" == "smoke" ]]; then
  SM=${FLOOR_SMOKE_OUT:-artifacts/floor_smoke}
  mkdir -p "$SM"
  sm() { local tag=$1; shift; echo "=== [$(date +%H:%M:%S)] smoke_$tag"
         $PY scripts/floor_eval.py --jobs "$J" --out-dir "$SM" "$@" 2>&1 \
           | tee -a "$LOG/smoke_$tag.log" | tail -4; }

  # wsd_fed c0 = 5 clients, the smallest real cluster with more than one client.
  sm wsd_heads --dataset wsd_fed --clusters c0 \
      --heads ma_c,ma_causal,diff1,random,ar,pca,gauss --modes local --suite off
  sm wsd_suite --dataset wsd_fed --clusters c0 --heads ma_c --modes local --suite on
  sm wsd_noimp --dataset wsd_fed --clusters c0 --heads ma_c --modes local \
      --impulse off --suite off
  sm wsd_ksweep --dataset wsd_fed --clusters c0 --heads ma_c --modes local --k 3 --suite off
  # A per-series-window dataset. THREE clusters with three DISTINCT windows on purpose: with
  # one cluster there is one window, and the case that actually breaks — one model wearing a
  # different `__w<W>` arm name per series, so `_arm` grouping shatters it into 5-entity
  # fragments — is never exercised. This is the regression test for the model/arm split.
  W2P=$($PY -c "
import json
m = json.load(open('data/raw/ucr_split_w2p/metadata.json'))
W = {c: int(w) for c, w in m['windows'].items()}
picks = []
for c in sorted(W, key=lambda c: (W[c], c)):
    if W[c] not in [W[p] for p in picks]:
        picks.append(c)
    if len(picks) == 3:
        break
print(','.join(picks))")
  echo "    w2p smoke clusters: $W2P"
  sm w2p_heads --dataset ucr_split_w2p --clusters "$W2P" \
      --heads ma_c,ma_causal,ar,pca,random --modes local --suite off

  echo "=== [$(date +%H:%M:%S)] smoke consolidate"
  for ds in wsd_fed ucr_split_w2p; do
    $PY scripts/floor_envelope.py --dataset "$ds" --records-dir "$SM" \
        --out-dir "$SM/_derived" 2>&1 | tail -7
    $PY scripts/floor_table.py --dataset "$ds" \
        --records-dir "$SM,$SM/_derived" --out-dir "$SM" 2>&1 | tail -22
    # Mirror the REAL consolidate: two metrics, and the tolerance looked up from metadata.
    # Smoke used to run vus_pr only, so the paper_top1_acc_at_<tol> path and the TOL lookup
    # were never exercised — and defect 8 (the wrong reference arm on w2p) lived in exactly
    # the floor_stats path the smoke was skipping.
    TOL=$($PY -c "
import json
m = json.load(open('data/raw/$ds/metadata.json'))
print(m.get('metrics_tolerance') or 64)")
    for metric in vus_pr "paper_top1_acc_at_$TOL"; do
      $PY scripts/floor_stats.py --dataset "$ds" --metric "$metric" --csv-dir "$SM" 2>&1 \
          | grep -E "^dataset=|^\[stats\]|^floor |^SKIPPED|m80 ="
    done
  done
  echo "=== [$(date +%H:%M:%S)] smoke done -> $SM/"
  exit 0
fi

# ─────────────────────────────────────────────────────────────────────────────
#  wsd_fed — the paper's primary benchmark. 4 clusters / 31 clients.
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$WHAT" == "all" || "$WHAT" == "wsd" ]]; then
  mkdir -p "$OUT"
  # A — the headline row of federated_method.tex:452. Full metric suite.
  run A_headline --dataset wsd_fed --heads ma_c --modes local --k 10 --impulse on --suite on

  # B — impulse OFF, the first-class secondary result (FLOOR_BASELINE §0.6). Full suite
  #     so it is paired with A on every metric, INCLUDING the two non-results that must
  #     be reported alongside it: f1 at a fixed threshold moves the wrong way (-0.062,
  #     ns) and paper_top1 does not move at all.
  run B_impulse_off --dataset wsd_fed --heads ma_c --modes local --k 10 --impulse off --suite on

  # C — sensitivity to k. All six points. Dropping any of them AFTER seeing that k=10
  #     is the argmax would turn the anti-tuning shield into the forking path it exists
  #     to prevent, and k=3 is the point verifiable against the selftest invariant
  #     ma_c(k=3) == 0.25*diff1.
  for k in 3 5 20 32 50 128; do
    run C_k$k --dataset wsd_fed --heads ma_c --modes local --k $k --impulse on --suite off
  done

  # D — the rest of the head set, per-client fit. ma_causal / ar / pca join ma_c in the
  #     ENVELOPE (scripts/floor_envelope.py); diff1 and random are the band ends and the
  #     inferior anchor, reported in the text rather than as table rows; gauss is the
  #     ~8k-parameter end of the band (0 -> 33 -> 1024 -> 8256 fitted parameters across the
  #     head set is the span the band statement is about).
  run D_heads --dataset wsd_fed --heads ma_causal,diff1,random,ar,pca,gauss \
      --modes local --impulse on --suite off

  # NOTE: there is no batch E. The whole federation axis (central / fed_exact / fed_fedavg /
  # fed_fedavg_uniform / fed_scaleonly / fed_oneclient / fed_naive / fed_naive_aligned /
  # fed_localgd, and the --witness gate) was cut on 2026-07-30. See the bottom of this file.
fi

# ─────────────────────────────────────────────────────────────────────────────
#  ucr_split_w2p — the second dataset, at the window the UCR literature uses
#  (W = 2*period, per series, read from metadata: do NOT pass --window).
#
#  ALL 180 series, not a subset. The historical 82-cluster list was privileged only
#  because the deep runners used it; no deep run exists under a cohort fingerprint, so
#  that list confers nothing and subsetting would be an unexplained restriction.
#  (For the record: 61 of those 82 survive into this build, not the 76 the old runbook
#  implied — that figure was about the ucr_split 226-series universe, a different set.)
#
#  ucr_split at W=128 is deliberately NOT run: 128 is the wrong window on all 250 UCR
#  series, so those rows are unpairable with any deep arm, present or future.
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$WHAT" == "all" || "$WHAT" == "ucr" ]]; then
  mkdir -p "$OUT"
  # The four decision heads — so the envelope exists here too, not only on wsd — plus
  # the random anchor. gauss and diff1 are omitted: neither enters the envelope, and
  # gauss is the O(W^3) head on a build whose W reaches 3028.
  run F_w2p_heads --dataset ucr_split_w2p --heads ma_c,ma_causal,ar,pca,random \
      --modes local --impulse on --suite off
fi

# ─────────────────────────────────────────────────────────────────────────────
#  CONSOLIDATE — envelope, then table, then matched statistics.
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$WHAT" == "all" || "$WHAT" == "consolidate" ]]; then
  for ds in wsd_fed ucr_split_w2p; do
    [[ -f "$OUT/records_$ds.jsonl" ]] || { echo "=== skip $ds (no records yet)"; continue; }
    echo "=== [$(date +%H:%M:%S)] envelope $ds"
    # A GATE, not a step. floor_envelope refuses to invent a number when a decision head is
    # missing, the entity sets are ragged, two arms are ambiguous or the geometries disagree.
    # Without this guard the chain walked past the refusal and produced a complete-LOOKING
    # CSV and stats table with no envelope row in them — the upper bar silently absent.
    if ! $PY scripts/floor_envelope.py --dataset "$ds" --records-dir "$OUT" \
        --out-dir "$DERIVED" 2>&1 | tee -a "$LOG/envelope_$ds.log" | tail -7; then
      echo "  !! envelope REFUSED for $ds -> $LOG/envelope_$ds.log — NOT consolidating"
      continue
    fi
    echo "=== [$(date +%H:%M:%S)] table $ds"
    $PY scripts/floor_table.py --dataset "$ds" --records-dir "$OUT,$DERIVED" \
        --out-dir "$OUT" 2>&1 | tee -a "$LOG/table_$ds.log" | tail -32
    # the top-K key carries the tolerance, which differs per dataset (14 on wsd, 64 on ucr)
    TOL=$($PY -c "
import json
m = json.load(open('data/raw/$ds/metadata.json'))
print(m.get('metrics_tolerance') or 64)")
    for metric in vus_pr "paper_top1_acc_at_$TOL"; do
      echo "--- stats $ds / $metric  (reference: the pre-registered head)"
      $PY scripts/floor_stats.py --dataset "$ds" --metric "$metric" --csv-dir "$OUT" \
          2>&1 | tee -a "$LOG/stats_$ds.log" | tail -28
    done

  done
fi

echo "=== [$(date +%H:%M:%S)] done -> $OUT/"

# ═════════════════════════════════════════════════════════════════════════════
#  SCOPE, in one line: SIMPLE BASELINES ONLY — every head, fitted per client
#  (mode=local), on two datasets. No federation axis at all.
#
#  Why the federation axis went (cut 2026-07-30 after a 21-agent audit of every model):
#  it was not a baseline, it was a mechanistic study on a convex twin, and the audit found
#  that essentially all of it is currently indefensible.
#
#    central == fed_exact         bit-identical on all 21 metrics for every (head, client):
#                                 one row counted twice, which is part of why the "18 paired
#                                 comparisons" headline is really about 9.
#    fed_scaleonly                Delta = 0 is FORCED by positive homogeneity of the impulse
#                                 term, the channel max and the quantile threshold. A theorem
#                                 sitting inside an empirical null family.
#    fed_oneclient                the treatment is applied once per CLUSTER but the p is
#                                 computed at n=31 ENTITIES — the same unit inflation the
#                                 project forbids on ucr_split. Broadcaster is ents_order[0],
#                                 i.e. arbitrary.
#    fed_localgd (tau=16, R=30)   TRUNCATED: rho(M-bar) = 0.9916, ~544 rounds to converge, and
#                                 at R=30 its ridge objective is WORSE than one-shot FedAvg
#                                 (146 vs 113). By the project's own train-to-convergence rule
#                                 it is not reportable.
#    fed_naive / _aligned         the two arms differ in the AGGREGATION OPERATOR (raw basis +
#                                 QR vs projector + eigh), not only in the gauge, so "identical
#                                 aggregation operator, only the gauge differs" is false. And
#                                 the gauge is ONE hardcoded draw (rng(1000+i), not in the arm
#                                 tag): over 40 draws the captured energy spans 0.226..0.906,
#                                 so the -0.33 effect size is unidentified.
#    fed_fedavg + excess obj.     the number is defensible only with two references that were
#                                 never printed: normalised by J(w*) the cluster ordering
#                                 flips, and the LOCAL arm's own excess on the same objective
#                                 is 4-6x LARGER — so "the objective degrades while the metric
#                                 does not move" reads backwards.
#    --witness                    a measuring instrument for fed_exact; with fed_exact gone it
#                                 measures nothing that is reported.
#
#  All of it is still implemented, self-tested and selectable by hand (floor_eval prints an
#  OUT OF PAPER SCOPE banner). It is follow-up work, not workshop material.
#
#  Also cut, for the reasons below:
#
#  fed_prox (mu sweep)   the mu=0 -> local and mu->inf -> global limits are already
#                        asserted at machine precision in the selftest (0.0e+00 and
#                        1.9e-13), and FLOOR_BASELINE §8.3 forbids the only inference a
#                        data sweep would buy ("do not use it to close the deep FedProx
#                        null").
#  central_capN          re-tests a ledger claim already marked INVERTED under the
#                        current architecture (centralized 0.585 < local 0.617): a
#                        better instrument for a retracted claim.
#  ucr_split at W=128    wrong window on all 250 series; unpairable with any deep arm.
#  the 5 toy federations no claim rests on them and their deep side is stale since the
#                        14->47 client rebuild. The one residual use — a heterogeneous
#                        fixture for the gauge control — is discharged by the selftest
#                        invariant pca(b).
#  ucr_ad / ucr_pool     one cluster of 248 entities each, so `local` vs `federated` has
#                        no meaning: pretraining corpora, not federations.
#  gauss on w2p          O(W^2) memory / O(W^3) solve with W up to 3028, and it never
#                        enters the envelope.
#  fed_sharedbasis[_avg] NOT IMPLEMENTED, despite FLOOR_BASELINE §2.2 advertising the
#  knn / fed_coreset     representation-vs-head 2x2 and the additivity demarcation test.
#  fed_ensemble          fed_ensemble would be the cleanest possible support for the
#  fed_dp / ema_shard    weight-space-vs-function-space claim and is the best follow-up
#                        candidate; fed_dp would open a privacy front the paper
#                        explicitly declines (tex L322-324); ema_shard is marked
#                        PLUMBING - NO CLAIM by the design itself.
# ═════════════════════════════════════════════════════════════════════════════
