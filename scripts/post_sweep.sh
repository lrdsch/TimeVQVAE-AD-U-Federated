#!/usr/bin/env bash
# post_sweep.sh — everything that has to happen AFTER run_converged_all.sh finishes.
#
# Waits for the sweep orchestrator to exit, then runs, in order:
#   1. verify_converged_run.py   — the two fixes, checked against the artefacts on disk
#   2. summarize_converged.py    — the matched local-vs-centralized table
#   3. an inventory of what actually got produced (and what is missing)
#
# Everything lands in ONE file so the whole outcome can be read in a single place:
#   logs/converged_all/_POSTSWEEP.md
#
# Detached on purpose: this outlives any interactive session, exactly like the sweep.
#   setsid nohup bash scripts/post_sweep.sh > /dev/null 2>&1 < /dev/null &
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PY="${PY:-/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10}"
LOGDIR="$REPO/logs/converged_all"
RESDIR="$REPO/artifacts/converged_all"
ORCH="$LOGDIR/_orchestrator.log"
OUT="$LOGDIR/_POSTSWEEP.md"
DONE_MARK="$LOGDIR/_POSTSWEEP.done"

rm -f "$DONE_MARK"

# ── 1. wait for the sweep ────────────────────────────────────────────────────
# Waits on the ACTUAL WORK, not on an orchestrator. Two reasons the orchestrator is
# the wrong thing to watch:
#   * jobs are `bash -c "train && quality"` children that keep running if the
#     orchestrator is killed — so "orchestrator gone" does not mean "work done";
#   * the queue may be served by more than one orchestrator (a second one can pick up
#     the remainder at a different concurrency), so one writing SWEEP COMPLETE says
#     nothing about the jobs still owned by another.
# Condition: no orchestrator AND no worker process of any kind still alive.
WORKERS='pipeline/federated_eval\.py|scripts/quality_pass\.py|run\.py -d ucr_ad'
sleep 120                                  # let the first jobs claim the GPU
while :; do
  n_work=$(pgrep -cf "$WORKERS" 2>/dev/null || echo 0)
  n_orch=$(pgrep -cf "bash scripts/run_converged_all.sh" 2>/dev/null || echo 0)
  [[ "$n_work" -eq 0 && "$n_orch" -eq 0 ]] && break
  sleep 60
done
REASON=$(grep -q "SWEEP COMPLETE" "$ORCH" 2>/dev/null \
         && echo "all workers finished (completion marker present)" \
         || echo "all workers finished, but NO completion marker — an orchestrator was killed or crashed")

{
  echo "# Post-sweep report"
  echo
  echo "- generated: $(date +'%F %T')"
  echo "- exit condition: **$REASON**"
  echo
  echo '## Job outcomes'
  echo '```'
  grep -E "START|DONE|FAILED" "$ORCH" 2>/dev/null || echo "(no orchestrator log)"
  echo '```'
} > "$OUT"

# ── 2. verify the two fixes ──────────────────────────────────────────────────
{
  echo
  echo '## CHECK — did the two fixes take effect in every arm?'
  echo '```'
} >> "$OUT"
$PY scripts/verify_converged_run.py --logdir "$LOGDIR" --resdir "$RESDIR" >> "$OUT" 2>&1
VERIFY_RC=$?
{ echo '```'; echo; echo "verify exit code: **$VERIFY_RC** (0 = pass)"; } >> "$OUT"

# ── 3. the results table ─────────────────────────────────────────────────────
{
  echo
  echo '## SCHEDULING — were the concurrency and GPU-affinity invariants held?'
  echo '```'
  if [[ -f "$LOGDIR/_WATCHDOG.log" ]]; then
    grep -c VIOLATION "$LOGDIR/_WATCHDOG.log" | sed 's/^/violation samples: /'
    echo "-- heartbeats + any violations --"
    grep -E "ok:|VIOLATION|TOTAL|watchdog up|orchestrator gone" "$LOGDIR/_WATCHDOG.log" | tail -40
  else
    echo "(no watchdog log — the invariants were NOT monitored for this run)"
  fi
  echo '```'
  echo
  echo '## RESULTS — matched local vs centralized'
  echo '```'
} >> "$OUT"
$PY scripts/summarize_converged.py --resdir "$RESDIR" >> "$OUT" 2>&1
{ echo '```'; } >> "$OUT"

# ── 4. inventory: what exists, what is missing ───────────────────────────────
{
  echo
  echo '## Inventory'
  echo '```'
  echo "result JSON:        $(ls -1 "$RESDIR"/*.json 2>/dev/null | wc -l) / 11 expected"
  echo "stage1 checkpoints: $(find "$REPO/artifacts/fed_eval" -name stage1.ckpt -newermt '2026-07-20 12:40' 2>/dev/null | wc -l)"
  echo "quality_stage1 out: $(find "$REPO/artifacts/fed_eval" -path '*quality_stage1/summary.json' -newermt '2026-07-20 12:40' 2>/dev/null | wc -l)"
  echo "quality_stage2 out: $(find "$REPO/artifacts/fed_eval" -path '*quality_stage2/*' -name '*.json' -newermt '2026-07-20 12:40' 2>/dev/null | wc -l)"
  echo
  echo "-- jobs whose log records a failure --"
  grep -lE "Traceback|FAILED|CUDA out of memory" "$LOGDIR"/*.log 2>/dev/null | xargs -r -n1 basename || echo "(none)"
  echo
  echo "-- UCR local (run.py, full 7-script chain) --"
  find "$REPO/artifacts/reports/ucr_ad" -name report.json -newermt '2026-07-20 12:40' 2>/dev/null | sed "s|$REPO/||" || true
  echo '```'
} >> "$OUT"

touch "$DONE_MARK"
