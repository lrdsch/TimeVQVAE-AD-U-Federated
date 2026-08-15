#!/usr/bin/env bash
# c50_status.sh — stato della campagna C50 a colpo d'occhio.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
Q="evidence/c50_queue_20260809.txt"

echo "== C50 · $(date -u '+%F %T') UTC =="
echo "in coda: $(wc -l < "$Q") celle su 150"
for t in c50_local c50_central c50_a2; do
  n=$(ls "artifacts/runs/$t/ucr_split_w2p/"*.json 2>/dev/null | wc -l)
  echo "  fatte $t: $n/50"
done

echo "-- celle in esecuzione --"
pgrep -af "federated_eval.py" | sed -nE 's/.*--cluster (ucr_[0-9]+).*runs\/(c50_[a-z0-9]+)\/.*/  \2 \1/p' | sort -u
pgrep -af "federated_eval.py" | grep -q "runs/c50_" || echo "  (nessuna)"

echo "-- corsie vive --"
alive=0
for p in logs/c50/gpu*_corsia*.pid; do
  [ -f "$p" ] || continue
  pid=$(cat "$p")
  kill -0 "$pid" 2>/dev/null && { echo "  $(basename "$p" .pid) (pid $pid)"; alive=$((alive+1)); }
done
[ "$alive" -eq 0 ] && echo "  (nessuna)"

echo "-- GPU --"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed 's/^/  /'

echo "-- centralized vs paper originale (±100) --"
/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10 scripts/authors_check.py 2>/dev/null | sed 's/^/  /'

echo "-- ultimi eventi --"
tail -n 4 logs/c50/worker_gpu*.out 2>/dev/null | sed 's/^/  /'
