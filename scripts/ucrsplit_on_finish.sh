#!/usr/bin/env bash
# ucrsplit_on_finish.sh -- attende la fine dell'orchestratore e produce il report di verifica.
#
# Staccato dalla sessione: se la sessione interattiva muore, il report viene prodotto comunque.
#   setsid nohup bash scripts/ucrsplit_on_finish.sh > logs/ucrsplit/_on_finish.log 2>&1 < /dev/null &
#
# Attende l'USCITA del processo, non il successo: se l'orchestratore muore o crasha, il report
# viene scritto ugualmente (e dira' quanti cluster mancano).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
PY="${PY:-/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10}"
PAT="${PAT:-run_ucrsplit.sh}"
OUT="artifacts/ucrsplit/FINAL_CHECK.txt"; OUTJ="artifacts/ucrsplit/FINAL_CHECK.json"

echo "[$(date -Iseconds)] in attesa della fine di '$PAT'..."
while pgrep -f "$PAT" > /dev/null 2>&1; do sleep 120; done
echo "[$(date -Iseconds)] orchestratore terminato -- genero il report"

{
  echo "# ucr_split -- verifica delle previsioni pre-registrate"
  echo "# generato $(date -Iseconds) da scripts/ucrsplit_on_finish.sh"
  echo "# commit $(git rev-parse HEAD)  (working tree: $(git status --porcelain | wc -l) file modificati)"
  echo
  echo "## conteggio job"
  printf '  json prodotti: %s / 492\n' "$(ls -1 artifacts/ucrsplit/*__*.json 2>/dev/null | wc -l)"
  echo "  per arm:"; ls -1 artifacts/ucrsplit/*__*.json 2>/dev/null | sed 's/.*__//;s/\.json//' | sort | uniq -c | sed 's/^/    /'
  echo
  echo "## coda dell'orchestratore"
  tail -25 logs/ucrsplit/_orchestrator.log 2>/dev/null | sed 's/^/  /'
  echo
  echo "## job falliti (rc != 0)"
  grep 'rc=' logs/ucrsplit/_orchestrator.log 2>/dev/null | grep -v 'rc=0' | sed 's/^/  /' || echo "  nessuno"
  echo
  "$PY" scripts/ucrsplit_predictions_check.py --json "$OUTJ"
  echo
  echo "## metriche secondarie (robustezza della conclusione)"
  for m in auprc pate_f1 affiliation_f1; do
    echo "### $m"
    "$PY" scripts/ucrsplit_predictions_check.py --metric "$m" 2>&1 | sed -n '/previsto vs osservato/,/^$/p' | sed 's/^/  /'
  done
} > "$OUT" 2>&1

echo "[$(date -Iseconds)] report in $OUT"
