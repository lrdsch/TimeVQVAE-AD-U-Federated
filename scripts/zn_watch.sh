#!/usr/bin/env bash
# Aggiornamento compatto della campagna, pensato per girare ogni 15 minuti.
#
# Ordine deliberato: PRIMA il vincolo verso terzi (le GPU libere su g4), POI la scienza.
# Il 2026-08-05 il vincolo e' caduto per 3h30m e me ne sono accorto solo perche' l'utente
# ha chiesto: un controllo che non e' nel ciclo di monitoraggio non e' un controllo.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
PY=/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10
G4=${G4:-leonardo@g4.etsisi.upm.es}

echo "═══ $(TZ=Europe/Madrid date '+%H:%M Madrid') ═══"

# 1. vincolo g4 + occupazione per scheda
timeout 30 ssh -o BatchMode=yes "$G4" '
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
    awk -F", " "{if(\$2<=200){n++; l=l\" \"\$1}} END{printf \"  g4: %d GPU libere (%s )\n\", n, l}"
  for g in 0 1 2 3 4 5; do
    u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F", " -v x=$g "\$1==x{print \$2}")
    n=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F", " -v u="$u" "\$1==u" | wc -l)
    printf "%s:%s " "$g" "$n"
  done; echo " (job per scheda)"
  k=$(grep -c "la libero" evidence/ada_guard_$(date -u +%Y%m%d).log 2>/dev/null || echo 0)
  [ "$k" -gt 0 ] && echo "  ⚠ la guardia e intervenuta $k volte oggi"
  [ -f evidence/.ada_guard_off ] && echo "  (guardia Ada disattivata: g4 e tutta nostra)"
  true
' 2>/dev/null || echo "  g4 non raggiungibile"

# 2. A1 — il ramo che stiamo aspettando
echo "  ── A1 (zn_a1) ──"
for s in ucr_001 ucr_011 ucr_229 ucr_222; do
  out="artifacts/runs/zn_a1/ucr_split_w2p/${s}__federated_enc_fedavg.json"
  log="logs/runs/zn_a1/ucr_split_w2p__${s}__federated_enc_fedavg.log"
  if [ -f "$out" ]; then
    echo "    $s  ✅ CHIUSA"
  elif [ -f "$log" ]; then
    r=$(grep -c "round " "$log"); s2=$(grep -c "\[s2 " "$log")
    st=$(grep -o "CONVERGED[^;]*" "$log" | tail -1)
    printf "    %s  stadio %s · %s round%s\n" "$s" \
      "$([ "$s2" -gt 0 ] && echo 2 || echo 1)" "$r" \
      "$([ -n "$st" ] && echo "  [$st]" || echo "")"
  fi
done

# A2 — `--fed-enc-prior partial`, lanciato su Ada 0 e 5
if compgen -G "logs/runs/zn_a2/*.log" > /dev/null 2>&1; then
  echo "  ── A2 (zn_a2) ──"
  for log in logs/runs/zn_a2/ucr_split_w2p__*.log; do
    s=$(basename "$log" | sed 's/.*__\(ucr_[0-9]*\)__.*/\1/')
    out=$(ls artifacts/runs/zn_a2/ucr_split_w2p/${s}__*.json 2>/dev/null | head -1)
    if [ -n "$out" ]; then echo "    $s  ✅ CHIUSA"
    else printf "    %s  %s round\n" "$s" "$(grep -c 'round ' "$log")"; fi
  done
fi

# 3. conteggio celle + gate
# ⚠ zn_matrix conosce 120 celle e non `zn_a2`: il conteggio buono e' quello dell'audit,
# che legge il disegno da cohorts/zn_ownership.json (126 celle).
"$PY" scripts/zn_audit.py 2>/dev/null | grep -E "^DISEGNO" | sed 's/^/  /'
g=$(tail -1 logs/zn_a1_gate.log 2>/dev/null | cut -c1-95)
[ -n "$g" ] && echo "  gate: ${g#*gate-A1] }"
