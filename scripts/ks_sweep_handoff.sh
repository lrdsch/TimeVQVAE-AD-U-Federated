#!/usr/bin/env bash
# Passaggio automatico CPU -> GPU1 dello sweep ks, quando la catena zn_cbfa_tau64 finisce.
# Legale perche' ogni processo ri-punteggia `paper` INSIEME ai ks nuovi, nello stesso
# processo e sullo stesso device: nessun confronto appaiato attraversa due schede
# (PREREG_KS_SWEEP_2026-08-09 §A1). La coda e' ripartibile: salta cio' che e' gia' a disco.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
LOG="$REPO/evidence/ks_sweep_20260809.log"
say() { echo "[$(date -u '+%F %T') UTC][handoff] $*" | tee -a "$LOG"; }

say "attendo la fine di zn_cbfa_tau64 per liberare GPU1…"
for _ in $(seq 1 300); do          # max 5 ore
  pgrep -f "scripts/zn_cbfa_tau64.sh" >/dev/null 2>&1 || break
  sleep 60
done
if pgrep -f "scripts/zn_cbfa_tau64.sh" >/dev/null 2>&1; then
  say "!! la catena gira ancora dopo 5 ore — resto su CPU"; exit 0
fi
say "catena finita. Fermo i worker CPU dello sweep."
for p in $(pgrep -f "scripts/ks_sweep_queue.sh"; pgrep -f "xargs -a /tmp/tmp"; pgrep -f "scripts/rescore_ks.py"); do
  case "$(ps -o cmd= -p "$p" 2>/dev/null)" in *federated_eval*) continue;; esac
  kill "$p" 2>/dev/null
done
sleep 8
G="$(bash scripts/_g2_gpu.sh)" || { say "!! guardiano nega la GPU — resto su CPU"; exit 4; }
case " $G " in *" 0 "*) say "!! guardiano ha dato GPU0 — non riparto"; exit 4;; esac
say "riparto su GPU $G"
exec bash scripts/ks_sweep_queue.sh cuda 1
