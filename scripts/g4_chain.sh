#!/usr/bin/env bash
# Catena completa di g4, RILANCIATA dopo il deadlock del ponte del 2026-08-01 15:54.
#
# Cosa era in volo quando il mount si e' piantato, e cosa succede al rilancio:
#  * ucr001_repg4  (2 celle, ~5 h di stage 2)  -> RIPARTE DA ZERO. Il bundle di resume esiste
#    ma sta in `_resume_tmp_<arm>_<pid-morto>/`, e federated_eval.py:1437 marca col PID
#    APPOSTA per impedire che un processo nuovo adotti il bundle di uno crashato (successe
#    davvero: un `local` si ritrovo' un bundle con rounds_done=300). Non lo promuovo a mano.
#  * ucr014_v1     (8/16 gia' su disco)        -> launch.sh salta le 8 fatte, rifa' le 8 perse.
#  * ucr001_floor100 (105/210 righe)           -> floor_eval.py:722 rilegge le righe esistenti
#    e riprende dalla 106 (chiave `(_arm,_entity)`), quindi NON si duplica. Sta in un altro
#    screen perche' e' puro CPU e non deve aspettare la GPU.
#
# L'ordine mette per primo il replicato: e' il numero che decide se le tabelle di g2 e g4 si
# possono mescolare, e finche' non c'e' ogni serie girata su g4 e' sospesa a quel verdetto.
set -u
cd /home/leonardo/PhD/TimeVQVAE-AD-U-Federated
PY=${PY:-/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10}

ablations() {
  local c="$1"
  bash scripts/launch.sh --cohort "$c" --tag "${c}_enc" \
    --arms federated_enc_fedavg,federated_enc_fedprox,federated_enc_fedproto,federated_enc_commoninit
  bash scripts/launch.sh --cohort "$c" --tag "${c}_cb128" \
    --arms federated_cb_only,federated_fedavg_cb_only --extra "--codebook-size 128"
  bash scripts/launch.sh --cohort "$c" --tag "${c}_proto_count" \
    --arms federated_enc_fedproto --extra "--fedproto-agg count"
}

export SLOTS_PER_GPU=4

echo "########## [$(date '+%F %T')] g4 — REPLICATO ucr001_repg4 ##########"
bash scripts/launch.sh --cohort ucr001 --arms federated_cb_only --tag ucr001_repg4

for s in ucr_014 ucr_043 ucr_086 ucr_170 ucr_083; do
  c="${s/_/}"
  echo "########## [$(date '+%F %T')] g4 — SERIE $s ##########"
  [ -f "cohorts/$c.json" ] || \
    "$PY" scripts/cohort.py new "$c" --datasets ucr_split,ucr_split_w2p --clusters "$s"
  bash scripts/launch.sh --cohort "$c" --arms paper --tag "${c}_v1"
  ablations "$c"
done
echo "=== [$(date '+%F %T')] g4: catena completa ==="
