#!/usr/bin/env bash
# Le 5 serie restanti, COMPLETE (paper + tutte le ablazioni), su g4.
# Gira SU g4, legge e scrive il repo di g2 attraverso il mount sshfs passive.
# LAUNCH_GPUS / NO_MPS li esporta run_on_g4.sh.
#
# SLOTS_PER_GPU=4 -> 8 job su 2 schede. Scelto per MISURARE, non per fede: con 2 job la GPU 4
# era gia' all'84% mentre la CPU di g4 stava all'88% IDLE, quindi qui il collo di bottiglia
# sono le GPU, non i core. Se a 8 job il tempo per round degrada in proporzione, il tetto e'
# vicino e servira' una terza scheda; se regge, si sale.
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
for s in ucr_014 ucr_043 ucr_086 ucr_170 ucr_083; do
  c="${s/_/}"
  echo "########## [$(date '+%F %T')] g4 — SERIE $s ##########"
  [ -f "cohorts/$c.json" ] || \
    "$PY" scripts/cohort.py new "$c" --datasets ucr_split,ucr_split_w2p --clusters "$s"
  bash scripts/launch.sh --cohort "$c" --arms paper --tag "${c}_v1"
  ablations "$c"
done
echo "=== [$(date '+%F %T')] g4: 5 serie complete ==="
