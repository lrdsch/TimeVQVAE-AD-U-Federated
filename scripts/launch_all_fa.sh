#!/usr/bin/env bash
# Launch ALL Federated Analytics experiments, one screen each (wsd then toy),
# staggered ~4s to avoid a CUDA-context allocation spike. GPU1 only.
REPO=/home/leonardo/PhD/TimeVQVAE-AD-U-Federated
PY=/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10
cd "$REPO" || exit 1
mkdir -p logs

start() {  # name  <full inner command string>
  local name=$1; shift
  screen -dmS "fa_$name" bash -c "cd $REPO && export CUDA_VISIBLE_DEVICES=1 && { $*; } > logs/fa_$name.log 2>&1"
  echo "launched fa_$name"
  read -t 4 x < /dev/zero || true
}

# ---- light inference experiments (full scope) ----
start backoff     "$PY scripts/fa_backoff.py --dataset wsd_fed --clusters c0,c2,c3 --seeds 0,1,2 --lams 0.1,0.3,0.5,0.7 ; $PY scripts/fa_backoff.py --dataset toy_fed_uni --clusters all --seeds 0,1,2 --lams 0.1,0.3,0.5,0.7"
start ngram_ho    "$PY scripts/fa_ngram_ho.py --dataset wsd_fed --clusters c0,c2,c3 --seeds 0,1,2 --variants unigram,bigram,bigram_trigram ; $PY scripts/fa_ngram_ho.py --dataset toy_fed_uni --clusters all --seeds 0,1,2 --variants unigram,bigram,bigram_trigram"
start calibration "$PY scripts/fa_calibration.py --dataset wsd_fed --clusters c0,c2,c3 --seeds 0,1,2 --quantiles 0.99,0.995 ; $PY scripts/fa_calibration.py --dataset toy_fed_uni --clusters M1_rotary,M2_valve,M3_pump,M4_cardiac,M5_bearing,M6_drive --seeds 0,1,2 --quantiles 0.99,0.995"
start onboard     "$PY scripts/fa_onboard.py --dataset wsd_fed --clusters c0,c2,c3 --seeds 0,1,2 ; $PY scripts/fa_onboard.py --dataset toy_fed_uni --clusters all --seeds 0,1,2"
start dp          "$PY scripts/fa_dp.py --dataset wsd_fed --clusters c0,c2,c3 --seeds 0,1,2 --sigmas 0,0.5,1,2,4 ; $PY scripts/fa_dp.py --dataset toy_fed_uni --clusters M1_rotary,M2_valve,M3_pump,M4_cardiac,M5_bearing,M6_drive --seeds 0,1,2 --sigmas 0,0.5,1,2,4"
start robust      "$PY scripts/fa_robust.py --dataset wsd_fed --clusters c0,c2,c3 --seeds 0,1,2 ; $PY scripts/fa_robust.py --dataset toy_fed_uni --clusters all --seeds 0,1,2"
start transfer    "$PY scripts/fa_transfer.py --dataset wsd_fed --clusters c0,c2,c3 --seeds 0,1,2 ; $PY scripts/fa_transfer.py --dataset toy_fed_uni --clusters all --seeds 0,1,2"
start cbusage     "$PY scripts/fa_cbusage.py --dataset wsd_fed --clusters all --seeds 0,1,2 --variants local,pooled,usage_aware ; $PY scripts/fa_cbusage.py --dataset toy_fed_uni --clusters all --seeds 0,1,2 --variants local,pooled,usage_aware"

# ---- bounded-scope tokenizer-stat probes ----
start bnstats     "$PY scripts/fa_bnstats.py --dataset wsd_fed --clusters c0,c3 --seeds 0,1"
start inputnorm   "$PY scripts/fa_inputnorm.py --dataset wsd_fed --clusters c0,c3 --seeds 0,1"
start pca         "$PY scripts/fa_pca.py --dataset wsd_fed --clusters c0,c3 --seeds 0,1 --refit-iters 2 ; $PY scripts/fa_pca.py --dataset toy_fed_uni --clusters all --seeds 0,1,2 --refit-iters 2"

# ---- training-heavy experiments ----
start rarity      "$PY -u scripts/fa_rarity.py --dataset wsd_fed --clusters c0,c2,c3 --seeds 0,1,2 --s2-epochs 18 ; $PY -u scripts/fa_rarity.py --dataset toy_fed_uni --clusters all --seeds 0,1,2 --s2-epochs 18"
start coldstart   "$PY scripts/fa_coldstart.py --dataset wsd_fed --clusters c0,c2,c3 --seeds 0,1,2 --fracs 0.1,0.25,0.5 --lam 0.3 --s1-epochs 20 --s2-epochs 20 ; $PY scripts/fa_coldstart.py --dataset toy_fed_uni --clusters all --seeds 0,1,2 --fracs 0.1,0.25,0.5 --lam 0.3 --s1-epochs 20 --s2-epochs 20"
start mergevar    "$PY scripts/fa_mergevar.py --dataset wsd_fed --clusters c0,c3 --seeds 0,1 --s1-rounds 15 --local-epochs 1 --s2-epochs 15 --lloyd-iters 3 --batch 128"

echo "=== all 14 FA experiments launched ==="
