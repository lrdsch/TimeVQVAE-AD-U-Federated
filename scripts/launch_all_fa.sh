#!/usr/bin/env bash
# Launch ALL Federated Analytics experiments, one screen each (wsd then toy),
# staggered ~4s to avoid a CUDA-context allocation spike. GPU1 only.
#
#!FA! ─── SHELVED 2026-07-27 — DO NOT RUN WITHOUT READING THIS ────────────────────────────────
#!FA! `scripts/mixture_eval.py:174` tokenizes EVERY client with client-0's encoder, but
#!FA!   ^ that line number is STALE (it predates the 2026-07-30 retraction banner). The real
#!FA!     site is the `load_stage1(s1_ckpts[have[0]], ...)` call inside `_load_pool` —
#!FA!     `scripts/mixture_eval.py:234` as of 2026-07-30. Grep for `have[0]`, not for a line.
#!FA! `federated_cb_only` federates ONLY the codebook — encoders stay local and diverge
#!FA! (measured cross-client token agreement 0.0000, codeword-occupancy JS 0.686-0.693 against a
#!FA! ln2=0.6931 ceiling, i.e. disjoint support). Any script that feeds a client's own prior with
#!FA! another client's tokens is scoring a transformer on symbols it never saw.
#!FA!   WITHDRAWN (unrecoverable without encoder federation): transfer, backoff, inputnorm,
#!FA!              onboard's onboard_mixture arm
#!FA!   CLEAN     (already pair per-client stage1 + prior): calibration, bnstats, mergevar
#!FA!   FIAT-TOKENIZER (internally valid, externally mis-specified — cite only with the
#!FA!              conditional caveat): robust, cbusage, ngram_ho, pca, rarity, dp (at --lam 0),
#!FA!              onboard's onboard_count, coldstart's fa_assisted
#!FA! Also: NO fa_* result was ever produced through the post-purge path, and the audit of
#!FA! 2026-07-15 retracted all four legs of the FA value proposition (interoperability, DP,
#!FA! Byzantine robustness, cold-start lead). `mixture_eval.py` already raises SystemExit at its
#!FA! own choke point; this launcher is the OTHER entry point and had no guard until 2026-07-30.
#!FA! Every scripts/fa_*.py now carries the same retraction banner at the top of the file, and
#!FA! each one refuses to write a 0-byte ledger, so a shelved run cannot leave behind an artifact
#!FA! that downstream reducers mistake for "not yet run".
#!FA! Full reasoning: documentation/RESEARCH_LEDGER.md (Group 4) and
#!FA! documentation/LAUNCH_RUNBOOK.md §5.2b.
#!FA! Reopening requires `federated_enc_fedavg` (the only arm with bit-identical encoders).
#!FA! Note on checkpoints: TVQ_CONVERGED_ROOT must point at artifacts/runs/<tag>/ckpt (a cohort
#!FA! tag from scripts/launch.sh). artifacts/converge60/ckpt no longer exists (archived
#!FA! 2026-07-29) and artifacts/_archive_20260729/ is read-only history, not a substitute.
if [[ "${FA_I_KNOW_ITS_SHELVED:-}" != "1" ]]; then
  grep '^#!FA!' "$0" | cut -c6- >&2
  echo "" >&2
  echo "REFUSING to launch 14 GPU-day-scale FA experiments whose suite is shelved." >&2
  echo "The suite has NEVER produced a single record, and its results were retracted before" >&2
  echo "they existed: running it now can only manufacture numbers that must not be cited." >&2
  echo "If you have read the above and still want it: FA_I_KNOW_ITS_SHELVED=1 $0" >&2
  exit 2
fi
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
