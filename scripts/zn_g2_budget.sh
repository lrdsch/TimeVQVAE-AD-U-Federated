#!/usr/bin/env bash
# I 6 SLOT FERMI DI g2 — 2026-08-06 03:35.
#
#   bash scripts/zn_g2_budget.sh centr   # GPU0: `centralized` a tetto alzato, 5 serie
#   bash scripts/zn_g2_budget.sh local   # GPU1: `local` su ucr_222 e ucr_229, poi zn_ot
#
# PERCHE'. `zn_es` strozzava su 2 slot di GPU5 di g4, dove per giunta convive con `zn_a2`:
# 5 slot richiesti su una sola Ada. g2 aveva spazio.
#
# 🔴 ERRORE DA NON RIFARE — 2026-08-06 03:52. Al primo tentativo ho letto
# `nvidia-smi --query-compute-apps` come «3 job per GPU, sono nostri, il tetto e' 6-7, quindi
# ci sono 6 slot liberi» e ho lanciato 5 job. **Su g2 c'e' ssanchez**: 2 processi su GPU0 e 1
# su GPU1 (piu' il nostro mps-server, che pure compare nella lista). Di NOSTRO su g2 girava
# UN job solo. Avevo contato i suoi come nostri e gli sono finito sopra.
#
# Il conteggio dei job per GPU **non dice di chi sono**. Va sempre incrociato con l'utente:
#     nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader
#     ... e per ogni pid:  ps -o user= -p <pid>
# e il `nvidia-cuda-mps-server` va scartato, non e' un job.
#
# ⚠️ REGOLA g2 (definitiva, 2026-08-06). MAI entrambe le schede se qualcun altro lavora:
#     altri su 0 GPU -> entrambe · altri su 1 -> l'altra, una sola · altri su 2 -> UNA sola,
#     quella dove diamo meno fastidio, senza impedirgli il lavoro.
# Non e' una nota: la calcola `scripts/_g2_gpu.sh` dallo stato reale, e questo script la usa
# come default di LAUNCH_GPUS. Su g4 la regola resta piu' dura: mai una scheda con altri.
#
# ⚠️ Il vincolo vero su g2 NON e' la GPU, e' la CPU: 16 core, e ssanchez da solo ne usa il
# 491% contro il nostro 129%. Riempire "la nostra" scheda a fondo gli toglie i core lo
# stesso. Restare leggeri anche in SLOT, non solo in schede.
#
# QUALI CELLE, E PERCHE' PROPRIO QUELLE. La catena viva su GPU5 (`zn_budget.sh es`) esegue in
# sequenza: PRIMA tutte e 6 le `local`, POI le `centralized`. Quindi la coda con piu' slack e'
# `centralized` per intero — non ci arriva prima di ~un giorno — piu' le due `local` di coda,
# `ucr_222` e `ucr_229`. Prendere dalla TESTA della sua coda sarebbe una collisione a
# distanza di minuti; dalla CODA e' a distanza di ore.
#
# Le tre difese restano tutte in piedi, in ordine: il manifesto (`cohorts/zn_ownership.json`,
# riassegnato PRIMA di lanciare, `zn_owner.py --check` verde a 197 celle), l'idempotenza di
# `launch.sh` (salta la cella se l'out-json c'e' gia'), e `zn_dupe_guard.py` che uccide il
# piu' giovane. Lo slack fa si' che la prima basti da sola.
#
# ⚠️ TETTO ALZATO, NON EARLY STOPPING SPENTO. Qui `EARLY_STOPPING` resta acceso: e' la
# variante `es`, che misura il modello CONVERGITO. La variante sovrallenata (`ot`) e' un'altra
# cosa e va lanciata a parte.
# ⚠️ NIENTE bf16 su g2: sono Turing sm_75, dove bf16 e' EMULATO e piu' lento di fp32.
#    `launch.sh` pinna gia' FEDVQ_AMP=fp16.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
PY=/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10

ROLE="${1:?centr | local}"
export S1_MAX_STEPS="${S1_MAX_STEPS:-25000}"
export S2_MAX_STEPS="${S2_MAX_STEPS:-100000}"
export SLOTS_PER_GPU="${SLOTS_PER_GPU:-3}"     # 3 IN PIU' dei 3 gia' presenti = 6, il tetto
export FEDVQ_AMP=fp16
ZN='--window-normalization zscore'
say() { echo "[$(date -u '+%F %T') UTC | $(TZ=Europe/Madrid date '+%H:%M') Madrid][g2-budget/$ROLE] $*"; }

"$PY" scripts/zn_owner.py --check >/dev/null || { say "!! partizione ROTTA — non parto"; exit 2; }

case "$ROLE" in
  centr)
    export LAUNCH_GPUS="${LAUNCH_GPUS:-$(bash "$REPO/scripts/_g2_gpu.sh")}"
    export LAUNCH_ONLY_CLUSTERS="$("$PY" scripts/zn_owner.py --owner 'zn_g2_budget.sh centr' --tag zn_es_c)"
    say "centralized · GPU $LAUNCH_GPUS · $SLOTS_PER_GPU slot · $LAUNCH_ONLY_CLUSTERS"
    # ⚠️ Il tag e' `zn_es`, non `zn_es_c`: `zn_es_c` esiste solo nel manifesto per poter
    # dichiarare che l'arm `centralized` ha un proprietario diverso da `local`. Su disco le
    # due meta' devono finire nello STESSO run-dir, o la tabella si spacca in due.
    bash scripts/launch.sh --cohort ucr2p_10 --tag zn_es --arms centralized --extra "$ZN"
    ;;
  local)
    export LAUNCH_GPUS="${LAUNCH_GPUS:-$(bash "$REPO/scripts/_g2_gpu.sh")}"
    export LAUNCH_ONLY_CLUSTERS="$("$PY" scripts/zn_owner.py --owner 'zn_g2_budget.sh local' --tag zn_es)"
    say "local · GPU $LAUNCH_GPUS · $SLOTS_PER_GPU slot · $LAUNCH_ONLY_CLUSTERS"
    bash scripts/launch.sh --cohort ucr2p_10 --tag zn_es --arms local --extra "$ZN"
    ;;
  *) echo "ruolo sconosciuto '$ROLE' (centr|local)" >&2; exit 2;;
esac
say "=== $ROLE finito (rc=$?) ==="
