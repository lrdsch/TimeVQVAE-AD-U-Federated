#!/usr/bin/env bash
# IL CONFRONTO EQUO SUL BUDGET — due punti di valutazione per cella, non uno.
#
#   bash scripts/zn_budget.sh es   # early stopping ACCESO, tetto alzato: il modello CONVERGITO
#   bash scripts/zn_budget.sh ot   # early stopping SPENTO, corsa al tetto: il SOVRALLENATO
#
# ── IL DIFETTO CHE CHIUDE ────────────────────────────────────────────────────────
# Le due famiglie di arm si fermavano per ragioni DIVERSE (misurato il 2026-08-06):
#
#   federati            pazienza 6 round, s1 e s2      tetto 300 round   mai toccato (max 119)
#   local/centralized   pazienza 2000 step  OPPURE     tetto 10 000 step SCATTA PRIMA su 6/10
#
# Cioe' gli arm federati si fermano a convergenza e le due baseline per esaurimento di
# budget. `local` colpisce il tetto su 001/011/014/170/222/229 e `centralized` su
# 001/011/014/222/229 (piu' lo stage 2 su 001 e 222). Su `ucr_011` — la serie da cui dipende
# TUTTO l'effetto A2, il leave-one-out lo azzera senza di lei — sono 4 client su 5.
# L'handicap e' a senso unico e GONFIA A2.
#
# ── PERCHE' DUE PUNTI E NON UNO ──────────────────────────────────────────────────
# `es` da' il modello che la regola di arresto sceglie: e' il confronto onesto, «A2 batte
# `local` alla convergenza di `local`». `ot` da' il modello spinto fino al tetto: risponde
# alla domanda che un revisore fa subito dopo, «e se lo allenaste di piu'?». Riportarli
# entrambi toglie ogni spazio all'obiezione — in una direzione o nell'altra.
#
# ⚠️ `EARLY_STOPPING=0` DA SOLO NON BASTA per `ot`: `_converged_loop`
# (`federated_eval.py:412`) ripristina comunque `best_state` a fine corsa, quindi darebbe lo
# stesso modello di `es`, ore dopo. Serve `KEEP_LAST_WEIGHTS=1` insieme. E' scritto anche in
# `config.py:334`, ed e' la trappola numero uno di questo esperimento.
#
# ── PERCHE' NON APPAIO I BUDGET ──────────────────────────────────────────────────
# Dare a `local` gli step di A2 lo RIDURREBBE su 4 delle 6 serie troncate (0,50-0,81x):
# handicappare la baseline per far vincere il metodo. Il budget effettivo si riporta come
# colonna di step per cella — l'informazione c'e' lo stesso, gratis, senza truccare la corsa.
#
# ── PERCHE' NON HO TOCCATO `_converged_loop` ─────────────────────────────────────
# Si potrebbero avere entrambi i modelli da UNA run: la traiettoria fino al punto di stop e'
# identica, basterebbe salvare anche i pesi finali. Ma quel ciclo e' il percorso di training
# di OGNI arm, inclusi i 19 job in volo, e un bug li' corrompe in silenzio tutta la campagna.
# Pago compute invece di rischio. Nel codice condiviso e' entrato solo il knob d'ambiente di
# `config.py`, che a variabile non settata e' un no-op verificato (output vuoto, valori 10000
# / 50000 / True / False).
#
# ── IL TETTO: 25 000 / 100 000, e perche' proprio quello ─────────────────────────
# Non e' un numero tondo a caso. L'early stop piu' tardivo osservato fra i client convergiti
# e' a step 9552 e la pazienza e' 2000: 25 000 lascia ~15 000 di margine oltre il caso
# peggiore misurato. E resta un TETTO — un client che ci arriva e' ANCORA
# `TRUNCATED, NOT CONVERGED` e va riportato come tale, non usato in silenzio.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"

MODE="${1:?es | ot}"
export S1_MAX_STEPS="${S1_MAX_STEPS:-25000}"
export S2_MAX_STEPS="${S2_MAX_STEPS:-100000}"
export LAUNCH_GPUS="${LAUNCH_GPUS:-0 4}"
export SLOTS_PER_GPU="${SLOTS_PER_GPU:-2}"
export NO_MPS=1
export FEDVQ_AMP=fp16
ZN='--window-normalization zscore'
say() { echo "[$(date -u '+%F %T') UTC | $(TZ=Europe/Madrid date '+%H:%M') Madrid][budget/$MODE] $*"; }

case "$MODE" in
  es)
    # Solo le celle TRONCATE. Le 4 serie gia' convergite non si rifanno: stesso seed e un
    # tetto che non morde danno la stessa traiettoria fino all'early stop, quindi
    # riprodurrebbero se' stesse bit per bit. Non rifarle vuol dire anche che quelle 4 serie
    # tengono i numeri di adesso, senza nessuna questione di mescolanza.
    TAG=zn_es
    SER_LOCAL="ucr_001,ucr_011,ucr_014,ucr_170,ucr_222,ucr_229"
    SER_CENTR="ucr_001,ucr_011,ucr_014,ucr_222,ucr_229"
    ;;
  ot)
    # TUTTE e 10 le serie: il sovrallenato non esiste per nessuna, nemmeno per quelle
    # convergite. E' una condizione nuova, non una riparazione.
    TAG=zn_ot
    # ORDINE NON ALFABETICO, DI PROPOSITO. `launch.sh` dispaccia in ordine di lista, e la
    # domanda a cui questa variante risponde — «e se le baseline le allenassimo di piu'?» —
    # si decide su poche serie. Davanti quelle con un divario `centralized - local` reale
    # (011, 014, 043, 170, 001, 086); in coda le quattro che non discriminano (082 tutti a
    # 0,00 · 083 con `local` gia' a 1,00 · 222 e 229 a 0,00 su top-K), che contribuiscono
    # ZERO a ogni contrasto appaiato. Cosi' la risposta arriva in ore invece che in giorni.
    SER_LOCAL="ucr_011,ucr_014,ucr_043,ucr_170,ucr_001,ucr_086,ucr_083,ucr_222,ucr_229,ucr_082"
    SER_CENTR="$SER_LOCAL"
    export EARLY_STOPPING=0
    export KEEP_LAST_WEIGHTS=1
    ;;
  *) echo "modo sconosciuto '$MODE' (es|ot)" >&2; exit 2;;
esac

# ⚠️ TAG NUOVO OBBLIGATORIO. Il `cohort_fingerprint` copre solo `datasets` e `seeds`: non
# copre il budget di training, ne' `window_normalization`, ne' `amp_dtype`. Due celle con lo
# stesso fingerprint possono quindi essere modelli diversi, e gli script che aggregano per
# fingerprint le dichiarerebbero confrontabili. Finche' non e' corretto, il tag distinto e'
# l'UNICA protezione (CLAUDE.md).
say "tetto s1=$S1_MAX_STEPS s2=$S2_MAX_STEPS  ·  tag=$TAG  ·  GPU $LAUNCH_GPUS"

say "=== local su $SER_LOCAL ==="
LAUNCH_ONLY_CLUSTERS="$SER_LOCAL" bash scripts/launch.sh --cohort ucr2p_10 --tag "$TAG" \
  --arms local --extra "$ZN"

say "=== centralized su $SER_CENTR ==="
LAUNCH_ONLY_CLUSTERS="$SER_CENTR" bash scripts/launch.sh --cohort ucr2p_10 --tag "$TAG" \
  --arms centralized --extra "$ZN"

say "=== $MODE finito (rc=$?) ==="
