#!/usr/bin/env bash
# I 4 floor MANCANTI: ucr_043, ucr_083, ucr_086, ucr_170.
#
# BUCO TROVATO 2026-08-01 18:47. Hanno il floor solo 5 serie su 9 (001, 011, 014, 222, 229):
# i tre script deep di g4 (`g4_chain.sh`, `g4_chain_gpu0.sh`, `g4_chain_gpu5.sh`) lanciano solo
# l'engine deep e nessuno lancia il floor, quindi le 4 serie nuove sarebbero arrivate a fine
# campagna senza baseline. Il floor non e' un contorno: e' la calibrazione che dice se il deep
# batte una media mobile a zero parametri, e senza di lui le righe di quelle serie non sono
# interpretabili.
#
# Gira su g4 perche' e' PURO CPU (teste analitiche: media mobile, AR, PCA — zero GPU): su g2,
# 16 core Bronze gia' saturi dai training, faceva 15 righe in 2 h. Qui non toglie GPU a
# nessuno dei tre flussi deep.
#
# Niente FLOOR_JOBS fisso: floor_eval.py calcola da solo il budget come
# (core - job vivi - 2), quindi si adatta al carico che i flussi deep gli lasciano invece di
# fidarsi di un numero cablato che sarebbe sbagliato appena il carico cambia.
set -u
cd /home/leonardo/PhD/TimeVQVAE-AD-U-Federated

# ⚠ CORRETTO 2026-08-02. Mancava questa guardia, che invece gli script deep hanno: `ucr043` e
# `ucr086` non avevano ancora un file di coorte (le crea la catena deep quando ARRIVA a quella
# serie, e non c'era ancora arrivata). launch.sh e' uscito subito e i due floor sono morti in
# silenzio, mentre `ucr083` e `ucr170` passavano perche' i flussi gpu0/gpu5 avevano gia' creato
# le loro. Scoperto solo guardando la tabella: 2 floor a 0/210 senza un errore visibile.
PYBIN=${PY:-/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10}
for c in ucr043 ucr083 ucr086 ucr170; do
  s="ucr_${c#ucr}"
  [ -f "cohorts/$c.json" ] || \
    "$PYBIN" scripts/cohort.py new "$c" --datasets ucr_split,ucr_split_w2p --clusters "$s"
  echo "=== [$(date '+%F %T')] floor $c su $(hostname) ==="
  bash scripts/launch.sh --cohort "$c" --engine floor \
    --heads ma_c,ma_causal,ar,pca --modes all --tag "${c}_floor"
done
echo "=== [$(date '+%F %T')] i 4 floor mancanti sono completi ==="
