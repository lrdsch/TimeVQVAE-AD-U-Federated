#!/usr/bin/env bash
# `local` e `centralized` a K=128: la BASELINE e il TETTO che mancavano al blocco K=128.
#
# PERCHE'. `--codebook-size` scrive cfg.quantizer.codebook_size PRIMA di qualunque ramo di arm
# (federated_eval.py:1232): K non e' una manopola di federazione, e' un iperparametro del
# modello, e `local`/`centralized` hanno un codebook come tutti gli altri. Il blocco cb128
# conteneva solo `federated_cb_only` e `federated_fedavg_cb_only`, quindi permetteva il
# contrasto fra primitive di codebook a K=128, ma NON la domanda che conta -- «la federazione
# continua a non battere `local` anche a K=128?» -- perche' `local@128` non esisteva e
# confrontarlo con `local@64` avrebbe messo K dentro il contrasto.
#
# Ripetere tutti e 8 gli arm sarebbe stato 144 celle (piu' di meta' campagna) e in gran parte
# ridondante: l'asse del fattoriale e' gia' coperto. Mancava il RIFERIMENTO, non la copertura.
#
# 🔴 UNA SERIE, UNA MACCHINA vale anche qui, ANZI SOPRATTUTTO qui: queste celle si confrontano
# con le celle cb128 della stessa serie, quindi devono girare sull'host dove quelle sono
# girate. Questo script fa le 4 serie di g2; le altre 5 vanno su g4.
set -u
cd /home/leonardo/PhD/TimeVQVAE-AD-U-Federated

for c in "$@"; do
  echo "=== [$(date '+%F %T')] $c: local+centralized @ K=128 su $(hostname) ==="
  bash scripts/launch.sh --cohort "$c" --tag "${c}_cb128base" \
    --arms local,centralized --extra "--codebook-size 128"
done
echo "=== [$(date '+%F %T')] baseline K=128 completate: $* ==="

# ── nota sulla ripartizione su g4 ────────────────────────────────────────────────────────────
# Le 5 serie di g4 girano su un unico flusso sparso sulle TRE schede (LAUNCH_GPUS="0 4 5",
# SLOTS_PER_GPU=2) invece che ciascuna sulla scheda dove sta il suo blocco cb128. Si puo' fare
# perche' il confondente SCHEDA e' stato misurato ed e' nullo (top-1 identico fra GPU0 e GPU1
# su g2; le tre Ada di g4 sono lo stesso modello), mentre quello che conta -- la MACCHINA --
# resta rispettato: tutte e cinque su g4, dove stanno i loro cb128.
# Nessun rischio di collisione: il tag `*_cb128base` non compare in nessun altro script.
