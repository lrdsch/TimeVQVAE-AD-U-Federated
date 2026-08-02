#!/usr/bin/env bash
# I 3 floor gia' completi, rigirati per prendere la colonna @100 (detect.PROTOCOL_TOLERANCE).
# Il floor non salva i punteggi grezzi (floor_eval.py:142 mette save_scores=False), quindi
# non e' ricalcolabile offline come il deep: va rigirato.
#
# GIRA SU g4, non su g2. Il floor e' puro CPU (teste analitiche: media mobile, AR, PCA) e non
# tocca la GPU. Su g2 andava a 15 righe in 2 h -> 28 h per serie, perche' g2 ha 16 core
# Bronze @1,9 GHz gia' saturi dai training. g4 ha 48 core Silver, 3,2x piu' veloci per core,
# con ~33 liberi. FLOOR_JOBS=16 li usa senza avvicinarsi al limite della macchina condivisa.
#
# TAG NUOVI (*_floor100): non si sovrascrivono 13 h di dati validi. Le teste sono analitiche,
# quindi il re-run DEVE riprodurre le colonne vecchie -- controllo di riproducibilita' gratis.
set -u
cd /home/leonardo/PhD/TimeVQVAE-AD-U-Federated

for c in ucr001 ucr011 ucr222; do
  echo "=== [$(date '+%F %T')] re-floor $c su $(hostname) ==="
  FLOOR_JOBS=16 bash scripts/launch.sh --cohort "$c" --engine floor \
    --heads ma_c,ma_causal,ar,pca --modes all --tag "${c}_floor100"
done
echo "=== [$(date '+%F %T')] re-floor finito: confrontare *_floor100 con *_floor ==="
