#!/usr/bin/env bash
# Esegue una fase della pipeline su g4 (aida-g4), leggendo e scrivendo il repo di g2
# attraverso il mount sshfs passive (scripts/passive_mount.sh). I dati NON si spostano:
# g4 presta solo CPU e GPU.
#
#   bash scripts/run_on_g4.sh "<comando bash da eseguire nel repo su g4>" <nome-screen>
#
# Vincoli che questo script codifica, e perche':
#  * LAUNCH_GPUS="4 5"  -- SOLO le due RTX 4500 Ada. g4 e' un server di dipartimento con 20+
#    utenti; la GPU 2 e' occupata da giorni. Prendere piu' di due schede e' scortese e ce le
#    fa togliere. Le due sono IDENTICHE fra loro apposta: mescolare Ada e 3090 darebbe job di
#    durata diversa e, con cudnn.benchmark=True (che sceglie i kernel a tempo), traiettorie
#    di training diverse dentro la stessa serie.
#  * NO_MPS=1 -- MAI avviare un daemon MPS su un host condiviso: instraderebbe i contesti
#    CUDA degli altri utenti attraverso il nostro server.
#  * screen su g4 -- il lavoro deve sopravvivere alla chiusura della connessione ssh.
set -uo pipefail

CMD=${1:?serve il comando da eseguire su g4}
NAME=${2:?serve il nome dello screen su g4}
G4=${G4:-leonardo@g4.etsisi.upm.es}
REPO_G4=${REPO_G4:-/home/leonardo/PhD/TimeVQVAE-AD-U-Federated}

ssh -o BatchMode=yes "$G4" "test -d '$REPO_G4/scripts'" \
  || { echo "il mount su g4 non c'e' o e' morto: rilancia scripts/passive_mount.sh" >&2; exit 1; }

# Il set di schede e' sovrascrivibile perche' due flussi paralleli su g4 devono poter prendere
# schede DISGIUNTE (altrimenti si contendono gli slot senza saperlo). Il default resta "4 5":
# chi non specifica niente ottiene il comportamento prudente di sempre.
#
# ⚠ Qualunque valore si passi, devono essere schede della STESSA architettura. Su g4 le uniche
# tre RTX 4500 Ada sono la 0, la 4 e la 5; 1/2/3 sono 3090 (sm_86). Mescolarle darebbe job di
# durata diversa e, con cudnn.benchmark=True che sceglie i kernel a tempo, traiettorie di
# training diverse DENTRO la stessa serie -- cioe' il confondente hardware spostato dal punto
# dove e' innocuo (fra serie) a quello dove non lo e' (dentro un contrasto).
GPUS_G4=${LAUNCH_GPUS:-4 5}

ssh -o BatchMode=yes "$G4" \
  "cd '$REPO_G4' && screen -dmS '$NAME' bash -lc '
     export LAUNCH_GPUS=\"$GPUS_G4\" NO_MPS=1
     export PY=/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10
     echo \"[\$(date +%F\\ %T)] g4 start: $NAME\"
     $CMD
     echo \"[\$(date +%F\\ %T)] g4 done: $NAME\"
   '"
echo "lanciato su g4 nello screen '$NAME'"
ssh -o BatchMode=yes "$G4" "screen -ls | sed -n '2,\$p'"
