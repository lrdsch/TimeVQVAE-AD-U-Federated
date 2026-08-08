#!/usr/bin/env bash
# Esegue una fase della pipeline su g4 (aida-g4), leggendo e scrivendo il repo di g2
# attraverso il mount sshfs passive (scripts/passive_mount.sh). I dati NON si spostano:
# g4 presta solo CPU e GPU.
#
#   bash scripts/run_on_g4.sh "<comando bash da eseguire nel repo su g4>" <nome-screen>
#
# Vincoli che questo script codifica, e perche':
#  * L'UNICA regola dura rimasta (2026-08-06): MAI una GPU con processi di ALTRI UTENTI.
#    Il tetto sul numero di schede e quello sugli slot per scheda sono REVOCATI: g4 e' tutta
#    nostra. Verificata a runtime piu' sotto, non solo documentata -- e il controllo e' sul
#    PROPRIETARIO dei processi, mai sulla memoria libera: una scheda a 1 MiB puo' essere di
#    qualcun altro che sta per ripartire, e una a 20 GiB puo' essere tutta nostra.
#
#    ⚠ NOTA DI THROUGHPUT, non un vincolo: g4 satura misuratamente a ~3 job per scheda (da 9
#    a 15 job il throughput fa 0,93x, cioe' PEGGIO -- senza MPS il driver alterna i contesti
#    invece di sovrapporre i kernel). Superare i 3 slot e' permesso, ma non rende piu' veloce.
#  * LAUNCH_GPUS="4 5" resta il default prudente. Le due sono IDENTICHE fra loro apposta:
#    mescolare Ada e 3090 darebbe job di durata diversa e, con cudnn.benchmark=True (che
#    sceglie i kernel a tempo), traiettorie di training diverse dentro la stessa serie.
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
# ⛔ Il default NON e' piu' "4 5": dal 2026-08-06 **GPU0 e GPU5 sono RISERVATE** (utente), sono
# le schede che cede ad altri e non si toccano nemmeno se vuote. Delle Ada resta quindi la sola
# GPU4; "1 2" sono due 3090, stessa architettura fra loro come vuole il vincolo del contrasto.
GPUS_G4=${LAUNCH_GPUS:-1 2}

# Guardiano delle riservate. Esce 4 se la lista tocca 0 o 5; deroga solo con
# G4_ALLOW_RESERVED=1, che e' scomoda apposta. Vedi scripts/_g4_gpu.sh e CLAUDE.md.
bash "$(dirname "${BASH_SOURCE[0]}")/_g4_gpu.sh" "$GPUS_G4" || exit 4

# ── L'unico vincolo rimasto, verificato invece che sperato ─────────────────────────────────
#
# Perche' un controllo e non un commento: il 2026-08-05 ho lanciato due sonde sulle Ada 23
# secondi dopo che `zn_ada_guard.sh` le aveva restituite al dipartimento, e me le sono viste
# uccidere con `exit 143` senza capire perche' per tre tentativi. Un vincolo che si puo'
# violare per distrazione, prima o poi si viola: va messo dove e' impossibile aggirarlo per
# sbaglio (stessa logica della z-norm pinnata nella coorte invece che passata come --extra).
#
# ⚠ Il controllo vale AL LANCIO. Se un altro utente arriva su una scheda dopo, questo script
# non se ne accorge -- servirebbe un presidio periodico come `zn_ada_guard.sh`.
_gpu_census=$(ssh -o BatchMode=yes "$G4" '
  me=$(id -un)
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader | tr -d " " > /tmp/.g4_uuid.$$
  nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | tr -d " " | while IFS=, read -r u p; do
    idx=$(awk -F, -v x="$u" "\$2==x{print \$1}" /tmp/.g4_uuid.$$)
    usr=$(ps -o user= -p "$p" 2>/dev/null | tr -d " ")
    [ -n "$idx" ] && [ -n "$usr" ] && echo "$idx $usr"
  done | sort -u
  rm -f /tmp/.g4_uuid.$$
') || { echo "non riesco a leggere lo stato GPU di g4" >&2; exit 1; }

_theirs=$(awk -v m="leonardo" '$2!=m{print $1}' <<<"$_gpu_census" | sort -u)

# Usare la scheda di un altro non e' inefficienza, e' rubargliela.
for g in $GPUS_G4; do
  if grep -qx "$g" <<<"$_theirs"; then
    echo "RIFIUTO: GPU$g su g4 e' occupata da un altro utente ($(awk -v x="$g" '$1==x{print $2}' <<<"$_gpu_census" | paste -sd,)) -- non si tocca." >&2
    exit 2
  fi
done
echo "[vincoli g4] ok: nessuna delle schede richieste ($GPUS_G4) ha processi di altri utenti" \
     "$([ -n "$_theirs" ] && echo "| di altri: $(tr '\n' ' ' <<<"$_theirs")" || echo "| g4 e' interamente nostra")"

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
