#!/usr/bin/env bash
# Monta una dir di g2 su g4 SENZA che g4 debba autenticarsi verso g2.
# Il server SFTP gira qui su g2; sshfs su g4 gira in modalita' "passive" e
# parla attraverso la connessione ssh uscente g2->g4. Zero accessi in entrata a g2.
#
#   uso:  passive_mount.sh <dir-su-g2> <mountpoint-su-g4>
#
# Il mount vive finche' vive questo processo. Per smontare: kill di questo PID.
#
# ⚠ RISCRITTO 2026-08-01. La versione precedente cablava le due direzioni con una pipeline
# di shell e un FIFO:
#
#     ssh ... < "$FIFO" | "$SFTP" > "$FIFO"
#
# cioe' un ciclo chiuso con due buffer da 64 KB. Si e' impiantato in deadlock alle 15:54
# sotto 11 job concorrenti (ssh in `pipe_write`, 2 MB fermi nella Recv-Q del socket, tutti i
# processi di g4 in stato D per 1h20m). Non era un incidente: con abbastanza I/O in volo il
# ciclo si chiude sempre, e allargare i buffer sposta solo la soglia.
#
# Ora il cablaggio passa da scripts/_passive_relay.py, che tiene lettori e scrittori su thread
# separati: un lettore non e' mai bloccato dal proprio scrittore, quindi l'attesa circolare non
# puo' formarsi. Il perche' completo e' nel docstring di quel file.
set -uo pipefail

SRC=${1:?serve la dir sorgente su g2}
MNT=${2:?serve il mountpoint su g4}
G4=${G4:-leonardo@g4.etsisi.upm.es}
SFTP=/usr/lib/openssh/sftp-server
# python3 di sistema apposta, non il venv: il venv vive sotto /home/leonardo/PhD e il suo
# interprete e' un symlink nella dir uv -- cioe' proprio le due dir che questo script monta.
# Il ponte non deve dipendere da cio' che il ponte serve. Il relay usa solo la stdlib.
PY=${PY:-python3}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

[ -d "$SRC" ] || { echo "SORGENTE INESISTENTE: $SRC" >&2; exit 1; }
[ -x "$SFTP" ] || { echo "sftp-server assente: $SFTP" >&2; exit 1; }

echo "[$(date +%T)] mount  g2:$SRC  ->  g4:$MNT  (passive, nessun login g4->g2)"
exec "$PY" "$HERE/_passive_relay.py" --host "$G4" --src "$SRC" --mnt "$MNT" --sftp "$SFTP"
