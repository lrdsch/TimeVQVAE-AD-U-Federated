#!/usr/bin/env bash
# Le 2 celle `ucr083_proto_count`, staccate da `g083_tail.sh` e messe IN PARALLELO a `cb128`.
#
# PERCHE' SEPARATO. `g083_tail.sh` chiamava `launch.sh` due volte di fila — cb128 e poi
# proto_count — e due chiamate sequenziali sono **due onde**, non una: i 6 slot non si
# condividono fra invocazioni diverse. Sarebbero state ~5 h + ~5 h invece di ~5 h totali, e la
# seconda onda avrebbe incrociato `gpu0_g4` che arriva a proto_count verso le 04:30.
# Ho ucciso **solo il ciclo** di g083_tail.sh (pid del processo la cui riga di comando e'
# esattamente `bash scripts/g083_tail.sh`): i 4 job di cb128 gia' in volo sono rimasti vivi come
# orfani e finiscono da soli — `launch.sh:352` li lancia con `nohup ... > file 2>&1 &`, cioe'
# redirezione diretta su file e non una pipe verso il padre, quindi la morte del padre non li
# tocca. Verificato: 4 job padre vivi e log che avanzano.
#
# ⚠ Le virgolette di --extra devono restare in uno SCRIPT: passate a `run_on_g4.sh` finiscono
# dentro un `bash -lc '...'` e `count` si perde (argparse: "expected one argument").
set -u
cd /home/leonardo/PhD/TimeVQVAE-AD-U-Federated

export SLOTS_PER_GPU=1           # 2 schede x 1 = 2 slot per 2 celle
bash scripts/launch.sh --cohort ucr083 --tag ucr083_proto_count \
  --arms federated_enc_fedproto --extra "--fedproto-agg count"
echo "=== [$(date '+%F %T')] ucr083_proto_count completo (2 celle) ==="
