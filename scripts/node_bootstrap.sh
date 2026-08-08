#!/usr/bin/env bash
# Porta l'ambiente della coorte su un nodo NUOVO (il portatile di casa) e rimanda indietro
# i risultati. Da eseguire SU QUEL NODO, dopo che node_preflight.sh ha detto GO.
#
#   bash node_bootstrap.sh pull     # 1. porta qui ambiente + codice + dati (~9 GB, una volta sola)
#   bash node_bootstrap.sh check    # 2. valida il trapianto (7 controlli)
#   bash node_bootstrap.sh push     # 3. rimanda a g2 i risultati delle celle finite
#
# ── PERCHE' NON sshfs ──────────────────────────────────────────────────────────────
# Il ponte sshfs esiste per g4, dove serve a "prestare CPU/GPU senza spostare i dati"
# perche' g4 sta sulla stessa LAN di dipartimento. Qui la premessa cade due volte:
#
#  1. I dati sono PICCOLI. La build ucr_split_w2p pesa 752 MB e una cella finita rimanda
#     indietro 4 KB di JSON piu' ~32 MB di checkpoint. Non c'e' niente da "non spostare":
#     copiare una volta costa meno che montare per ore.
#  2. Il mount passivo si e' gia' rotto in modo STRUTTURALE su LAN — 11 job congelati per
#     1h20m senza un solo messaggio di errore, e i resume dei job uccisi non si recuperano.
#     Su una linea domestica dietro NAT, con la macchina che puo' sospendersi e il Wi-Fi che
#     puo' cambiare AP, un training di 3-9 ore con la working dir su sshfs non e' un rischio:
#     e' un guasto silenzioso programmato.
#
# Copiare-eseguire-ricopiare non ha stato condiviso: se cade la rete, il job continua.
#
# ── PERCHE' il trasferimento parte da QUI ──────────────────────────────────────────
# g2 non puo' raggiungerti (porta 22 chiusa, ICMP filtrato: NAT domestico). Il traffico lo
# apri tu. Per lo stesso motivo nessuno su g2 puo' sorvegliare questi job: il polling e i
# risultati li porti tu con `push`.

set -uo pipefail

G2=${G2:-leonardo@138.100.156.38}
REPO=${REPO:-/home/leonardo/PhD/TimeVQVAE-AD-U-Federated}
VENV=${VENV:-/home/leonardo/PhD/TimeVQVAE-AD-M/.venv}
UVPY=${UVPY:-/home/leonardo/.local/share/uv/python/cpython-3.10.20-linux-x86_64-gnu}
MODE=${1:-}

R="rsync -a --info=progress2 --partial"

case "$MODE" in
pull)
  echo "== 1/4 interprete uv (l'oggetto a cui puntano gli shebang del venv) =="
  mkdir -p "$(dirname "$UVPY")"
  $R "$G2:$UVPY/" "$UVPY/" || exit 1

  echo "== 2/4 venv 8,2 GB =="
  # -H preserva gli hardlink: senza, il venv si gonfia parecchio.
  mkdir -p "$(dirname "$VENV")"
  rsync -aH --info=progress2 --partial "$G2:$VENV/" "$VENV/" || exit 1

  echo "== 3/4 codice del repo (senza artifacts: quelli restano su g2) =="
  mkdir -p "$REPO"
  $R --exclude 'artifacts/' --exclude 'artifacts_archive/' --exclude '.git/' \
     --exclude 'logs/' --exclude '.released_results/' --exclude 'data/raw/' \
     "$G2:$REPO/" "$REPO/" || exit 1

  echo "== 4/4 dati della build ucr_split_w2p (752 MB) =="
  mkdir -p "$REPO/data/raw"
  $R "$G2:$REPO/data/raw/ucr_split_w2p/" "$REPO/data/raw/ucr_split_w2p/" || exit 1

  echo; echo "fatto. Ora:  bash node_bootstrap.sh check"
  ;;

check)
  PY="$VENV/bin/python3.10"
  [[ -x "$PY" ]] || { echo "!! $PY non eseguibile: il pull non e' andato a buon fine" >&2; exit 1; }
  echo "== interprete =="
  "$PY" -c "import sys;print(sys.version)" || exit 1
  echo "== torch e architettura =="
  "$PY" - <<'EOF' || exit 1
import torch
cap = torch.cuda.get_device_capability(0)
sm = f"sm_{cap[0]}{cap[1]}"
arch = torch.cuda.get_arch_list()
print(f"torch {torch.__version__} cuda {torch.version.cuda}")
print(f"scheda: {torch.cuda.get_device_name(0)} {sm}")
print(f"arch compilate: {arch}")
same_major = [a for a in arch if a.startswith(f"sm_{cap[0]}")]
assert same_major, f"{sm} non eseguibile: nessun cubin del major {cap[0]} in {arch}"
print(f"eseguibile via {same_major[-1]}")
EOF
  echo "== fp16 accelera davvero (su Turing bf16 e' emulato: la coorte e' pinnata a fp16) =="
  "$PY" - <<'EOF' || exit 1
import time, torch
res = {}
for dt, lab in ((torch.float32, "fp32"), (torch.float16, "fp16")):
    x = torch.randn(4096, 4096, device="cuda", dtype=dt)
    for _ in range(3): x @ x
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(10): x @ x
    torch.cuda.synchronize(); res[lab] = (time.perf_counter() - t) / 10 * 1000
sp = res["fp32"] / res["fp16"]
print(f"matmul 4096^3: fp32 {res['fp32']:.1f} ms · fp16 {res['fp16']:.1f} ms -> fp16 {sp:.1f}x")
assert sp > 1.5, f"fp16 non accelera (x{sp:.2f}): tensor core assenti o non usati"
EOF
  echo "== il codice del repo importa =="
  cd "$REPO" && FEDVQ_AMP=fp16 "$PY" -c "
import config, pipeline.federated, pipeline.federated_eval
from pipeline.federated import _amp_dtype
print('moduli ok · _amp_dtype ->', _amp_dtype())" || exit 1
  echo; echo "TUTTO OK. Puoi lanciare le celle."
  ;;

push)
  # Solo i risultati: i JSON sono 4 KB, i checkpoint ~32 MB a cella e servono alla sonda
  # latente. Mai --delete: su g2 ci sono i risultati degli altri nodi.
  for tag in zn_a1p zn_a1s; do
    [[ -d "$REPO/artifacts/runs/$tag" ]] || continue
    echo "== $tag -> g2 =="
    ssh "$G2" "mkdir -p '$REPO/artifacts/runs/$tag'"
    $R "$REPO/artifacts/runs/$tag/" "$G2:$REPO/artifacts/runs/$tag/" || exit 1
  done
  echo "fatto."
  ;;

*)
  sed -n '2,12p' "$0"; exit 2 ;;
esac
