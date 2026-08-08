#!/usr/bin/env bash
# Usa la macchina di casa come nodo di calcolo, attraverso il TUNNEL INVERSO che lei stessa
# apre verso g2:
#
#     (da casa)   ssh -R 2222:localhost:22 leonardo@138.100.156.38
#     (da g2)     bash scripts/run_on_home.sh <fase>
#
#   fasi:  probe | pull | check | launch | status | fetch
#
# ── PERCHE' IL TUNNEL, E COSA IMPLICA ─────────────────────────────────────────────
# g2 non puo' aprire una connessione verso casa: porta 22 chiusa dall'esterno e ICMP
# filtrato (NAT domestico, verificato). Il tunnel inverso ribalta la direzione: la
# macchina di casa espone la propria sshd su 127.0.0.1:2222 DI g2. Da li' in poi g2 puo'
# spingere dati ed eseguire comandi.
#
# ⚠ IL TUNNEL E' FRAGILE E I JOB NON DEVONO DIPENDERNE. Muore quando si chiude la sessione
# ssh di casa, quando il portatile si sospende, quando il NAT scarta la connessione inattiva.
# Percio':
#   * i job si lanciano dentro `screen -dmS` SULLA macchina di casa: sopravvivono alla
#     caduta del tunnel, che e' solo un telecomando;
#   * niente working dir remota, niente sshfs. Il repo, i dati e i checkpoint stanno sul
#     disco locale di casa. Il ponte passivo si e' gia' rotto in modo strutturale su LAN
#     (11 job congelati 1h20m, zero errori): su NAT domestico sarebbe peggio;
#   * i risultati si vanno a prendere con `fetch` quando il tunnel c'e'. Se non c'e', i job
#     continuano lo stesso e si recuperano dopo.
#
# ── PERCHE' fp16 E NON bf16 ───────────────────────────────────────────────────────
# FEDVQ_AMP e' pinnato a fp16 da launch.sh. Su Blackwell fp16 ha tensor core pieni, quindi
# la scelta non costa niente qui, e tiene le celle confrontabili con g2 (Turing, dove bf16
# e' emulato e piu' lento di fp32) e con g4.

set -uo pipefail

HOME_ALIAS=${HOME_ALIAS:-homegpu}
REPO=${REPO:-/home/leonardo/PhD/TimeVQVAE-AD-U-Federated}
VENV=${VENV:-/home/leonardo/PhD/TimeVQVAE-AD-M/.venv}
UVPY=${UVPY:-/home/leonardo/.local/share/uv/python/cpython-3.10.20-linux-x86_64-gnu}
SCREEN=${SCREEN:-zn_home}
PHASE=${1:-}

SSH="ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=30 -o ServerAliveCountMax=3"
RS="rsync -a --info=progress2 --partial -e '$SSH'"

alive() { $SSH "$HOME_ALIAS" 'true' 2>/dev/null; }

case "$PHASE" in
probe)
  alive || { echo "!! tunnel assente o chiave non autorizzata. Da casa:
     ssh -R 2222:localhost:22 leonardo@138.100.156.38
   e una volta sola, sempre da casa:
     mkdir -p ~/.ssh && chmod 700 ~/.ssh
     echo '$(cat ~/.ssh/id_ed25519.pub)' >> ~/.ssh/authorized_keys
     chmod 600 ~/.ssh/authorized_keys" >&2; exit 1; }
  echo "== nodo raggiungibile: $($SSH "$HOME_ALIAS" 'hostname; whoami' | tr '\n' ' ') =="
  SKIP=1 $SSH "$HOME_ALIAS" 'SKIP_SSH_CHECK=1 bash -s' < "$(dirname "$0")/node_preflight.sh"
  ;;

pull)
  # g2 SPINGE (il verbo e' "pull" dal punto di vista del nodo, che riceve il suo ambiente).
  alive || { echo "!! tunnel giu'" >&2; exit 1; }
  echo "== 1/4 interprete uv =="
  $SSH "$HOME_ALIAS" "mkdir -p '$(dirname "$UVPY")'"
  rsync -a --info=progress2 --partial -e "$SSH" "$UVPY/" "$HOME_ALIAS:$UVPY/" || exit 1
  echo "== 2/4 venv 8,2 GB (con -H: senza hardlink si gonfia) =="
  $SSH "$HOME_ALIAS" "mkdir -p '$(dirname "$VENV")'"
  rsync -aH --info=progress2 --partial -e "$SSH" "$VENV/" "$HOME_ALIAS:$VENV/" || exit 1
  echo "== 3/4 codice del repo (artifacts esclusi: restano su g2) =="
  $SSH "$HOME_ALIAS" "mkdir -p '$REPO'"
  rsync -a --info=progress2 --partial -e "$SSH" \
    --exclude 'artifacts/' --exclude 'artifacts_archive/' --exclude '.git/' \
    --exclude 'logs/' --exclude '.released_results/' --exclude 'data/raw/' \
    "$REPO/" "$HOME_ALIAS:$REPO/" || exit 1
  echo "== 4/4 dati ucr_split_w2p (752 MB) =="
  $SSH "$HOME_ALIAS" "mkdir -p '$REPO/data/raw'"
  rsync -a --info=progress2 --partial -e "$SSH" \
    "$REPO/data/raw/ucr_split_w2p/" "$HOME_ALIAS:$REPO/data/raw/ucr_split_w2p/" || exit 1
  echo "fatto."
  ;;

check)
  alive || { echo "!! tunnel giu'" >&2; exit 1; }
  $SSH "$HOME_ALIAS" "cd '$REPO' && bash scripts/node_bootstrap.sh check"
  ;;

launch)
  alive || { echo "!! tunnel giu'" >&2; exit 1; }
  # UNA cella sola, non tre. Il nodo e' una RTX 5060 Laptop: 8 GB bastano (0,5-0,7 GB per
  # cella, misurato) ma il tetto e' 77 W, quindi sotto carico prolungato va in throttling
  # termico e sta sotto una 3090. Tre job concorrenti su quel chip non aumentano il
  # throughput, lo spalmano -- e' la stessa lezione di g4 (da 9 a 15 job: 0,93x, cioe'
  # PEGGIO), aggravata dai limiti termici di un portatile.
  #
  # Sceglo `ucr_043` fra le due candidate: e' la piu' corta (1h46m contro 3h12m di ucr_014
  # su 3090) ED e' l'unica delle due nel regime "shift di covariata", cioe' esattamente
  # quello che la BN poolata dovrebbe curare. Massima informazione per ora di calcolo.
  # screen -dmS: il job sopravvive alla caduta del tunnel.
  # Il job va in un FILE trasferito ed eseguito, non in una stringa annidata dentro screen:
  # il quoting a tre livelli (locale -> ssh -> screen -> bash -c) e' il modo classico di
  # lanciare un comando diverso da quello che si e' letto.
  JOBFILE=$(mktemp /tmp/zn_home_job.XXXXXX.sh)
  cat > "$JOBFILE" <<EOF
#!/usr/bin/env bash
set -uo pipefail
cd '$REPO' || exit 1
export FEDVQ_AMP=fp16
echo "[\$(date '+%F %T')] avvio zn_a1p su ucr_043 (encoder FedAvg + BN poolata + cb suffstat + prior federato)"
LAUNCH_ONLY_CLUSTERS="ucr_043" LAUNCH_GPUS="0" SLOTS_PER_GPU=1 \\
  bash scripts/launch.sh --cohort ucr2p_10 --tag zn_a1p --arms federated_enc_fedavg \\
  --extra "--window-normalization zscore --fed-enc-bn shared --fed-enc-cb suffstat --fed-enc-prior partial"
echo "[\$(date '+%F %T')] FINITO"
EOF
  rsync -a -e "$SSH" "$JOBFILE" "$HOME_ALIAS:$REPO/scripts/_zn_home_job.sh" || exit 1
  rm -f "$JOBFILE"
  $SSH "$HOME_ALIAS" "cd '$REPO' && mkdir -p logs && \
      (screen -dmS '$SCREEN' bash scripts/_zn_home_job.sh \
       || nohup bash scripts/_zn_home_job.sh > logs/home_launch.log 2>&1 &) && echo avviato" || exit 1
  echo "lanciato sul nodo di casa (screen '$SCREEN'). Sopravvive alla caduta del tunnel."
  ;;

status)
  alive || { echo "!! tunnel giu' (i job continuano lo stesso)" >&2; exit 1; }
  $SSH "$HOME_ALIAS" "
    echo '--- screen ---'; screen -ls 2>/dev/null | head
    echo '--- GPU ---'; nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader
    echo '--- celle finite ---'; ls -1 '$REPO'/artifacts/runs/zn_a1{p,s}/ucr_split_w2p/*.json 2>/dev/null | sed 's|.*/||' || echo nessuna
    echo '--- ultime righe ---'; tail -n 3 \$(ls -t '$REPO'/logs/runs/zn_a1*/*.log 2>/dev/null | head -3) 2>/dev/null"
  ;;

fetch)
  alive || { echo "!! tunnel giu'" >&2; exit 1; }
  for tag in zn_a1p zn_a1s; do
    $SSH "$HOME_ALIAS" "test -d '$REPO/artifacts/runs/$tag'" || continue
    echo "== $tag -> g2 =="
    mkdir -p "$REPO/artifacts/runs/$tag"
    rsync -a --info=progress2 --partial -e "$SSH" \
      "$HOME_ALIAS:$REPO/artifacts/runs/$tag/" "$REPO/artifacts/runs/$tag/" || exit 1
  done
  echo "fatto. Le celle sono su g2 e l'audit puo' leggerle."
  ;;

*)
  sed -n '2,10p' "$0"; exit 2 ;;
esac
