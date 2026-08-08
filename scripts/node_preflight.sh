#!/usr/bin/env bash
# Preflight di un nodo di calcolo NUOVO (tipicamente il portatile/desktop di casa), da
# eseguire SU QUEL NODO, prima di trasferire 9 GB di ambiente.
#
#   bash node_preflight.sh
#
# Risponde a UNA domanda: questa macchina puo' eseguire una cella della coorte producendo
# numeri confrontabili con quelli di g2/g4? Se la risposta e' no, e' meglio saperlo in 10
# secondi che dopo il trasferimento.
#
# Perche' i controlli sono questi:
#
#  * ARCHITETTURA. Il nostro torch e' 2.11.0+cu128, compilato per
#    sm_75/80/86/90/100/120. Un cubin gira su capability piu' ALTE dello stesso major
#    (8.9 Ada esegue codice sm_86 -- e' esattamente cosi' che le RTX 4500 di g4 lavorano,
#    pur non essendo sm_89 nella lista), ma MAI su un major diverso. Volta sm_70 e' fuori:
#    verificato sul campo su g1, "no kernel image available". Pascal (GTX 10xx) e' fuori
#    per lo stesso motivo. In pratica: RTX 20xx / GTX 16xx o piu' recente.
#
#  * VRAM. Misurata sui job vivi di g2: 468-628 MiB per cella. La VRAM non e' il vincolo,
#    lo e' il calcolo. 4 GB bastano; il controllo serve solo a escludere le iGPU.
#
#  * PERCORSI. Il venv NON e' rilocabile: gli shebang sono assoluti
#    (#!/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10) e l'interprete vero vive
#    sotto /home/leonardo/.local/share/uv/. Il trapianto funziona solo se su questa
#    macchina esistono gli STESSI percorsi assoluti. Se l'utente locale non e' `leonardo`
#    servira' creare /home/leonardo (da root) o un bind mount.
#
#  * SSH VERSO g2. Il trasferimento lo inizia questa macchina: g2 non puo' raggiungerti
#    (NAT domestico). Se questo controllo fallisce, nessun altro passo e' possibile.

set -uo pipefail

G2=${G2:-leonardo@138.100.156.38}
VENV_PATH=/home/leonardo/PhD/TimeVQVAE-AD-M/.venv
UV_PATH=/home/leonardo/.local/share/uv
NEED_GB=${NEED_GB:-15}

# Architetture per cui il nostro torch ha cubin. Un major presente qui accetta anche
# capability minori PIU' ALTE (regola di compatibilita' binaria CUDA).
ARCHS="75 80 86 90 100 120"

fail=0
ok()   { printf '  \033[32mOK\033[0m    %s\n' "$1"; }
bad()  { printf '  \033[31m!!\033[0m    %s\n' "$1"; fail=$((fail+1)); }
warn() { printf '  \033[33m~~\033[0m    %s\n' "$1"; }

echo "=================================================================="
echo " PREFLIGHT NODO DI CALCOLO  ($(hostname) - $(uname -s) $(uname -m))"
echo "=================================================================="

echo; echo "--- 1. sistema operativo ---"
if [[ "$(uname -s)" != "Linux" ]]; then
  bad "non e' Linux ($(uname -s)). Il venv e' linux-x86_64: su macOS non gira in nessun modo.
        Su Windows serve WSL2 con supporto CUDA, e questo script va eseguito DENTRO WSL2."
else
  ok "Linux $(uname -r)"
fi

echo; echo "--- 2. la GPU esiste e il driver risponde ---"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  bad "nvidia-smi assente: nessuna GPU NVIDIA utilizzabile (o driver non installato).
        Senza GPU NVIDIA questa macchina non serve: la pipeline non ha un percorso CPU utile."
else
  drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
  ok "driver $drv"
  nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv,noheader 2>/dev/null \
    | while IFS=, read -r idx name cap mem; do
        printf '        GPU%s%s (cc %s, %s)\n' "$idx" "$name" "$cap" "$mem"
      done
fi

echo; echo "--- 3. l'architettura e' fra quelle compilate ---"
if command -v nvidia-smi >/dev/null 2>&1; then
  caps=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null)
  if [[ -z "$caps" ]]; then
    bad "non riesco a leggere la compute capability"
  fi
  usable=0
  while read -r cap; do
    [[ -z "$cap" ]] && continue
    major=${cap%%.*}; minor=${cap##*.}
    sm=$((major * 10 + minor))
    best=""
    for a in $ARCHS; do
      amaj=$((a / 10))
      # stesso major e cubin non piu' recente della scheda => eseguibile
      if [[ $amaj -eq $major && $a -le $sm ]]; then best=$a; fi
    done
    if [[ -n "$best" ]]; then
      if [[ "$best" -eq "$sm" ]]; then
        ok "cc $cap -> sm_$sm compilato nativamente"
      else
        ok "cc $cap -> esegue i cubin sm_$best (compatibilita' minor, come le Ada di g4)"
      fi
      # Blackwell (cc 10.x datacenter, 12.x consumer) non e' supportato dai driver vecchi,
      # qualunque cosa dica la lista delle architetture: il silicio stesso richiede r570+.
      # Un driver piu' vecchio non degrada, fallisce all'inizializzazione di CUDA.
      if [[ $major -ge 10 ]]; then
        dmaj=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)
        if [[ "$dmaj" =~ ^[0-9]+$ ]] && [[ $dmaj -ge 570 ]]; then
          ok "Blackwell + driver $dmaj (>= 570 richiesto)"
        else
          bad "Blackwell con driver ${dmaj:-?}: serve r570 o piu' recente, altrimenti CUDA non inizializza."
        fi
        # cuDNN 9.7 e' la prima con kernel per sm_120; la nostra e' 9.19, ma se qualcuno
        # ricrea il venv a mano il controllo serve.
        warn "promemoria: il venv trapiantato ha cuDNN 91900 -- non ricrearlo a mano sotto 90700"
      fi
      usable=$((usable+1))
    else
      bad "cc $cap NON eseguibile: nessun cubin del major $major in [$ARCHS].
        Volta (7.0), Pascal (6.x) e Maxwell (5.x) sono esclusi. Serve RTX 20xx / GTX 16xx o piu' recente."
    fi
  done <<< "$caps"
  [[ $usable -gt 0 ]] && ok "$usable scheda/e utilizzabile/i"
fi

echo; echo "--- 4. VRAM sufficiente (misurato: 0,5-0,7 GB per cella) ---"
if command -v nvidia-smi >/dev/null 2>&1; then
  while read -r mem; do
    mib=${mem%% *}
    if [[ "$mib" =~ ^[0-9]+$ ]] && [[ $mib -ge 3500 ]]; then
      ok "${mib} MiB -- basta per piu' celle in parallelo"
    else
      warn "${mib} MiB: stretta. Una cella sola, SLOTS_PER_GPU=1."
    fi
  done < <(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null)
fi

echo; echo "--- 5. spazio disco (${NEED_GB} GB: 8,2 venv + 0,8 dati + margine) ---"
target_fs=/home; [[ -d /home ]] || target_fs=$HOME
avail_gb=$(df -BG --output=avail "$target_fs" 2>/dev/null | tail -1 | tr -dc '0-9')
if [[ -n "$avail_gb" && "$avail_gb" -ge "$NEED_GB" ]]; then
  ok "${avail_gb} GB liberi su $target_fs"
else
  bad "solo ${avail_gb:-?} GB liberi su $target_fs, ne servono $NEED_GB"
fi

echo; echo "--- 6. i percorsi assoluti del venv sono creabili ---"
# Il venv non e' rilocabile: shebang e sysconfig puntano a questi percorsi esatti.
for p in "$VENV_PATH" "$UV_PATH"; do
  parent=$p
  while [[ ! -e "$parent" && "$parent" != "/" ]]; do parent=$(dirname "$parent"); done
  if [[ -e "$p" ]]; then
    ok "$p esiste gia'"
  elif [[ -w "$parent" ]]; then
    ok "$p creabile (il primo genitore esistente, $parent, e' scrivibile)"
  else
    bad "$p NON creabile: $parent non e' scrivibile da $(whoami).
        Il venv ha shebang assoluti, quindi il percorso non e' negoziabile. Rimedi:
          sudo mkdir -p $(dirname "$p") && sudo chown -R $(whoami) /home/leonardo"
  fi
done

echo; echo "--- 7. ssh verso g2 (il trasferimento lo inizi TU: g2 non ti raggiunge) ---"
if [[ "${SKIP_SSH_CHECK:-0}" == "1" ]]; then
  # Con un tunnel inverso attivo (ssh -R) e' g2 a spingere i dati: questa macchina non
  # ha bisogno di raggiungere g2, quindi il controllo non si applica.
  ok "saltato (SKIP_SSH_CHECK=1): il trasferimento arriva da g2 via tunnel inverso"
elif timeout 15 ssh -o BatchMode=yes -o ConnectTimeout=10 "$G2" 'echo pong' >/dev/null 2>&1; then
  ok "ssh $G2 funziona senza password (chiave gia' installata)"
else
  if timeout 15 ssh -o ConnectTimeout=10 -o PreferredAuthentications=none "$G2" true 2>&1 | grep -qi 'permission denied\|password'; then
    warn "g2 raggiungibile ma serve la password: installa la chiave con  ssh-copy-id $G2
        (rsync su 9 GB con password interattiva e' impraticabile)"
  else
    bad "g2 NON raggiungibile come $G2. Controlla VPN/host: senza questo non si trasferisce niente."
  fi
fi

echo
echo "=================================================================="
if [[ $fail -eq 0 ]]; then
  echo " ESITO: GO -- questa macchina puo' eseguire celle della coorte."
  echo " Passo successivo, da QUI:   bash node_bootstrap.sh"
else
  echo " ESITO: NO-GO -- $fail controlli falliti (vedi sopra)."
fi
echo "=================================================================="
exit $fail
