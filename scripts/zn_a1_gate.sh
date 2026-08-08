#!/usr/bin/env bash
# GATE FUNZIONALE del ramo A1 — accordo di token fra client = 100%, o la cella non vale.
#
# PERCHE' UN GATE E NON UNA LOSS. `federated_enc_fedavg` col default `--fed-enc-bn
# buffers_local` condivide gamma/beta e tiene LOCALI le running stats: al momento della
# valutazione ogni client usa una CHIMERA -- pesi del consenso, statistiche sue. La sonda
# D2b l'ha misurata PEGGIORE dell'encoder straniero intero (accordo 0,10-0,13, quota 1,48).
# Una loss di training che scende non se ne accorge: il difetto e' nell'oggetto valutato,
# non nell'ottimizzazione. L'unica cosa che lo vede e' una misura FUNZIONALE.
#
# COSA DEVE DARE. Con `--fed-enc-bn shared` il gate deve passare PER COSTRUZIONE: alla fine
# di ogni round il server media i pesi (_fedavg_encoder) e poi mette in comune le statistiche
# per legge della varianza totale (_pool_encoder_bn, federated.py:1004), quindi dopo l'ultimo
# round i 5 encoder sono IDENTICI; il codebook lo e' gia' per assert. Tokenizzando la stessa
# probe, l'accordo e' 1,000 esatto.
#
# Proprio per questo e' un buon gate: essendo atteso al 100%, QUALUNQUE valore diverso e' un
# bug, non un risultato. Il riferimento per l'ordine di grandezza e' `federated_cb_only`
# (encoder interamente locale), dove l'accordo misurato e' 0,001 -- cioe' il caso.
#
#   bash scripts/zn_a1_gate.sh            # aspetta la prima cella A1 e la verifica
#   bash scripts/zn_a1_gate.sh ucr_001    # verifica una serie precisa (se gia' pronta)
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
PY=/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10

TAG=zn_a1; DS=ucr_split_w2p; SEED=0
# ⚠ IL NOME DELLA CARTELLA PORTA I KNOB. La pipeline scrive i checkpoint sotto
# `federated_enc_fedavg_bn-shared` (e A2 sotto `..._bn-shared_prior-partial`): e' cosi' che
# due configurazioni dello stesso arm non collidono su disco. `latent_probe.py` vuole quel
# nome, non quello nudo, altrimenti trova "0/5 client" e salta senza scrivere niente.
# DUE nomi, non uno: l'out-json usa l'arm NUDO (`ucr_011__federated_enc_fedavg.json`),
# i checkpoint quello con i knob (`federated_enc_fedavg_bn-shared`). Confonderli fa fallire
# il gate in due modi opposti a seconda di quale si usa dove.
ARM_JSON=federated_enc_fedavg
ARM="$(basename "$(ls -d artifacts/runs/$TAG/ckpt/$DS/*/seed0/${ARM_JSON}* 2>/dev/null | head -1)")"
ARM="${ARM:-$ARM_JSON}"
WANT="${1:-}"
OUT="$REPO/evidence/a1_gate.txt"
mkdir -p "$REPO/evidence"
say() { echo "[$(TZ=Europe/Madrid date '+%F %H:%M') Madrid][gate-A1] $*" | tee -a "$OUT"; }

# ── 1. la prima cella A1 finita ──────────────────────────────────────────────────
pick() {
  if [ -n "$WANT" ]; then
    [ -f "artifacts/runs/$TAG/$DS/${WANT}__${ARM_JSON}.json" ] && echo "$WANT"; return
  fi
  ls artifacts/runs/$TAG/$DS/*__${ARM_JSON}.json 2>/dev/null | head -1 \
    | sed "s|.*/||; s|__${ARM_JSON}.json||"
}
SER="$(pick)"
if [ -z "$SER" ]; then
  say "aspetto la prima cella A1 (controllo ogni 5 min)"
  while [ -z "$SER" ]; do sleep 300; SER="$(pick)"; done
fi
say "cella scelta: $SER  ($TAG/$ARM/seed$SEED)"

# ── 2. la sonda, solo D0+D1 ──────────────────────────────────────────────────────
# D0 prima di D1 e non per completezza: se le 5 finestre di input NON fossero identiche,
# un disaccordo di token non sarebbe attribuibile all'encoder e il gate misurerebbe
# un'altra cosa. D2/D3 non servono qui -- sono la VALIDAZIONE SCIENTIFICA del fix
# (il tokenizer straniero smette di distruggere il punteggio?), non la sua ammissibilita'.
say "sonda D0+D1 in corso (CPU, qualche minuto)..."
"$PY" scripts/latent_probe.py --run-dir "artifacts/runs/$TAG" --dataset "$DS" \
  --cluster "$SER" --arm "$ARM" --seed "$SEED" --only d0,d1 --device cpu \
  >> "$OUT" 2>&1
# ⚠ Il nome del file lo decide la sonda e ci mette il RUN-DIR quando non e' quello di
# default: `latent_probe_zn_a1__ucr_split_w2p_...`. Ricostruirlo a mano fallisce; si cerca.
PROBE="$(ls -t evidence/latent_probe*${SER}_${ARM}_seed${SEED}.json 2>/dev/null | head -1)"
[ -f "$PROBE" ] || { say "!! la sonda non ha scritto $PROBE — gate NON eseguito"; exit 2; }

# ── 3. il verdetto ───────────────────────────────────────────────────────────────
"$PY" - "$PROBE" <<'PYEOF' | tee -a "$OUT"
import json, sys
d = json.load(open(sys.argv[1]))
d0, d1 = d["D0"], d["D1"]
pairs = d1["pairs"]
worst = min(v["agree_raw"] for v in pairs.values())
mean = d1["mean_agree_raw"]
cb = d1["codebook_max_delta"]

print(f"\n{'='*70}\nGATE A1 — accordo di token fra client")
print(f"  input identici (D0)      : {d0['inputs_identical']}  (max |Δ| = {d0['worst']:.2e})")
print(f"  codebook identico        : max |Δ| = {cb:.3e}")
print(f"  accordo grezzo medio     : {mean:.6f}   su {len(pairs)} coppie")
print(f"  coppia PEGGIORE          : {worst:.6f}")
for k, v in sorted(pairs.items(), key=lambda x: x[1]["agree_raw"])[:3]:
    print(f"      {k:<32} {v['agree_raw']:.6f}")

ok = d0["inputs_identical"] and worst >= 0.9999 and cb < 1e-6
print()
if ok:
    print("ESITO: ✅ PASSA — i 5 encoder tokenizzano in modo identico.")
    print("       Le celle del ramo A1 sono valide. Il confronto che conta ora e'")
    print("       `zn_a1 - zn_enc` sullo stesso arm: isola il SOLO regime BN.")
else:
    print("ESITO: ❌ NON PASSA — le celle A1 NON sono valide.")
    if not d0["inputs_identical"]:
        print("       causa a monte: le finestre di input non sono identiche fra client,")
        print("       quindi il disaccordo NON e' attribuibile all'encoder.")
    elif cb >= 1e-6:
        print(f"       il codebook differisce fra client (max |Δ| = {cb:.3e}): il merge")
        print("       suff-stat o il broadcast non hanno fatto il loro lavoro.")
    else:
        print(f"       gli encoder differiscono nonostante --fed-enc-bn shared. Atteso 1,000")
        print(f"       per costruzione (media dei pesi + pooling delle statistiche a fine")
        print(f"       round), misurato {worst:.4f}. E' un BUG, non un risultato.")
sys.exit(0 if ok else 1)
PYEOF
rc=${PIPESTATUS[0]}
say "referto in $OUT  (rc=$rc)"
exit "$rc"
