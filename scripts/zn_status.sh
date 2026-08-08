#!/usr/bin/env bash
# Stato della campagna z-norm in una schermata. Nessun effetto collaterale.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
PY="${PY:-/home/leonardo/PhD/TimeVQVAE-AD-M/.venv/bin/python3.10}"
G4=${G4:-leonardo@g4.etsisi.upm.es}

echo "===== $(TZ=Europe/Madrid date '+%H:%M Madrid') · $(date -u '+%H:%M UTC') ====="

"$PY" - <<'PYEOF'
import glob, os, time, collections, datetime, re
import statistics as st
import zoneinfo
MAD = zoneinfo.ZoneInfo("Europe/Madrid")
def mad(t):
    return datetime.datetime.fromtimestamp(t, MAD).strftime("%a %H:%M")

# Peso relativo di ogni arm, dalle mediane START->DONE misurate su 60 log dell'archivio
# (centralized = 1.0). Senza pesi l'ETA estrapola dalle celle piu' economiche e sottostima
# di ~4x: `centralized` e' l'arm meno caro di tutti.
W = {"centralized": 1.00, "local": 3.28, "federated": 4.20, "federated_cb_only": 4.50,
     "federated_cb_only_ema": 5.29, "federated_fedavg_cb_only": 3.90,
     "federated_enc_fedavg": 3.61, "federated_enc_fedprox": 4.22,
     "federated_enc_fedproto": 4.22, "federated_enc_commoninit": 4.12,
     # Ramo N, aggiunto il 2026-08-05. Nessuna misura propria ancora: il peso e' quello del
     # PARTNER con cui sara' confrontato (lo stesso arm meno la rianimazione dei codici).
     # Il ramo A1 non compare qui: gira `federated_enc_fedavg`, che un peso ce l'ha gia'.
     "federated_cb_only_ema_norevive": 5.29}
TAGS = {"zn_main": ["centralized", "local", "federated", "federated_cb_only",
                    "federated_cb_only_ema", "federated_fedavg_cb_only"],
        "zn_enc":  ["federated_enc_fedavg", "federated_enc_fedprox",
                    "federated_enc_fedproto", "federated_enc_commoninit"],
        # Un tag per ramo, non arm nuovi dentro i tag esistenti: `zn_a1` gira lo STESSO
        # arm di `zn_enc` (federated_enc_fedavg) cambiando solo --fed-enc-bn, quindi
        # nello stesso tag i due si sovrascriverebbero lo stesso out-json. E il
        # cohort_fingerprint non copre `fed_enc_bn` (vedi CLAUDE.md), percio' il tag
        # distinto e' l'unica cosa che tiene separate due configurazioni diverse.
        "zn_norev": ["federated_cb_only_ema_norevive"],   # solo 6 serie: vedi SER_NOREV
        "zn_a1":    ["federated_enc_fedavg"]}
SER = ["ucr_001", "ucr_011", "ucr_014", "ucr_043", "ucr_082",
       "ucr_083", "ucr_086", "ucr_170", "ucr_222", "ucr_229"]
G2 = {"ucr_001", "ucr_011"}
# Slot = PROCESSI VIVI, contati adesso. Non una costante: la capacita' cambia in corsa (il
# 2026-08-04 alle 23:07 le tre Ada di g4 hanno portato la campagna da 15 a 24 slot, e alle
# 07:00 UTC il coprifuoco la riporta a 15). Con la costante cablata l'ETA restava indietro
# di un terzo senza che niente lo segnalasse.
import subprocess
live = set()
for cmd in ("ps -eo args",
            "ssh -o BatchMode=yes -o ConnectTimeout=10 "
            + os.environ.get("G4", "leonardo@g4.etsisi.upm.es") + " 'ps -eo args'"):
    try:
        out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout
    except Exception:
        continue
    for m in re.finditer(r"--cluster (\S+) --arms (\S+)", out):
        live.add((m.group(1), m.group(2)))
SLOTS = max(len(live), 1)

# Il mtime del log dell'orchestratore NON va bene: viene APPESO, quindi e' sempre "adesso".
# Serve il timestamp della prima riga scritta.
ts = []
for p in glob.glob("logs/runs/zn_*/_orchestrator.log"):
    with open(p, errors="ignore") as f:
        m = re.match(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\]", f.readline())
        if m:
            ts.append(datetime.datetime.fromisoformat(m.group(1)).timestamp())
el = (time.time() - min(ts, default=time.time())) / 3600

done_n = done_w = tot_n = tot_w = 0
for tag, arms in TAGS.items():
    cells = {os.path.basename(p)[:-5] for p in glob.glob(f"artifacts/runs/{tag}/*/*__*.json")}
    tn, tw = len(SER) * len(arms), len(SER) * sum(W[a] for a in arms)
    dw = sum(W[c.split("__")[1]] for c in cells if c.split("__")[1] in W)
    g2 = sum(1 for c in cells if c.split("__")[0] in G2)
    done_n += len(cells); done_w += dw; tot_n += tn; tot_w += tw
    print(f"  {tag:<9} {len(cells):>3}/{tn:<3} celle   (g2 {g2}, g4 {len(cells)-g2})"
          f"   lavoro {dw/tw*100:4.1f}%")

print(f"\n  TOTALE   {done_n}/{tot_n} celle   ·   lavoro {done_w:.1f}/{tot_w:.1f} unita "
      f"({done_w/tot_w*100:.1f}%)   ·   trascorse {el:.1f} h")

# ETA per CAPACITA', non per completamenti. Contare solo le celle chiuse ignora che 15 job
# sono gia' a ore di lavoro senza aver chiuso: a inizio campagna quella stima sbaglia di un
# ordine di grandezza (misurato: 267 h contro ~17 reali). Qui si stima il costo di UNA unita'
# di lavoro dalle celle chiuse, e si divide il lavoro che resta per il numero di slot.
dur = []
for lg in glob.glob("logs/runs/zn_*/_orchestrator.log"):
    txt = open(lg, errors="ignore").read()
    start = {m.group(2): datetime.datetime.fromisoformat(m.group(1))
             for m in re.finditer(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\](?:\[[\w.-]+\])? START (\S+) ", txt)}
    for m in re.finditer(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\](?:\[[\w.-]+\])? DONE\s+(\S+) .*?rc=0", txt):
        j = m.group(2)
        if j in start:
            arm = j.split("__")[-1]
            h = (datetime.datetime.fromisoformat(m.group(1)) - start[j]).total_seconds() / 3600
            if 0 < h < 24 and arm in W:
                dur.append(h / W[arm])          # ore per UNITA' di lavoro
if done_w >= tot_w:
    print("  COMPLETA")
elif dur:
    unit = st.median(dur)
    eta = (tot_w - done_w) * unit / SLOTS
    print(f"  costo unita  {unit*60:.0f} min  (da {len(dur)} celle chiuse, {SLOTS} slot)")
    print(f"  ETA          ~{eta:.0f} h  ->  {mad(time.time() + eta * 3600)} Madrid"
          f"   (pessimista: ignora il lavoro dei job in volo)")
else:
    print("  ETA          non ancora stimabile (nessuna cella chiusa)")

# In volo = c'e' un PROCESSO VIVO per quella (serie, arm). NON "il log e' stato scritto di
# recente": gli arm federati stampano una riga per ROUND, e il primo round di stage 1 su una
# serie a finestra lunga puo' tacere per mezz'ora (misurato: ucr_082 federated, W=892, 18 min
# di silenzio col processo al 102% di CPU). E il solo "log senza out-json" conta anche i job
# uccisi, i cui log restano su disco per sempre. Solo il processo dice la verita'.
run, dead = collections.Counter(), []
for f in glob.glob("logs/runs/zn_*/*__*.log"):
    b = os.path.basename(f)[:-4].split("__")
    if len(b) != 3:
        continue
    if os.path.exists(f"artifacts/runs/{f.split('/')[2]}/{b[0]}/{b[1]}__{b[2]}.json"):
        continue
    if (b[1], b[2]) in live:
        run[b[2]] += 1
    else:
        dead.append(f"{b[1]}/{b[2]}")
if run:
    print(f"\n  in volo ({sum(run.values())}): " + ", ".join(f"{k} {v}" for k, v in run.most_common()))
if dead:
    print(f"  senza processo ne' risultato ({len(dead)}): " + ", ".join(sorted(dead)[:6])
          + (" ..." if len(dead) > 6 else ""))

# ── CELLE ORFANE: nessun risultato, nessun processo, e NESSUNA CATENA che le abbia in coda.
# E' il controllo che mancava. Il 2026-08-04 lo screen `zn_g2` e' morto portandosi via il
# processo della catena; i job gia' dispatchati erano nohup e hanno continuato a chiudere
# celle regolarmente, quindi da fuori sembrava tutto a posto. Ma le 8 celle `zn_enc` di
# ucr_001/ucr_011 non erano piu' in coda a NESSUNO, e contando solo le celle chiuse non
# emergeva: sarebbero risultate mancanti a campagna "finita".
done = {tuple(os.path.basename(p)[:-5].split("__"))
        for p in glob.glob("artifacts/runs/zn_*/*/*__*.json")}
chains = []          # (nome, insieme di cluster che coprira')
for cmd, host in ((["ps", "-eo", "args"], "g2"),
                  (["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                    os.environ.get("G4", "leonardo@g4.etsisi.upm.es"),
                    "ps -eo args; echo ---; for p in $(pgrep -f 'bash scripts/zn_'); do "
                    "tr '\\0' '\\n' < /proc/$p/environ 2>/dev/null | grep ^LAUNCH_ONLY_CLUSTERS=; done"],
                   "g4")):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
    except Exception:
        continue
    if host == "g2":
        for p in glob.glob("/proc/*/environ"):
            try:
                env = open(p, "rb").read().decode("utf8", "replace")
                cl = open(p.replace("environ", "cmdline"), "rb").read().decode("utf8", "replace")
            except Exception:
                continue
            if "scripts/zn_" in cl and "bash" in cl:
                m = re.search(r"LAUNCH_ONLY_CLUSTERS=([^\x00\n]*)", env)
                chains.append((host + ":" + cl.split("scripts/")[-1].split("\x00")[0],
                               set(m.group(1).split(",")) if m else set(SER)))
    else:
        for m in re.finditer(r"LAUNCH_ONLY_CLUSTERS=(\S+)", out):
            chains.append((host + ":chain", set(m.group(1).split(","))))
        if re.search(r"bash scripts/zn_\w+\.sh", out) and not chains:
            chains.append((host + ":chain", set(SER)))

covered = set().union(*[c for _, c in chains]) if chains else set()
orphan = [(cl, a) for tag, arms in TAGS.items() for cl in SER for a in arms
          if (cl, a) not in done and (cl, a) not in live and cl not in covered]
if orphan:
    print(f"\n  🔴 {len(orphan)} CELLE ORFANE: nessun risultato, nessun processo, NESSUNA CATENA in coda")
    print("     " + ", ".join(f"{c}/{a}" for c, a in sorted(orphan)[:8])
          + (" ..." if len(orphan) > 8 else ""))
    print("     ^ non le fara' nessuno finche' non si rilancia una catena che le copra")
elif chains:
    print(f"\n  catene vive: {len(chains)} -> coprono {len(covered)} serie, nessuna cella orfana")
PYEOF

echo
echo "  g2  $(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader | tr '\n' ' ')"
timeout 25 ssh -o BatchMode=yes -o ConnectTimeout=10 "$G4" \
  'echo "  g4  $(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader | tr "\n" " ") load=$(cut -d" " -f1 /proc/loadavg)"' 2>/dev/null \
  || echo "  g4  (non raggiungibile)"

fail=$(cat logs/runs/zn_*/_orchestrator.log 2>/dev/null | grep -ac "FAILED" || true)
[ "${fail:-0}" -gt 0 ] 2>/dev/null && echo "  !! $fail job FALLITI -- vedi logs/runs/zn_*/_orchestrator.log"
exit 0
