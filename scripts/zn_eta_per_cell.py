"""ETA per OGNI cella ancora da fare, simulando le code delle tre catene.

Non basta dividere il lavoro residuo per gli slot: le catene sono indipendenti, hanno serie
disgiunte e cadenze diverse, e quella sulle Ada muore alle 07:00 UTC lasciando indietro
delle celle che il raccoglitore su g2 riprende dopo. Una media non descrive nulla di tutto
questo, quindi qui si simula la coda evento per evento.

Ordine di dispatch: `launch.sh` scorre le JOBLINES di `cohort.py jobs`, che sono
cluster-major e arm-minor nell'ordine passato a --arms. Prima tutto `zn_main`, poi `zn_enc`.
"""
import glob
import json
import os
import re
import subprocess
import time
import datetime
import statistics as st
import zoneinfo
MAD = zoneinfo.ZoneInfo("Europe/Madrid")
def mad(t):
    return datetime.datetime.fromtimestamp(t, MAD).strftime("%a %H:%M")

REPO = "/home/leonardo/PhD/TimeVQVAE-AD-U-Federated"
os.chdir(REPO)

W = {"centralized": 1.00, "local": 3.28, "federated": 4.20, "federated_cb_only": 4.50,
     "federated_cb_only_ema": 5.29, "federated_fedavg_cb_only": 3.90,
     "federated_enc_fedavg": 3.61, "federated_enc_fedprox": 4.22,
     "federated_enc_fedproto": 4.22, "federated_enc_commoninit": 4.12}
W["federated_cb_only_ema_norevive"] = 5.29     # ramo N: = cb_only_ema meno la rianimazione
MAIN = ["centralized", "local", "federated", "federated_cb_only",
        "federated_cb_only_ema", "federated_fedavg_cb_only"]
ENC = ["federated_enc_fedavg", "federated_enc_fedprox",
       "federated_enc_fedproto", "federated_enc_commoninit"]
# ⚠ La cella si identifica con (TAG, serie, arm), mai con (serie, arm). Il ramo A1 gira lo
# STESSO arm di `zn_enc` -- `federated_enc_fedavg` -- cambiando solo `--fed-enc-bn`: con la
# chiave a due campi, la cella `zn_enc` chiusa farebbe risultare chiusa anche quella `zn_a1`,
# e il simulatore toglierebbe dalla coda lavoro che nessuno ha fatto.
NOREV = ["federated_cb_only_ema_norevive"]
A1 = ["federated_enc_fedavg"]

CHAINS = {
    "g2":      {"ser": ["ucr_001", "ucr_011"], "slots": 6, "stop": None},
    "g4-3090": {"ser": ["ucr_014", "ucr_043", "ucr_082", "ucr_083", "ucr_086"],
                "slots": 9, "stop": None},
    "g4-Ada":  {"ser": ["ucr_170", "ucr_222", "ucr_229"], "slots": 9, "stop": "07:00"},
}
# I due rami aggiunti il 2026-08-05: partono quando i dispatcher del loro host hanno finito
# (scripts/zn_branches.sh aspetta), e l'ordine e' quello scritto li' -- g2 fa A1 per primo
# per avere presto la risposta del gate, g4 fa il ramo di priorita' 1.
# ⚠ La lista di serie e' PER RAMO, non per host: i due rami hanno partner diversi, e su
# ucr_222/229 il partner di N (`cb_only_ema`) e' girato su g4-3090 mentre quello di A1
# (`enc_fedavg`) e' girato su g2. Il ramo N gira solo sulle 6 serie dove la rianimazione
# puo' scattare: sulle altre 4 `revive=False` da' una run bit-identica (federated.py:556,
# `if revive and n_dead > 0:`), quindi sono pareggi per costruzione e non si lanciano.
BRANCHES = {
    "g2-rami":      {"slots": 6, "after": ("g2", "g2-pickup"),
                     "tags": [("zn_a1", A1, ["ucr_001", "ucr_011", "ucr_222", "ucr_229"]),
                              ("zn_norev", NOREV, ["ucr_001", "ucr_011"])]},
    "g4-3090-rami": {"slots": 9, "after": ("g4-3090", "g4-3090-pickup"),
                     "tags": [("zn_norev", NOREV, ["ucr_014", "ucr_043", "ucr_222", "ucr_229"]),
                              ("zn_a1", A1, ["ucr_014", "ucr_043", "ucr_082",
                                             "ucr_083", "ucr_086", "ucr_170"])]},
}

now = time.time()

# ── costo di UNA unita' di lavoro, misurato dalle celle gia' chiuse ──────────────
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
                dur.append(h / W[arm])
UNIT = st.median(dur) if dur else 0.8

# ── stato: chiuse, e in volo da quanto ──────────────────────────────────────────
done = {(p.split("/")[2], *os.path.basename(p)[:-5].split("__"))
        for p in glob.glob("artifacts/runs/zn_*/*/*__*.json")}

live = {}
for cmd in ("ps -eo args",
            "ssh -o BatchMode=yes -o ConnectTimeout=10 leonardo@g4.etsisi.upm.es 'ps -eo args'"):
    try:
        out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout
    except Exception:
        continue
    # Il tag si legge da --out-json, che e' l'unico posto della riga di comando dove compare:
    # `--arms federated_enc_fedavg` da solo non dice se e' la cella di `zn_enc` o di `zn_a1`.
    for m in re.finditer(r"--cluster (\S+) --arms (\S+).*?/runs/([A-Za-z0-9_]+)/", out):
        live[(m.group(3), m.group(1), m.group(2))] = True

# da quanto gira ciascuna cella viva: primo START senza DONE nel log dell'orchestratore
started_at = {}
for lg in glob.glob("logs/runs/zn_*/_orchestrator.log"):
    tag = lg.split("/")[2]
    txt = open(lg, errors="ignore").read()
    for m in re.finditer(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\](?:\[[\w.-]+\])? START (\S+) ", txt):
        _, cl, arm = m.group(2).split("__")
        if (tag, cl, arm) in live:
            started_at[(tag, cl, arm)] = datetime.datetime.fromisoformat(m.group(1)).timestamp()

# ── istante del coprifuoco Ada ──────────────────────────────────────────────────
def next_utc(hhmm):
    h, m = hhmm.split(":")
    t = datetime.datetime.now(datetime.timezone.utc).replace(
        hour=int(h), minute=int(m), second=0, microsecond=0).timestamp()
    return t if t > now else t + 86400


# ── simulazione a eventi, catena per catena ─────────────────────────────────────
def simulate(queue, slots, stop_at, t0=None):
    """queue = [(tag, serie, arm)]. Restituisce [(tag, serie, arm, istante_fine)] e le
    celle che NON entrano prima di `stop_at`."""
    t0 = now if t0 is None else t0
    free = [t0] * slots             # istante in cui ogni slot torna libero
    out, dropped = [], []
    for tag, cl, arm in queue:
        if (tag, cl, arm) in done:
            continue
        cost = W[arm] * UNIT * 3600
        i = min(range(slots), key=lambda k: free[k])
        if (tag, cl, arm) in live:  # gia' in corso: resta solo il residuo
            el = now - started_at.get((tag, cl, arm), now)
            fin = free[i] = max(free[i], now) + max(cost - el, 300)
        else:
            fin = free[i] = max(free[i], t0) + cost
        if stop_at and fin > stop_at:
            dropped.append((tag, cl, arm))
        else:
            out.append((tag, cl, arm, fin))
    return out, dropped


# ⚠ `next_utc` restituisce l'occorrenza SUCCESSIVA: passate le 07:00 UTC di oggi, dà quelle
# di domani, e la simulazione ha creduto per ore che le Ada fossero ancora disponibili per
# altre 16 celle. Non lo sono: sono state restituite al dipartimento e `zn_ada_guard.sh` le
# tiene libere. Se il coprifuoco è già passato, la catena Ada è MORTA e tutta la sua coda
# finisce nello spill, che è dove finisce davvero.
_h, _m = CHAINS["g4-Ada"]["stop"].split(":")
_today = datetime.datetime.now(datetime.timezone.utc).replace(
    hour=int(_h), minute=int(_m), second=0, microsecond=0).timestamp()
ADA_STOP = now if now > _today else _today
ADA_DEAD = now > _today
rows, spill = [], []
for name, c in CHAINS.items():
    q = ([("zn_main", s, a) for s in c["ser"] for a in MAIN]
         + [("zn_enc", s, a) for s in c["ser"] for a in ENC])
    got, drop = simulate(q, c["slots"], ADA_STOP if c["stop"] else None)
    rows += [(name, *g) for g in got]
    spill += drop

# ── i raccoglitori dello spill Ada: DUE, non uno ────────────────────────────────
# `ucr_170` va su g4-3090 (`zn_g4_pickup.sh`), `ucr_222`/`ucr_229` su g2 (`zn_g2_finish.sh`).
# Mandare tutto lo spill su g2, come faceva questa simulazione, caricava g2 di 4 celle che
# non fara' mai -- ~2 h di ETA in piu' su g2 e ~2 h in meno su g4, sempre nello stesso verso.
PICKUP = {"g4-3090-pickup": (["ucr_170"], "g4-3090", 9),
          "g2-pickup":      (["ucr_222", "ucr_229"], "g2", CHAINS["g2"]["slots"])}
for pname, (pser, host, pslots) in PICKUP.items():
    part = [c for c in spill if c[1] in pser]
    if not part:
        continue
    host_end = max([r[4] for r in rows if r[0] == host], default=now)
    got, _ = simulate(part, pslots, None, t0=max(host_end, ADA_STOP + 300))
    rows += [(pname, *g) for g in got]

# ── i due rami: partono quando i dispatcher del loro host hanno finito ───────────
# `zn_branches.sh` aspetta esplicitamente che non ci sia piu' nessun `launch.sh` sull'host,
# quindi il loro t0 e' la fine dell'ULTIMA catena di quell'host, raccoglitori compresi.
for name, b in BRANCHES.items():
    t0 = max([r[4] for r in rows if r[0] in b["after"]], default=now)
    q = [(tag, s, a) for tag, arms, ser in b["tags"] for s in ser for a in arms]
    got, _ = simulate(q, b["slots"], None, t0=max(t0, now))
    rows += [(name, *g) for g in got]

# ── stampa ──────────────────────────────────────────────────────────────────────
# ⚠ Gli orari sono di MADRID, convertiti esplicitamente. Il server gira in UTC, quindi
# `time.localtime()` restituisce UTC: usarlo qui ha prodotto per tutta la notte del
# 2026-08-04 ETA sistematicamente in anticipo di due ore, con l'intestazione che diceva
# "Madrid" perche' quella la calcolava la shell con TZ=Europe/Madrid. Non usare localtime.
print(f"costo di 1 unita = {UNIT*60:.0f} min (da {len(dur)} celle chiuse)")
print(f"coprifuoco Ada: {mad(ADA_STOP)}   ·   adesso: {mad(now)}\n")
rows.sort(key=lambda r: r[4])
print(f"{'#':>3} {'catena':<15}{'tag':<9}{'serie':<9}{'arm':<26}{'stato':<9}{'finisce':<12}{'fra'}")
for i, (ch, tag, cl, arm, t) in enumerate(rows, 1):
    stato = "in corso" if (tag, cl, arm) in live else "in coda"
    print(f"{i:>3} {ch:<15}{tag:<9}{cl:<9}{arm:<26}{stato:<9}{mad(t):<12}{(t-now)/3600:5.1f} h")

TOT = 116      # 10 serie x 11 celle + 6 celle del ramo N (4 fuori disegno)
print(f"\n{len(rows)} celle da fare · {len(done)} gia' chiuse · {len(rows)+len(done)}/{TOT}")
if rows:
    end = max(r[4] for r in rows)
    print(f"ULTIMA CELLA: {mad(end)}  (fra {(end-now)/3600:.1f} h)")
    for ch in ("g2", "g4-3090", "g4-Ada", "g2-pickup", "g4-3090-pickup", "g2-rami", "g4-3090-rami"):
        rs = [r for r in rows if r[0] == ch]
        if rs:
            print(f"   {ch:<14} finisce {mad(max(r[4] for r in rs))}  ({len(rs)} celle)")
if spill:
    print(f"\n{len(spill)} celle NON entrano prima del coprifuoco Ada -> raccolte cosi':")
    for pname, (pser, _, _) in PICKUP.items():
        part = [f"{c}/{a}" for _, c, a in spill if c in pser]
        if part:
            print(f"   {pname:<15} {', '.join(part)}")
