"""SENTINELLA ANTI-DOPPIO-DISPATCH su g2 e g4 insieme.

⚠ IL RISCHIO CHE CHIUDE. `launch.sh` salta una cella solo se l'out-json esiste GIA' al
momento del dispatch. Due dispatcher su HOST DIVERSI non si vedono e non si aspettano: se
prendono la stessa (tag, serie, arm) scrivono nella STESSA cartella di checkpoint —
`artifacts/runs/<tag>/ckpt/<ds>/<serie>/seed0/<arm>/` — sul filesystem condiviso via sshfs.
Il risultato non e' un crash: e' un checkpoint misto fra due run, cioe' un numero sbagliato
senza nessun errore. E' il modo caratteristico in cui questo repo si rompe.

Casi aperti al 2026-08-05 18:50, tutti cross-host e NON serializzati:
  zn_enc/ucr_222/{fedproto,fedprox}                  g2 in corso · zn_chain.sh g4 in coda
  zn_main/ucr_222/{federated,cb_only,cb_only_ema}    g4 in corso · zn_g2_finish in coda
Le contese su ucr_170 sono invece innocue: `zn_g4_after_chain.sh` aspetta esplicitamente che
`zn_chain.sh g4` esca, quindi sono serializzate per costruzione.

REGOLA: due processi vivi sulla stessa cella ⇒ si uccide il PIU' GIOVANE (ha meno lavoro
dentro). Il checkpoint parziale NON si cancella da soli: l'altro processo ci sta ancora
scrivendo, e una rm in quel momento e' peggio del problema. Si logga e si segnala.

⚠ I worker del DataLoader ereditano la riga di comando del padre: senza filtrare per
parentela ogni cella sembrerebbe duplicata una quarantina di volte. Qui il padre e' il
processo la cui `--out-json` compare con il PPID piu' basso del gruppo, identificato come
il processo il cui PPID non e' a sua volta un federated_eval.
"""
import os
import re
import subprocess
import sys
import time
import datetime

REPO = "/home/leonardo/PhD/TimeVQVAE-AD-U-Federated"
os.chdir(REPO)
G4 = os.environ.get("G4", "leonardo@g4.etsisi.upm.es")
EVERY = int(sys.argv[1]) if len(sys.argv) > 1 else 45
LOG = f"{REPO}/evidence/dupe_guard_{datetime.datetime.utcnow():%Y%m%d}.log"
MAD = None
try:
    import zoneinfo
    MAD = zoneinfo.ZoneInfo("Europe/Madrid")
except Exception:
    pass


def say(msg):
    t = datetime.datetime.now(MAD).strftime("%F %H:%M:%S") if MAD else time.strftime("%F %H:%M:%S")
    line = f"[{t} Madrid][dupe] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def run(cmd, remote=False, t=30):
    if remote:
        cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", G4, cmd]
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=t).stdout
        except Exception:
            return ""
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=t).stdout
    except Exception:
        return ""


PS = "ps -eo pid,ppid,etimes,args --no-headers"
RX = re.compile(r"--cluster (\S+) --arms (\S+)")
RXT = re.compile(r"--out-json \S*?/runs/([A-Za-z0-9_]+)/")


def snapshot():
    """{(tag, serie, arm): [(host, pid, eta_secondi)]} — solo processi PADRE."""
    out = {}
    for host, remote in (("g2", False), ("g4", True)):
        txt = run(PS, remote)
        rows, pids = [], set()
        for line in txt.splitlines():
            if "federated_eval.py" not in line:
                continue
            p = line.split(None, 3)
            if len(p) < 4:
                continue
            pid, ppid, et, args = p[0], p[1], p[2], p[3]
            m, mt = RX.search(args), RXT.search(args)
            if not (m and mt):
                continue
            rows.append((pid, ppid, int(et), mt.group(1), m.group(1), m.group(2)))
            pids.add(pid)
        for pid, ppid, et, tag, cl, arm in rows:
            if ppid in pids:            # e' un worker del DataLoader, non il padre
                continue
            out.setdefault((tag, cl, arm), []).append((host, pid, et))
    return out


say(f"=== SENTINELLA ANTI-DOPPIO-DISPATCH attiva (ogni {EVERY}s · g2+g4) ===")
seen_clean = 0
while True:
    snap = snapshot()
    dupes = {k: v for k, v in snap.items() if len(v) > 1}
    if dupes:
        seen_clean = 0
        for (tag, cl, arm), procs in dupes.items():
            say(f"!! DOPPIO DISPATCH  {tag} / {cl} / {arm}")
            for h, p, e in sorted(procs, key=lambda x: -x[2]):
                say(f"     {h} pid{p} vivo da {e//60} min")
            h, p, e = min(procs, key=lambda x: x[2])         # il piu' giovane
            say(f"   uccido il piu' giovane: {h} pid{p} ({e//60} min di lavoro)")
            if h == "g2":
                run(f"kill {p}")
            else:
                run(f"kill {p}", remote=True)
            ck = f"artifacts/runs/{tag}/ckpt/ucr_split_w2p/{cl}/seed0/{arm}"
            out = f"artifacts/runs/{tag}/ucr_split_w2p/{cl}__{arm}.json"
            if os.path.isdir(ck) and not os.path.exists(out):
                say(f"   ⚠ checkpoint scritto da due processi: {ck}")
                say("     NON lo cancello (l'altro ci sta ancora scrivendo). Da verificare a mano.")
    else:
        seen_clean += 1
        if seen_clean % 40 == 1:
            say(f"ok — {len(snap)} celle vive, nessun doppione")
    time.sleep(EVERY)
