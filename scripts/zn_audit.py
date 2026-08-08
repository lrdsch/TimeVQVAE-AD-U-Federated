"""AUDIT della campagna: ogni cella del disegno, chi l'ha fatta, chi la sta facendo, chi la fara'.

Risponde a tre domande che un conteggio di celle non risponde:

  ORFANE    celle senza risultato, senza processo e senza NESSUN dispatcher che le abbia in
            coda. Il 2026-08-04 lo screen `zn_g2` e' morto portandosi via la catena: i job
            gia' dispatchati hanno continuato a chiudere celle, quindi da fuori sembrava
            tutto a posto, ma 8 celle non erano piu' in coda a nessuno.

  CONTESE   celle che DUE dispatcher hanno in coda. `launch.sh` salta una cella solo se
            l'out-json esiste GIA' al momento del dispatch: due dispatcher che partono
            vicini possono prendere la stessa cella e scrivere nella stessa cartella di
            checkpoint. E' l'unico modo in cui questa campagna puo' corrompere un risultato.

  CARICO    job per GPU. Su g4 oltre 3 per scheda il throughput CALA (0,93x da 9 a 15).

⚠ La cella e' (TAG, serie, arm). `zn_a1` e `zn_a2` girano lo STESSO arm di `zn_enc`
(`federated_enc_fedavg`) cambiando solo i flag, quindi con la chiave a due campi
risulterebbero fatte quando e' chiusa la cella di `zn_enc`.
"""
import glob
import os
import re
import subprocess

REPO = "/home/leonardo/PhD/TimeVQVAE-AD-U-Federated"
os.chdir(REPO)
G4 = "leonardo@g4.etsisi.upm.es"
SER = ["ucr_001", "ucr_011", "ucr_014", "ucr_043", "ucr_082",
       "ucr_083", "ucr_086", "ucr_170", "ucr_222", "ucr_229"]
MAIN = ["centralized", "local", "federated", "federated_cb_only",
        "federated_cb_only_ema", "federated_fedavg_cb_only"]
ENC = ["federated_enc_fedavg", "federated_enc_fedprox",
       "federated_enc_fedproto", "federated_enc_commoninit"]
# Ramo N solo dove la rianimazione puo' scattare (+ ucr_001 come controllo di determinismo):
# altrove `revive=False` da' una run bit-identica, cioe' un pareggio per costruzione.
NOREV_SER = ["ucr_001", "ucr_011", "ucr_014", "ucr_043", "ucr_222", "ucr_229"]

DESIGN = ([("zn_main", s, a) for s in SER for a in MAIN]
          + [("zn_enc", s, a) for s in SER for a in ENC]
          + [("zn_norev", s, "federated_cb_only_ema_norevive") for s in NOREV_SER]
          + [("zn_a1", s, "federated_enc_fedavg") for s in SER]
          + [("zn_a2", s, "federated_enc_fedavg") for s in SER])

# ── chi copre cosa: letto dagli script, non indovinato ───────────────────────────
# (nome dispatcher, host, tag, arms, serie). Un dispatcher "in attesa" copre lo stesso
# insieme di uno in esecuzione: la coda e' quella, parta ora o fra sei ore.
COVER = [
    ("zn_g2_finish.sh",          "g2", "zn_enc",   ENC,   ["ucr_001", "ucr_011", "ucr_222", "ucr_229"]),
    ("zn_g2_finish.sh",          "g2", "zn_main",  MAIN,  SER),
    ("zn_branches.sh g2",        "g2", "zn_a1",    ["federated_enc_fedavg"], ["ucr_222"]),
    ("zn_branches.sh g2",        "g2", "zn_norev", ["federated_cb_only_ema_norevive"], ["ucr_001", "ucr_011"]),
    ("zn_chain.sh g4",           "g4", "zn_main",  MAIN,  ["ucr_014", "ucr_043", "ucr_082", "ucr_083",
                                                           "ucr_086", "ucr_170", "ucr_222", "ucr_229"]),
    ("zn_chain.sh g4",           "g4", "zn_enc",   ENC,   ["ucr_014", "ucr_043", "ucr_082", "ucr_083",
                                                           "ucr_086", "ucr_170", "ucr_222", "ucr_229"]),
    ("zn_g4_fill.sh",            "g4", "zn_norev", ["federated_cb_only_ema_norevive"], ["ucr_229"]),
    ("zn_a1_rush.sh",            "g4", "zn_a1",    ["federated_enc_fedavg"], ["ucr_001", "ucr_011", "ucr_229"]),
    ("zn_a2.sh 0",               "g4", "zn_a2",    ["federated_enc_fedavg"], ["ucr_001", "ucr_011", "ucr_229",
                                                                              "ucr_014", "ucr_043"]),
    ("zn_a2.sh 5",               "g4", "zn_a2",    ["federated_enc_fedavg"], ["ucr_082", "ucr_083", "ucr_086",
                                                                              "ucr_222", "ucr_170"]),
    ("zn_g4_after_chain.sh a1",  "g4", "zn_a1",    ["federated_enc_fedavg"], ["ucr_014", "ucr_043", "ucr_082",
                                                                              "ucr_083", "ucr_086", "ucr_170"]),
    ("zn_g4_after_chain.sh gpu3", "g4", "zn_enc",  ENC,   ["ucr_170"]),
    ("zn_g4_after_chain.sh gpu3", "g4", "zn_main", MAIN,  ["ucr_170"]),
    ("zn_g4_after_chain.sh gpu3", "g4", "zn_norev", ["federated_cb_only_ema_norevive"],
     ["ucr_014", "ucr_043", "ucr_222", "ucr_229"]),
]


def sh(cmd, remote=False):
    c = f"ssh -o BatchMode=yes -o ConnectTimeout=10 {G4} '{cmd}'" if remote else cmd
    try:
        return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=40).stdout
    except Exception:
        return ""


done = {(p.split("/")[2], *os.path.basename(p)[:-5].split("__"))
        for p in glob.glob("artifacts/runs/zn_*/*/*__*.json")}

live, gpu_load = {}, {}
for host, rem in (("g2", False), ("g4", True)):
    out = sh("ps -eo args", rem)
    for m in re.finditer(r"--cluster (\S+) --arms (\S+).*?/runs/([A-Za-z0-9_]+)/", out):
        live[(m.group(3), m.group(1), m.group(2))] = host
    alive_scripts = set(re.findall(r"^bash (scripts/\S+\.sh)(?: (\S+))?", out, re.M))
    gpu_load[host] = [tuple(x.split(",")) for x in sh(
        "nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader", rem).strip().splitlines() if x]

# dispatcher VIVI (per dire quali coperture sono reali e quali sono morte)
alive = set()
for host, rem in (("g2", False), ("g4", True)):
    out = sh("ps -eo args", rem)
    for line in out.splitlines():
        m = re.match(r"bash (scripts/[\w.]+\.sh)\s*(\S+)?", line)
        if m:
            alive.add((os.path.basename(m.group(1)) + (f" {m.group(2)}" if m.group(2) else "")).strip())

def cover_alive(name):
    base = name.split()[0]
    return any(a.split()[0] == base and (len(name.split()) == 1 or name.split()[1] in a)
               for a in alive)

cov = {}
for name, host, tag, arms, sers in COVER:
    for s in sers:
        for a in arms:
            k = (tag, s, a)
            if k in [(t, ss, aa) for t, ss, aa in DESIGN]:
                cov.setdefault(k, []).append((name, host, cover_alive(name)))

print(f"{'='*96}\nDISEGNO: {len(DESIGN)} celle   ·   fatte {len(done & set(DESIGN))}   ·   "
      f"in corso {len([k for k in DESIGN if k in live])}   ·   "
      f"da fare {len([k for k in DESIGN if k not in done and k not in live])}\n{'='*96}")

# ── ORFANE ──────────────────────────────────────────────────────────────────────
orf = [k for k in DESIGN if k not in done and k not in live
       and not any(al for _, _, al in cov.get(k, []))]
print(f"\n⛔ ORFANE — nessun risultato, nessun processo, nessun dispatcher VIVO: {len(orf)}")
for t, s, a in sorted(orf):
    morti = [n for n, _, al in cov.get((t, s, a), []) if not al]
    print(f"   {t:<9}{s:<10}{a:<32}" + (f"(la copriva {morti[0]}, ora morto)" if morti else "NESSUNO l'ha mai avuta"))
if not orf:
    print("   nessuna ✅")

# ── CONTESE ─────────────────────────────────────────────────────────────────────
# Coppie SERIALIZZATE per costruzione: il secondo aspetta che il primo esca, quindi non
# possono mai dispatchare insieme. Vanno tolte, o il rumore nasconde le contese vere.
SERIAL = {("zn_g4_after_chain.sh a1", "zn_chain.sh g4"),
          ("zn_g4_after_chain.sh gpu3", "zn_chain.sh g4"),
          ("zn_g4_after_chain.sh gpu3", "zn_g4_fill.sh")}
def serializzati(v):
    n = sorted({x[0] for x in v if x[2]})
    return len(n) == 2 and (tuple(n) in SERIAL or tuple(reversed(n)) in SERIAL)

cont = [(k, v) for k, v in cov.items()
        if k not in done and len([1 for _, _, al in v if al]) > 1 and not serializzati(v)]
ser_ok = [(k, v) for k, v in cov.items()
          if k not in done and len([1 for _, _, al in v if al]) > 1 and serializzati(v)]
print(f"\n⚠ CONTESE — due o piu' dispatcher VIVI hanno la stessa cella in coda: {len(cont)}")
for (t, s, a), v in sorted(cont):
    chi = ", ".join(f"{n}@{h}" for n, h, al in v if al)
    stato = "IN CORSO" if (t, s, a) in live else "in coda"
    print(f"   {t:<9}{s:<10}{a:<28}{stato:<9} -> {chi}")
if not cont:
    print("   nessuna ✅")
if ser_ok:
    print(f"   (+{len(ser_ok)} coppie SERIALIZZATE per costruzione, innocue: il secondo "
          f"dispatcher aspetta che il primo esca)")

# ── CARICO ──────────────────────────────────────────────────────────────────────
print("\n📊 CARICO per GPU (max misurato: 3 job/scheda su g4)")
for host in ("g2", "g4"):
    uu = sh("nvidia-smi --query-gpu=index,uuid --format=csv,noheader", host == "g4")
    idx = {u.strip(): i.strip() for i, u in (l.split(", ") for l in uu.strip().splitlines() if l)}
    cnt = {}
    for u, _p in gpu_load.get(host, []):
        cnt[idx.get(u.strip(), "?")] = cnt.get(idx.get(u.strip(), "?"), 0) + 1
    lim = 3 if host == "g4" else 7      # g4 senza MPS satura a 3; g2 con MPS regge 6-7
    s = "  ".join(f"GPU{g}:{n}" + ("⚠" if n > lim else "") for g, n in sorted(cnt.items()))
    print(f"   {host}: {s or 'nessun job'}")

# ── COSA MANCA, per tag ─────────────────────────────────────────────────────────
print("\n📋 DA FARE, per tag  (in corso / in coda)")
for tag in ("zn_main", "zn_enc", "zn_norev", "zn_a1", "zn_a2"):
    ks = [k for k in DESIGN if k[0] == tag]
    d = [k for k in ks if k in done]
    l = [k for k in ks if k in live]
    q = [k for k in ks if k not in done and k not in live]
    who = sorted({n for k in q for n, _, al in cov.get(k, []) if al})
    print(f"   {tag:<9} {len(d):>2}/{len(ks):<3} fatte · {len(l)} in corso · {len(q)} in coda"
          + (f"  -> {', '.join(who)}" if who else ("" if not q else "  -> ⛔ NESSUNO")))
