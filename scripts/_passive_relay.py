#!/usr/bin/env python3
"""Relay bidirezionale fra `ssh -> sshfs -o passive` (su g4) e `sftp-server` (qui su g2).

PERCHE' ESISTE. La prima versione del ponte era una pipeline di shell:

    ssh ... < "$FIFO" | "$SFTP" > "$FIFO"

cioe' ssh(stdout) -> pipe -> sftp(stdin) e sftp(stdout) -> FIFO -> ssh(stdin): un CICLO
CHIUSO con due buffer finiti da 64 KB. Se entrambi si riempiono, ssh e' bloccato in
`pipe_write` verso sftp e sftp e' bloccato in write verso il FIFO: attesa circolare, nessuno
puo' drenare l'altro. Deadlock permanente, non transitorio.

Successo il 2026-08-01 alle 15:54 sotto 11 job concorrenti: tutti i processi su g4 congelati
in stato D, 2 MB fermi nella Recv-Q del socket, 1h20m di wall clock e ~5 h di training in volo
persi. Allargare i buffer avrebbe solo spostato la soglia.

LA CORREZIONE. Il ciclo si rompe se un LETTORE non e' mai bloccato dal SUO scrittore. Qui ogni
direzione ha due thread indipendenti separati da una coda in memoria: il lettore drena sempre
il proprio fd, anche quando il peer e' fermo. Con questo, il riempirsi di un buffer non puo'
piu' propagarsi all'indietro fino a fermare la direzione opposta.

La coda e' limitata (`--max-buffer`) per non far crescere la RAM senza fine: raggiunto il
tetto il relay MUORE con un messaggio esplicito invece di piantarsi in silenzio. Un mount
morto e rumoroso si ripara; uno vivo e bloccato costa una giornata.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import threading
from collections import deque

CHUNK = 1 << 16


def _log(msg: str) -> None:
    print(f"[relay] {msg}", file=sys.stderr, flush=True)


class Direction:
    """Una direzione del relay: src.read() -> coda -> dst.write().

    Il lettore e lo scrittore sono thread distinti proprio perche' il lettore deve poter
    drenare `src` mentre `dst` e' fermo -- e' l'unica proprieta' che impedisce il deadlock.
    """

    def __init__(self, name: str, src: int, dst: int, max_buffer: int, die: threading.Event):
        self.name, self.src, self.dst = name, src, dst
        self.max_buffer, self.die = max_buffer, die
        self.q: deque[bytes] = deque()
        self.size = 0
        self.eof = False
        self.cv = threading.Condition()

    def reader(self) -> None:
        try:
            while not self.die.is_set():
                buf = os.read(self.src, CHUNK)
                if not buf:
                    break
                with self.cv:
                    if self.size + len(buf) > self.max_buffer:
                        _log(
                            f"{self.name}: coda oltre {self.max_buffer} byte -- il peer non "
                            f"consuma. Chiudo il mount invece di bloccarmi."
                        )
                        self.die.set()
                        self.cv.notify_all()
                        return
                    self.q.append(buf)
                    self.size += len(buf)
                    self.cv.notify_all()
        except OSError as e:
            _log(f"{self.name}: lettura interrotta ({e})")
        finally:
            with self.cv:
                self.eof = True
                self.cv.notify_all()

    def writer(self) -> None:
        try:
            while True:
                with self.cv:
                    while not self.q and not self.eof and not self.die.is_set():
                        self.cv.wait()
                    if self.die.is_set() or (not self.q and self.eof):
                        return
                    buf = self.q.popleft()
                    self.size -= len(buf)
                while buf:
                    buf = buf[os.write(self.dst, buf):]
        except OSError as e:
            _log(f"{self.name}: scrittura interrotta ({e})")
        finally:
            self.die.set()
            with self.cv:
                self.cv.notify_all()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--src", required=True, help="dir su g2 da esportare")
    ap.add_argument("--mnt", required=True, help="mountpoint su g4")
    ap.add_argument("--sftp", default="/usr/lib/openssh/sftp-server")
    ap.add_argument("--max-buffer", type=int, default=512 << 20)
    a = ap.parse_args()

    remote = f"mkdir -p {shlex.quote(a.mnt)} && exec sshfs -f -o passive :{shlex.quote(a.src)} {shlex.quote(a.mnt)}"
    ssh = subprocess.Popen(
        ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=15",
         "-o", "ServerAliveCountMax=3", a.host, remote],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0,
    )
    sftp = subprocess.Popen([a.sftp], stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)
    _log(f"g2:{a.src} -> {a.host}:{a.mnt}  (ssh pid {ssh.pid}, sftp pid {sftp.pid})")

    die = threading.Event()
    dirs = [
        Direction("g4->sftp", ssh.stdout.fileno(), sftp.stdin.fileno(), a.max_buffer, die),
        Direction("sftp->g4", sftp.stdout.fileno(), ssh.stdin.fileno(), a.max_buffer, die),
    ]
    threads = []
    for d in dirs:
        for fn in (d.reader, d.writer):
            t = threading.Thread(target=fn, name=f"{d.name}:{fn.__name__}", daemon=True)
            t.start()
            threads.append(t)

    try:
        die.wait()
    except KeyboardInterrupt:
        pass
    _log("chiusura")
    for p in (ssh, sftp):
        p.kill()
        p.wait()
    return 1


if __name__ == "__main__":
    sys.exit(main())
