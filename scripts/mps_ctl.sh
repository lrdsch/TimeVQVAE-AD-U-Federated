#!/usr/bin/env bash
# mps_ctl.sh — bring the CUDA MPS control daemon up/down on a PRIVATE pipe directory.
#
# WHY NOT THE DEFAULT LOCATION: with CUDA_MPS_PIPE_DIRECTORY unset the daemon listens on
# /tmp/nvidia-mps, which it creates 0777 with a 0666 `control` socket. On this shared box
# that has two consequences, both measured on 2026-07-28:
#   (a) every CUDA process of EVERY user auto-attaches to OUR daemon -- /tmp/nvidia-mps is
#       the compiled-in default, so nobody has to opt in;
#   (b) any user can send it control commands, `quit_server` included. User `ssanchez` did
#       have a nvidia-cuda-mps-control open against our socket at 06:59.
# A private 0700 directory closes both. Other users then get their own default daemon and
# stop sharing our server process.
#
# WHY THE LOG DIRECTORY MATTERS: when the server died at 2026-07-28 00:02:24 it left no
# trace anywhere. CUDA_MPS_LOG_DIRECTORY was unset and /var/log/nvidia-mps did not exist,
# so the post-mortem had nothing to read and the trigger was never identified. With this
# script the server log outlives the crash.
#
# The 14 jobs that were attached to that server did NOT crash -- they blocked forever on a
# socket poll with no error. Stopping/restarting MPS does NOT unblock an already-wedged
# client: those processes must be killed. See scripts/gpu_stall_watchdog.sh.
#
#   bash scripts/mps_ctl.sh up | down | status | restart
#
# Any runner that wants MPS must export the SAME two variables; sourcing this file with
# MPS_CTL_ENV_ONLY=1 sets them without touching the daemon.
set -uo pipefail

MPS_DIR="${CUDA_MPS_PIPE_DIRECTORY:-$HOME/.mps}"
MPS_LOG="${CUDA_MPS_LOG_DIRECTORY:-$MPS_DIR/log}"
export CUDA_MPS_PIPE_DIRECTORY="$MPS_DIR"
export CUDA_MPS_LOG_DIRECTORY="$MPS_LOG"

# Sourced for the environment only (runners do this); do not run a command.
[[ -n "${MPS_CTL_ENV_ONLY:-}" ]] && return 0 2>/dev/null

stamp() { date +'%F %T'; }
say() { echo "[$(stamp)] mps_ctl: $*"; }

LEGACY_DIR=/tmp/nvidia-mps

# Every daemon/server this user owns, whatever pipe directory it serves.
# Matched on the COMMAND LINE (-f), never on comm: the kernel truncates comm to 15 chars, so
# `pgrep -x nvidia-cuda-mps-server` matches nothing and `down` would silently leave the
# wedged server running -- which is the one process this script most needs to reach.
daemon_pids() { pgrep -u "$(id -u)" -f 'nvidia-cuda-mps-control -d' 2>/dev/null; }
server_pids() { pgrep -u "$(id -u)" -f 'nvidia-cuda-mps-server' 2>/dev/null; }

# `quit` blocks until every server has drained. A wedged server never drains, so a bare
# `echo quit | nvidia-cuda-mps-control` hangs forever -- that is exactly the state this
# script exists to clean up. Bound it, then fall back to signals.
quit_daemon() {  # $1 = pipe directory
  local dir="$1"
  [[ -S "$dir/control" ]] || return 0
  say "  quit -> $dir"
  CUDA_MPS_PIPE_DIRECTORY="$dir" timeout 20 nvidia-cuda-mps-control <<<'quit' >/dev/null 2>&1
  local rc=$?
  [[ $rc -eq 124 ]] && say "  (quit timed out after 20s -- daemon unresponsive, using signals)"
  return 0
}

cmd_status() {
  say "pipe dir : $MPS_DIR"
  if [[ -d "$MPS_DIR" ]]; then
    say "  perms  : $(stat -c '%A %U:%G' "$MPS_DIR")"
    [[ -S "$MPS_DIR/control" ]] && say "  control: $(stat -c '%A %U:%G' "$MPS_DIR/control")"
  else
    say "  (does not exist)"
  fi
  say "log dir  : $MPS_LOG $( [[ -d "$MPS_LOG" ]] && echo "($(ls -1 "$MPS_LOG" 2>/dev/null | wc -l) file(s))" || echo '(does not exist)')"
  local d s
  d="$(daemon_pids | paste -sd, )"; s="$(server_pids | paste -sd, )"
  say "daemon(s): ${d:-none}"
  say "server(s): ${s:-none}"
  if [[ -d "$LEGACY_DIR" ]]; then
    say "LEGACY   : $LEGACY_DIR still present -- $(stat -c '%A %U:%G' "$LEGACY_DIR")"
    [[ "$(stat -c '%A' "$LEGACY_DIR")" == "drwxrwxrwx" ]] && say "           ^ world-writable: any user on this box can reach a daemon listening there"
  fi
  # A server that is alive but has burned almost no CPU is the 2026-07-28 failure mode.
  local p t
  for p in $(server_pids); do
    t=$(ps -o times= -p "$p" 2>/dev/null | tr -d ' ')
    say "  server $p: ${t:-?}s CPU, state $(ps -o stat= -p "$p" 2>/dev/null | tr -d ' '), wchan $(ps -o wchan:24= -p "$p" 2>/dev/null | tr -d ' ')"
  done
}

cmd_down() {
  say "stopping MPS (private dir AND the legacy /tmp one)"
  quit_daemon "$MPS_DIR"
  quit_daemon "$LEGACY_DIR"
  # Signal whatever survived the quit. TERM first, KILL only for the wedged ones.
  local left
  left="$(daemon_pids; server_pids)"
  if [[ -n "$left" ]]; then
    say "  TERM: $(echo $left | tr '\n' ' ')"
    kill -TERM $left 2>/dev/null
    for _ in 1 2 3 4 5; do
      left="$(daemon_pids; server_pids)"; [[ -z "$left" ]] && break
      timeout 2 tail -f /dev/null 2>/dev/null   # 2s pause without invoking sleep
    done
  fi
  left="$(daemon_pids; server_pids)"
  if [[ -n "$left" ]]; then
    say "  KILL: $(echo $left | tr '\n' ' ')"
    kill -KILL $left 2>/dev/null
  fi
  say "remaining daemon(s)=$(daemon_pids | wc -l) server(s)=$(server_pids | wc -l)"
}

cmd_up() {
  if [[ -n "$(daemon_pids)" ]]; then
    say "a daemon is already running (pid $(daemon_pids | paste -sd,)); use 'restart' to replace it"
    cmd_status; return 0
  fi
  mkdir -p "$MPS_DIR" "$MPS_LOG"
  chmod 700 "$MPS_DIR" "$MPS_LOG"
  say "starting daemon on $MPS_DIR (0700), logs -> $MPS_LOG"
  nvidia-cuda-mps-control -d
  cmd_status
  say "NOTE: clients must export CUDA_MPS_PIPE_DIRECTORY=$MPS_DIR to reach this daemon."
}

case "${1:-status}" in
  up)      cmd_up ;;
  down)    cmd_down ;;
  restart) cmd_down; cmd_up ;;
  status)  cmd_status ;;
  *) echo "usage: $0 up|down|status|restart"; exit 2 ;;
esac
