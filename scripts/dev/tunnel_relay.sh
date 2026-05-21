#!/usr/bin/env bash
# tunnel_relay.sh — local <-> Bunya file transfer via the existing tmux `bunya` pane.
#
# Reuses the already-authenticated SSH session by adding port forwards at runtime
# (OpenSSH ~C escape). No new ssh/scp/rsync; no Duo prompt; no login-node ban risk.
# Validated 2026-05-15 on 8.78 GB at ~66 MB/s sustained, sha256 verified end-to-end.
#
# Usage:
#   scripts/dev/tunnel_relay.sh push <local_file> <bunya_path>
#   scripts/dev/tunnel_relay.sh pull <bunya_file> <local_path>
# Options:
#   --port N      pin port (default: first free at/above 18443, both ends)
#   --no-verify   skip sha256 verification (faster; only for trusted reruns)
# Env:
#   RELAY_PANE    tmux session/pane (default: "bunya")
#
# Mechanism:
#   push: workstation runs python http.server on 127.0.0.1:PORT; ~C -R adds the
#         remote forward; bunya pane curls back through the tunnel.
#   pull: same idea inverted — bunya pane runs the http.server; ~C -L adds the
#         local forward; workstation curls.
# Verification: source side hashes the file and serves /.sha256 alongside; the
# destination side compares before atomic mv .partial -> final.

set -euo pipefail

PANE="${RELAY_PANE:-bunya}"
PORT_BASE=18443
PORT=""
VERIFY=1
MODE=""
SRC=""
DST=""

die()  { echo "[relay] ERROR: $*" >&2; exit 1; }
log()  { echo "[relay] $*" >&2; }

usage() {
  sed -n '2,/^$/p' "$0" >&2
  exit 2
}

[[ $# -lt 3 ]] && usage
MODE=$1; SRC=$2; DST=$3; shift 3
while [[ $# -gt 0 ]]; do
  case $1 in
    --port)      PORT=$2; shift 2 ;;
    --no-verify) VERIFY=0; shift ;;
    *) usage ;;
  esac
done
[[ $MODE != push && $MODE != pull ]] && usage

tmux has-session -t "$PANE" 2>/dev/null || die "tmux session '$PANE' not found"

free_port() {
  local p=$PORT_BASE
  while ss -ltn 2>/dev/null | awk '{print $4}' | grep -q ":$p$"; do p=$((p+1)); done
  echo "$p"
}
[[ -z $PORT ]] && PORT=$(free_port)

token() { uuidgen | tr -d -; }
pane_buf() { tmux capture-pane -t "$PANE" -p -S -300; }

pane_exec() {
  # Run a command in the bunya pane; wait for OK_<tok> / FAIL_<tok>.
  local cmd=$1 tok t0
  tok=$(token); t0=$(date +%s)
  tmux send-keys -t "$PANE" -- "{ $cmd ; } && echo OK_$tok || echo FAIL_$tok" Enter
  while :; do
    if pane_buf | grep -qE "^OK_$tok$";   then return 0; fi
    if pane_buf | grep -qE "^FAIL_$tok$"; then return 1; fi
    (( $(date +%s) - t0 > 7200 )) && die "pane_exec timeout: $cmd"
    sleep 1
  done
}

ssh_escape() {
  # $1 is the ssh> arg, e.g. "-R 18443:127.0.0.1:18443" or "-KR 18443".
  tmux send-keys -t "$PANE" "" Enter
  sleep 0.3
  tmux send-keys -t "$PANE" '~C'
  sleep 0.7
  tmux send-keys -t "$PANE" -- "$1" Enter
  sleep 0.8
}

start_local_http() {
  ( cd "$1" && nohup python3 -m http.server "$2" --bind 127.0.0.1 \
      > "/tmp/relay-http-$2.log" 2>&1 & echo $! > "/tmp/relay-http-$2.pid" )
  sleep 1
  [[ -s /tmp/relay-http-$2.pid ]] || die "local http server failed to start"
}
stop_local_http() {
  [[ -f /tmp/relay-http-$1.pid ]] || return 0
  kill "$(cat /tmp/relay-http-$1.pid)" 2>/dev/null || true
  rm -f /tmp/relay-http-$1.pid /tmp/relay-http-$1.log
}

PUSH_STAGE=""
do_push() {
  [[ -f $SRC ]] || die "local file not found: $SRC"
  local fn size sum dd dn
  fn=$(basename "$SRC")
  size=$(stat -c%s "$SRC")
  PUSH_STAGE=$(mktemp -d /tmp/relay-stage.XXXXXX)
  trap 'set +e; stop_local_http "$PORT"; [[ -n $PUSH_STAGE ]] && rm -rf "$PUSH_STAGE"; ssh_escape "-KR $PORT" 2>/dev/null' EXIT
  ln -s "$(readlink -f "$SRC")" "$PUSH_STAGE/$fn"
  log "push: $SRC ($(numfmt --to=iec "$size")) -> bunya:$DST  [port=$PORT]"
  if [[ $VERIFY -eq 1 ]]; then
    log "hashing local..."
    sum=$(sha256sum "$(readlink -f "$SRC")" | awk '{print $1}')
    echo "$sum" > "$PUSH_STAGE/.sha256"
    log "sha256=$sum"
  fi
  start_local_http "$PUSH_STAGE" "$PORT"
  curl -sfI "http://127.0.0.1:$PORT/$fn" >/dev/null || die "local http unreachable"
  ssh_escape "-R $PORT:127.0.0.1:$PORT"
  pane_buf | grep -q "Forwarding port\." || die "remote forward failed"
  pane_exec "curl -sfI http://127.0.0.1:$PORT/$fn >/dev/null" || die "tunnel unreachable from bunya"
  dd=$(dirname "$DST"); dn=$(basename "$DST")
  log "transferring..."
  pane_exec "mkdir -p '$dd' && time curl -fsS --retry 3 --continue-at - -o '$dd/$dn.partial' http://127.0.0.1:$PORT/$fn" \
    || die "curl on bunya failed"
  if [[ $VERIFY -eq 1 ]]; then
    pane_exec "[ \"\$(sha256sum '$dd/$dn.partial' | awk '{print \$1}')\" = '$sum' ]" \
      || die "sha256 mismatch on bunya side"
    log "sha256 verified on bunya"
  fi
  pane_exec "mv '$dd/$dn.partial' '$dd/$dn'"
  log "push done: bunya:$DST"
}

do_pull() {
  local dstdir; dstdir=$(dirname "$DST")
  [[ -d $dstdir ]] || die "local dst dir missing: $dstdir"
  local fn stage; fn=$(basename "$SRC")
  stage="/tmp/relay-stage-$PORT"
  pane_exec "[ -f '$SRC' ]" || die "remote file not found: bunya:$SRC"
  log "pull: bunya:$SRC -> local:$DST  [port=$PORT]"
  log "staging on bunya..."
  if [[ $VERIFY -eq 1 ]]; then
    pane_exec "mkdir -p '$stage' && ln -sf \"\$(readlink -f '$SRC')\" '$stage/$fn' && sha256sum \"\$(readlink -f '$SRC')\" | awk '{print \$1}' > '$stage/.sha256'" \
      || die "stage on bunya failed"
  else
    pane_exec "mkdir -p '$stage' && ln -sf \"\$(readlink -f '$SRC')\" '$stage/$fn'"
  fi
  pane_exec "cd '$stage' && nohup python3 -m http.server $PORT --bind 127.0.0.1 > /tmp/relay-http-$PORT.log 2>&1 & echo \$! > /tmp/relay-http-$PORT.pid; sleep 1; [ -s /tmp/relay-http-$PORT.pid ]" \
    || die "bunya http server failed to start"
  trap "set +e; pane_exec \"kill \\\$(cat /tmp/relay-http-$PORT.pid 2>/dev/null) 2>/dev/null; rm -rf /tmp/relay-http-$PORT.pid /tmp/relay-http-$PORT.log '$stage'\"; ssh_escape '-KL $PORT' 2>/dev/null" EXIT
  ssh_escape "-L $PORT:127.0.0.1:$PORT"
  pane_buf | grep -q "Forwarding port\." || die "local forward failed"
  curl -sfI "http://127.0.0.1:$PORT/$fn" >/dev/null || die "tunnel unreachable from local"
  log "transferring..."
  curl -fsS --retry 3 --continue-at - -o "$DST.partial" "http://127.0.0.1:$PORT/$fn"
  if [[ $VERIFY -eq 1 ]]; then
    local rsum lsum
    rsum=$(curl -fsS "http://127.0.0.1:$PORT/.sha256")
    lsum=$(sha256sum "$DST.partial" | awk '{print $1}')
    [[ $lsum == "$rsum" ]] || die "sha256 mismatch: local=$lsum remote=$rsum"
    log "sha256 verified on local"
  fi
  mv "$DST.partial" "$DST"
  log "pull done: local:$DST"
}

case $MODE in
  push) do_push ;;
  pull) do_pull ;;
esac
