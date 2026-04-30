#!/usr/bin/env bash
set -euo pipefail

PRIMARY_PID="${1:-}"
LOG_DIR="/workspace/sota_rascal/notes/runtime_logs"
mkdir -p "$LOG_DIR"
QUEUE_LOG="$LOG_DIR/queue_10k_after_pr1855_$(date -u +%Y%m%d_%H%M%S).log"
TENK_DIR="/workspace/sota_rascal/legs/2026-04-30_mikey_sp10240_pr1855_lqer_pergroup_ttt_8x"
TENK_RUN_LOG="$TENK_DIR/logs/mikey_sp10240_pr1855_lqer_pergroup_ttt_8x_seed444.txt"

log() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$QUEUE_LOG"
}

log "queue_start primary_pid=${PRIMARY_PID:-none}"
log "waiting_for_current_torchrun"

while true; do
  if [ -n "$PRIMARY_PID" ] && kill -0 "$PRIMARY_PID" 2>/dev/null; then
    sleep 30
    continue
  fi

  active="$(pgrep -af '/venv/main/bin/torchrun|torchrun --standalone' || true)"
  if [ -n "$active" ]; then
    log "other_torchrun_still_active"
    printf '%s\n' "$active" | tee -a "$QUEUE_LOG"
    sleep 30
    continue
  fi
  break
done

log "starting_10k_runner dir=$TENK_DIR"
cd "$TENK_DIR"
mkdir -p logs
./launch_8x.sh > "$TENK_RUN_LOG.launcher.log" 2>&1
status=$?
log "10k_runner_exit status=$status"
exit "$status"
