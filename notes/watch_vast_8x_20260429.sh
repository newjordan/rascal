#!/usr/bin/env bash
set -euo pipefail

LOG=${1:-/home/frosty40/sota_rascal/notes/vast_8x_watch_20260429.log}
RAW=/tmp/vast_instances_20260429.json
INTERVAL=${WATCH_INTERVAL_SECONDS:-20}

echo "watch_start $(date -u +%Y-%m-%dT%H:%M:%SZ) interval=${INTERVAL}s" >> "$LOG"

while true; do
  ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  if /home/frosty40/miniconda3/bin/vastai show instances --raw > "$RAW" 2>/tmp/vast_watch_err.txt; then
    {
      echo "== $ts =="
      jq -r '.[] | select(.num_gpus == 8 or .id == 35002131 or .id == 35778269) |
        [.id, (.label // "-"), (.num_gpus|tostring)+"x", .actual_status, .cur_state, .intended_status,
         (.public_ipaddr // "-"), (((.ports? // {})["22/tcp"][0].HostPort) // .ssh_port // "-"), (.disk_usage|tostring)+"%",
         (.gpu_util|tostring)+"%", (.status_msg // "-")] | @tsv' "$RAW"
    } >> "$LOG"
  else
    echo "== $ts == vastai_failed $(tr "\n" " " </tmp/vast_watch_err.txt)" >> "$LOG"
  fi
  sleep "$INTERVAL"
done
