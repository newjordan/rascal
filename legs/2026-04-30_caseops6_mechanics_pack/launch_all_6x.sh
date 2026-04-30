#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/sota_rascal/legs}"

lanes=(
  "2026-04-30_caseops6_lane0_repro_mlp4_late035_1x"
  "2026-04-30_caseops6_lane1_mlp375_late050_1x"
  "2026-04-30_caseops6_lane2_mlp385_late050_1x"
  "2026-04-30_caseops6_lane3_mlp375_late060_1x"
  "2026-04-30_caseops6_lane4_mlp4_late050_1x"
  "2026-04-30_caseops6_lane5_mlp375_loop3_late060_1x"
)

for gpu in "${!lanes[@]}"; do
  lane="${lanes[$gpu]}"
  cd "${ROOT}/${lane}"
  mkdir -p logs
  stamp="$(date -u +%Y%m%d_%H%M%S)"
  launcher_log="logs/manual_${lane}_gpu${gpu}_${stamp}.launcher.log"
  echo "launch gpu=${gpu} lane=${lane} log=${launcher_log}"
  nohup env GPU_ID="${gpu}" ./launch_1x.sh > "${launcher_log}" 2>&1 &
  echo "$!" > "logs/manual_gpu${gpu}.pid"
done

echo "launched ${#lanes[@]} lanes"
echo "watch: tail -n +1 -f ${ROOT}/2026-04-30_caseops6_lane*/logs/*.txt"
