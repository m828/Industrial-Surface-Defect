#!/bin/bash
# Retry the FPS benchmark until a usable run (>=6 of 10 blocks free of foreign
# GPU processes) or max attempts reached. Contamination detection lives inside
# bench_fps_final.py (in-process 2s nvidia-smi poller, per-block judgement).
set -u
cd "/workspace/Industrial Surface Defect/Industrial-Surface-Defect"
DIR=experiments/baseline_comparison/results/fps_final
MAX_ATTEMPTS=8

for attempt in $(seq 1 $MAX_ATTEMPTS); do
  echo "=== attempt $attempt $(date '+%H:%M:%S') ==="
  # wait for durably idle GPU (3 clean checks, 10s apart)
  clean=0
  while true; do
    n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
    if [ "$n" -eq 0 ]; then
      clean=$((clean + 1))
      [ "$clean" -ge 3 ] && break
    else
      clean=0
      echo "$(date '+%H:%M:%S') gpu busy, waiting"
    fi
    sleep 10
  done
  echo "$(date '+%H:%M:%S') gpu idle confirmed, launching bench"

  python3 "$DIR/bench_fps_final.py"
  rc=$?
  echo "$(date '+%H:%M:%S') attempt $attempt bench rc=$rc"

  if [ "$rc" -eq 0 ]; then
    echo "=== USABLE RUN ACHIEVED on attempt $attempt ==="
    exit 0
  fi
  # archive unusable attempt
  [ -f "$DIR/fps_final_results.json" ] && mv "$DIR/fps_final_results.json" "$DIR/fps_attempt${attempt}_unusable.json"
  [ -f "$DIR/fps_final_run_meta.json" ] && mv "$DIR/fps_final_run_meta.json" "$DIR/fps_attempt${attempt}_unusable_meta.json"
  sleep 20
done
echo "=== NO USABLE RUN AFTER $MAX_ATTEMPTS ATTEMPTS ==="
exit 1
