#!/bin/bash
# Wait for GPU to be durably idle (3 consecutive clean checks, 20s apart),
# then run the unified 10-model FPS benchmark.
set -u
cd "/workspace/Industrial Surface Defect/Industrial-Surface-Defect"
clean=0
while true; do
  n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
  if [ "$n" -eq 0 ]; then
    clean=$((clean + 1))
    echo "$(date '+%H:%M:%S') gpu idle check $clean/3"
    if [ "$clean" -ge 3 ]; then break; fi
  else
    echo "$(date '+%H:%M:%S') gpu busy ($n foreign procs), waiting"
    clean=0
  fi
  sleep 20
done
echo "$(date '+%H:%M:%S') GPU idle confirmed, starting benchmark"
python3 experiments/baseline_comparison/results/fps_final/bench_fps_final.py
echo "$(date '+%H:%M:%S') benchmark exit=$?"
