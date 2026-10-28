#!/bin/bash
cd /d/marble-home
PY=.venv/Scripts/python.exe
for st in "ddim" "data_scaling" "proprio" "dr --seeds 0 1" "images" "curriculum --seeds 0 1"; do
  echo "=== START $st $(date +%T)"
  $PY scripts/run_v2_experiments.py $st
  echo "=== END $st rc=$? $(date +%T)"
done
echo "=== QUEUE DONE"
