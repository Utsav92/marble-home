#!/bin/bash
# runs after the curriculum run ends: low-memory serial chain (CNN re-eval 1 worker, latency, DR re-run 2 workers)
cd /d/marble-home
PY=.venv/Scripts/python.exe
until grep -q "QUEUE DONE" logs/v2_queue.log; do sleep 60; done
echo "=== START reeval $(date +%T)"; $PY scripts/reeval_cnn_v2.py; echo "=== END reeval rc=$?"
echo "=== START latency $(date +%T)"; $PY scripts/measure_latency.py; echo "=== END latency rc=$?"
echo "=== START dr $(date +%T)"; $PY scripts/run_v2_experiments.py dr --seeds 0 1 --workers 2; echo "=== END dr rc=$?"
echo "=== SERIAL DONE"
