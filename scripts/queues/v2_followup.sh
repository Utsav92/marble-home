#!/bin/bash
# waits for the first v2 queue to finish, then latency and the DR re-run (2 workers: the first attempt was OOM-killed)
cd /d/marble-home
PY=.venv/Scripts/python.exe
until grep -q "QUEUE DONE" logs/v2_queue.log; do sleep 60; done
echo "=== START latency $(date +%T)"; $PY scripts/measure_latency.py; echo "=== END latency rc=$?"
echo "=== START dr $(date +%T)"; $PY scripts/run_v2_experiments.py dr --seeds 0 1 --workers 2; echo "=== END dr rc=$?"
echo "=== FOLLOWUP DONE"
