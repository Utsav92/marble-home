#!/bin/bash
cd /d/marble-home
PY=".venv/Scripts/python -W ignore"
while ! grep -q EXP_QUEUE4_DONE results/exp_queue4_log.txt 2>/dev/null; do sleep 30; done
run() { echo "=== $* ($(date +%T))"; "$@" 2>&1 | grep --line-buffered -v "WARNING\|INFO\|macros\|robosuite_models\|To setup"; }
run $PY scripts/run_experiments.py dr
echo EXP_QUEUE5_DONE
