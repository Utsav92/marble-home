#!/bin/bash
cd /d/marble-home
PY=".venv/Scripts/python -W ignore"
while ! grep -q EXP_QUEUE3_DONE results/exp_queue3_log.txt 2>/dev/null; do sleep 30; done
run() { echo "=== $* ($(date +%T))"; "$@" 2>&1 | grep --line-buffered -v "WARNING\|INFO\|macros\|robosuite_models\|To setup"; }
run $PY scripts/run_experiments.py images
run $PY scripts/run_experiments.py curriculum
echo EXP_QUEUE4_DONE
