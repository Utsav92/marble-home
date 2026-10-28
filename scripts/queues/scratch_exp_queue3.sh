#!/bin/bash
cd /d/marble-home
PY=".venv/Scripts/python -W ignore"
run() { echo "=== $* ($(date +%T))"; "$@" 2>&1 | grep --line-buffered -v "WARNING\|INFO\|macros\|robosuite_models\|To setup"; }
run $PY scripts/run_experiments.py throughput
run $PY scripts/run_experiments.py action_repr
run $PY scripts/run_experiments.py dr
echo EXP_QUEUE3_DONE
