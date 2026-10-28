#!/bin/bash
cd /d/marble-home
PY=".venv/Scripts/python -W ignore"
while ! grep -q RL_QUEUE2_DONE results/rl_queue2_log.txt 2>/dev/null; do sleep 20; done
run() { echo "=== $* ($(date +%T))"; "$@" 2>&1 | grep -v "WARNING\|INFO\|macros\|robosuite_models\|To setup"; }
run $PY scripts/run_experiments.py throughput
run $PY scripts/run_experiments.py ddim --steps 1 2 5 10 20
run $PY scripts/train_bc.py --config configs/bc.yaml --set task=place_cup --workers 3
run $PY scripts/train_bc.py --config configs/bc.yaml --set task=reach_cup --workers 3
run $PY scripts/run_experiments.py scenes_probe
run $PY scripts/run_experiments.py scenes_collect
run $PY scripts/run_experiments.py scenes_eval
run $PY scripts/run_experiments.py data_scaling --sizes 10 25 50 100 240 --seeds 0 1 2
run $PY scripts/benchmark.py --task pick_up_cup --auto --episodes 30 --workers 3
run $PY scripts/benchmark.py --task place_cup --auto --episodes 30 --workers 3
run $PY scripts/benchmark.py --task reach_cup --auto --episodes 30 --workers 3
echo EXP_QUEUE_DONE
