#!/bin/bash
cd /d/marble-home
PY=".venv/Scripts/python -W ignore"
run() { echo "=== $* ($(date +%T))"; "$@" 2>&1 | grep --line-buffered -v "WARNING\|INFO\|macros\|robosuite_models\|To setup"; }
run $PY scripts/collect_demos.py --task pick_up_cup --episodes 300 --workers 3 --seed 1000
run $PY scripts/prepare_dataset.py --task pick_up_cup --verify 3
run $PY scripts/train_bc.py --config configs/bc.yaml --set task=pick_up_cup --set run_name=bc_mlp_v2_pick_up_cup_s0 --workers 3
run $PY scripts/collect_demos.py --task place_cup --episodes 200 --workers 3 --seed 2000
run $PY scripts/prepare_dataset.py --task place_cup --verify 3
run $PY scripts/train_bc.py --config configs/bc.yaml --set task=place_cup --set run_name=bc_mlp_v2_place_cup_s0 --workers 3
run $PY scripts/collect_demos.py --task reach_cup --episodes 300 --workers 3 --seed 3000
run $PY scripts/prepare_dataset.py --task reach_cup --verify 3
run $PY scripts/train_bc.py --config configs/bc.yaml --set task=reach_cup --set run_name=bc_mlp_v2_reach_cup_s0 --workers 3
run $PY scripts/train_diffusion.py --config configs/diffusion.yaml --set task=pick_up_cup --set run_name=diffusion_v2_pick_up_cup_s0 --workers 3
echo V2_QUEUE_DONE
