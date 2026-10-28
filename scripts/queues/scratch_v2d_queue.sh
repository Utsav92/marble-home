#!/bin/bash
cd /d/marble-home
PY=".venv/Scripts/python -W ignore"
run() { echo "=== $* ($(date +%T))"; "$@" 2>&1 | grep --line-buffered -v "WARNING\|INFO\|macros\|robosuite_models\|To setup"; }
# patch the already-trained delta pick checkpoint so it records its action mode
run $PY -c "
import torch
p='checkpoints/bc_mlp_v2delta_pick_up_cup_s0.pt'; ck=torch.load(p, weights_only=False); ck['meta']['action_mode']='ee_delta'; torch.save(ck, p); print('patched meta of', p)"
# place
run $PY scripts/relabel_delta.py --src place_cup --dst place_cup_v2delta
run $PY scripts/train_bc.py --config configs/bc.yaml --set task=place_cup --set dataset=place_cup_v2delta --set run_name=bc_mlp_v2d_place_cup_s0 --workers 3
# reach
run $PY scripts/collect_demos.py --task reach_cup --episodes 300 --workers 3 --seed 3000
run $PY scripts/prepare_dataset.py --task reach_cup --verify 3
run $PY scripts/relabel_delta.py --src reach_cup --dst reach_cup_v2delta
run $PY scripts/train_bc.py --config configs/bc.yaml --set task=reach_cup --set dataset=reach_cup_v2delta --set run_name=bc_mlp_v2d_reach_cup_s0 --workers 3
# diffusion on delta pick
run $PY scripts/train_diffusion.py --config configs/diffusion.yaml --set task=pick_up_cup --set dataset=pick_up_cup_v2delta --set run_name=diffusion_v2d_pick_up_cup_s0 --workers 3
echo V2D_QUEUE_DONE
