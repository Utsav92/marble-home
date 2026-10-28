#!/bin/bash
cd /d/marble-home
PY=".venv/Scripts/python -W ignore"
while ! grep -q V2D_QUEUE_DONE results/v2d_queue_log.txt 2>/dev/null; do sleep 30; done
run() { echo "=== $* ($(date +%T))"; "$@" 2>&1 | grep --line-buffered -v "WARNING\|INFO\|macros\|robosuite_models\|To setup"; }
# keep the v1 scene results
mv results/experiments/scene_probe.json results/experiments/scene_probe_v1.json 2>/dev/null
mv results/experiments/scene_generalization.json results/experiments/scene_generalization_v1.json 2>/dev/null
# 1. benchmarks (v2 checkpoints only live at the top level of checkpoints/)
run $PY scripts/benchmark.py --task pick_up_cup --auto --episodes 30 --workers 3
run $PY scripts/benchmark.py --task place_cup --auto --episodes 30 --workers 3
run $PY scripts/benchmark.py --task reach_cup --auto --episodes 30 --workers 3
# 2. multi-scene generalisation with the v2 expert + delta actions
run $PY scripts/run_experiments.py scenes_probe
run $PY scripts/run_experiments.py scenes_collect --dataset scenes_pick_v2
run $PY scripts/relabel_delta.py --src scenes_pick_v2 --dst scenes_pick_v2delta
run $PY scripts/run_experiments.py scenes_eval --dataset scenes_pick_v2delta --single-ckpt checkpoints/bc_mlp_v2delta_pick_up_cup_s0.pt
# 3. PPO reach under v2 (delta actions)
run $PY scripts/train_rl.py --config configs/rl.yaml --set total_timesteps=24576 --set n_envs=3 --set task=reach_cup --set action_mode=ee_delta --set run_name=ppo_v2_reach_cup_delta_s0 --workers 3
echo V2E_QUEUE_DONE
