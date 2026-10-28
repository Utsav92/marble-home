#!/bin/bash
cd /d/marble-home
PY=.venv/Scripts/python
# wait for the first PPO run to finish
while [ ! -f checkpoints/ppo_reach_cup_scratch_s0.zip ]; do sleep 15; done
$PY scripts/train_rl.py --config configs/rl.yaml --set total_timesteps=24576 --set n_envs=3 --set task=reach_cup --set action_mode=ee_delta --set run_name=ppo_reach_cup_delta_s0 --workers 3
$PY scripts/train_rl.py --config configs/rl.yaml --set total_timesteps=24576 --set n_envs=3 --set task=reach_cup --set action_mode=ee_abs --set log_std_init=-1.5 --set run_name=ppo_reach_cup_abs_lowstd_s0 --workers 3
echo RL_QUEUE_DONE
