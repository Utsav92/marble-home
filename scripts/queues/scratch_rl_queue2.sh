#!/bin/bash
cd /d/marble-home
while ! grep -q RL_QUEUE_DONE results/rl_queue_log.txt 2>/dev/null; do sleep 20; done
.venv/Scripts/python scripts/train_rl.py --config configs/rl.yaml --set total_timesteps=24576 --set n_envs=3 --set task=reach_cup --set action_mode=ee_abs --set bc_init=true --set run_name=ppo_reach_cup_bcinit_s0 --workers 3
echo RL_QUEUE2_DONE
