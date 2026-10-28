#!/bin/bash
cd /d/marble-home
PY=".venv/Scripts/python -W ignore"
f() { grep -v "WARNING\|INFO\|macros\|robosuite_models\|To setup"; }
echo "=== pick (full grid)"; $PY scripts/expert_map.py --task pick_up_cup --tag _v2 2>&1 | f
echo "=== place"; $PY scripts/expert_map.py --task place_cup --xs -0.05 0 0.05 --ys -0.21 -0.15 -0.09 --tag _v2 2>&1 | f
echo "=== reach"; $PY scripts/expert_map.py --task reach_cup --xs -0.05 0 0.05 --ys -0.21 -0.15 -0.09 --tag _v2 2>&1 | f
echo EXPERT_V2_DONE
