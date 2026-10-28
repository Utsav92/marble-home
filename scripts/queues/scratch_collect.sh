#!/bin/bash
cd /d/marble-home
PY=.venv/Scripts/python
$PY scripts/collect_demos.py --task pick_up_cup --episodes 300 --workers 3 --seed 1000
$PY scripts/collect_demos.py --task place_cup --episodes 200 --workers 3 --seed 2000
$PY scripts/collect_demos.py --task reach_cup --episodes 100 --workers 3 --seed 3000
echo ALL_COLLECTION_DONE
