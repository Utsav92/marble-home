#!/bin/bash
cd /d/marble-home
PY=".venv/Scripts/python -W ignore"
f() { grep -v "WARNING\|INFO\|macros\|robosuite_models\|To setup"; }
# scale = sqrt(prompt_area / raw_footprint_area); multiroom has no stated dimensions -> 1.0
declare -A SCALE=( [kitchen_001]=1.415 [living_room_001]=0.836 [bedroom_001]=1.038 [laundry_room_001]=0.829 [multiroom_apartment_001]=1.0 )
declare -A CAT=( [kitchen_001]=kitchen [living_room_001]=living_room [bedroom_001]=bedroom [laundry_room_001]=laundry_room [multiroom_apartment_001]=multiroom_apartment )
for k in kitchen_001 living_room_001 bedroom_001 laundry_room_001 multiroom_apartment_001; do
  echo "##### $k scale=${SCALE[$k]}"
  $PY scripts/import_marble.py --input data/marble_raw/marble_$k --category ${CAT[$k]} --scale ${SCALE[$k]} 2>&1 | f | grep -E "Error|RuntimeError|\"boxes\"|clearance|warnings" | head -5
  $PY scripts/validate_geometry.py --scene marble_$k --probe-expert 3 2>&1 | f | cut -c1-230
done
echo IMPORT_ALL_DONE
