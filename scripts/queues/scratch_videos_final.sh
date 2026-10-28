#!/bin/bash
cd /d/marble-home
PY=".venv/Scripts/python -W ignore"
f() { grep -v "WARNING\|INFO\|macros\|robosuite_models\|To setup" | cut -c1-230; }
V=results/videos
mkdir -p $V/v1_archive && mv $V/*.mp4 $V/v1_archive/ 2>/dev/null
PICK=checkpoints/bc_mlp_v2delta_pick_up_cup_s0.pt
PLACE=checkpoints/bc_mlp_v2d_place_cup_s0.pt
REACH=checkpoints/bc_mlp_v2d_reach_cup_s0.pt
DIFF=checkpoints/diffusion_v2d_pick_up_cup_s0.pt
MS=checkpoints/bc_mlp_scenes_pick_v2delta_multiscene_nogoal.pt
spec() { ls -d data/processed_worlds/$1/v*/ | sort -V | tail -1 | sed 's#$#scene_spec.json#'; }
$PY scripts/render_episode.py --policy random   --task pick_up_cup --seed 900003 --out $V/01_random_pick.mp4 2>&1 | f
$PY scripts/render_episode.py --policy scripted --task pick_up_cup --seed 900003 --out $V/02_scripted_expert_pick.mp4 2>&1 | f
$PY scripts/render_episode.py --policy $PICK    --task pick_up_cup --seed 900003 --out $V/03_bc_pick.mp4 2>&1 | f
$PY scripts/render_episode.py --policy $DIFF    --task pick_up_cup --seed 900003 --out $V/04_diffusion_pick.mp4 2>&1 | f
$PY scripts/render_episode.py --policy $REACH   --task reach_cup   --seed 900003 --out $V/05_bc_reach.mp4 --camera eval_front3q 2>&1 | f
$PY scripts/render_episode.py --policy $PLACE   --task place_cup   --seed 900003 --out $V/06_bc_place.mp4 --camera eval_front3q 2>&1 | f
$PY scripts/render_episode.py --policy $MS --task pick_up_cup --seed 930001 --scene-spec data/processed_worlds/proc_living_room_0016/v1/scene_spec.json --out $V/07_multiscene_bc_heldout_livingroom.mp4 2>&1 | f
for w in kitchen_001 living_room_001 bedroom_001 laundry_room_001 multiroom_apartment_001; do
  $PY scripts/render_episode.py --policy $MS --task pick_up_cup --seed 930001 --scene-spec $(spec marble_$w) --camera overview --out $V/08_marble_${w}_bc_pick.mp4 2>&1 | f
done
echo VIDEOS_FINAL_DONE
