#!/bin/bash
cd /d/marble-home
PY=".venv/Scripts/python -W ignore"
f() { grep -v "WARNING\|INFO\|macros\|robosuite_models\|To setup" | cut -c1-200; }
V=results/videos
MS=$(ls checkpoints/bc_mlp_scenes_pick*_multiscene.pt 2>/dev/null | head -1)
echo "multiscene ckpt: $MS"
SP_TEST=$(ls data/processed_worlds/proc_kitchen_0015/v1/scene_spec.json)
SP_MARBLE=$(ls -d data/processed_worlds/marble_living_room_001/v*/ | sort -V | tail -1)scene_spec.json
echo "marble spec: $SP_MARBLE"
$PY scripts/render_episode.py --policy random   --task pick_up_cup --seed 900003 --out $V/01_random_pick.mp4 2>&1 | f
$PY scripts/render_episode.py --policy scripted --task pick_up_cup --seed 900003 --out $V/02_scripted_expert_pick.mp4 2>&1 | f
$PY scripts/render_episode.py --policy checkpoints/diffusion_pick_up_cup_n240_s0.pt --task pick_up_cup --seed 900003 --out $V/04_diffusion_pick.mp4 2>&1 | f
$PY scripts/render_episode.py --policy checkpoints/bc_mlp_place_cup_n160_s0.pt --task place_cup --seed 900003 --out $V/05_bc_place.mp4 --camera eval_front3q 2>&1 | f
$PY scripts/render_episode.py --policy scripted --task place_cup --seed 900003 --out $V/05b_scripted_place.mp4 --camera eval_front3q 2>&1 | f
if [ -n "$MS" ]; then
  $PY scripts/render_episode.py --policy $MS --task pick_up_cup --seed 930001 --scene-spec $SP_TEST --out $V/06_multiscene_bc_heldout_scene.mp4 2>&1 | f
  $PY scripts/render_episode.py --policy $MS --task pick_up_cup --seed 930001 --scene-spec $SP_MARBLE --camera overview --out $V/07_multiscene_bc_in_marble_livingroom.mp4 2>&1 | f
fi
$PY scripts/render_episode.py --policy scripted --task pick_up_cup --seed 930001 --scene-spec $SP_MARBLE --camera overview --out $V/07b_scripted_in_marble_livingroom.mp4 2>&1 | f
echo VIDEOS_DONE
