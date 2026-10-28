# Architecture

```
World Labs Marble ──► worldlabs.client / scripts/generate_worlds.py   (paid; dry-run by default)
        │ (or a downloaded export: data/marble_raw/<id>/collider.glb [world.json])
        ▼
geometry.process / geometry.pipeline / scripts/import_marble.py
   load GLB → metric Z-up (scale, ground offset) → floor → repair → prune shards → dedupe
   → split (vertex connectivity) → decompose (voxel-merged AABBs | convex hulls) → task-placement search
        ▼   data/processed_worlds/<scene>/v<N>/{scene_spec.json, report.json, hulls/, visual.obj}
geometry.validate / scripts/validate_geometry.py        (pass/fail + screenshots)
        ▼
simulation.kitchen_arena  ──►  simulation.kitchen_env.KitchenManip (RoboSuite env, GR1 + cup + tasks)
        ▼                                   │ physics: MuJoCo only. Rendering: offscreen, used only for camera obs
simulation.gym_env.HouseholdEnv  (Gymnasium; obs modes; 4-D action space; privileged state kept in info)
        ▼
datasets.collect/hdf5  ──►  policies.{bc,diffusion}  ──►  training.{train_bc,train_diffusion,train_rl,curriculum}
        ▼                                   ▼
evaluation.{rollout,metrics}  ──►  scripts/benchmark.py ──► reports/ + results/benchmark/
        ▼
visualization.{video,dashboard}
```

## Replaceable interfaces (spec §1)

| Interface | Contract | Implementations |
|---|---|---|
| World generation | `WorldLabsClient.generate/wait/get_world/export/download` | World API client; procedural `scene_gen.make_scene`; synthetic `geometry.fixtures` |
| Scene export/import | raw dir (`collider.glb`, `world.json`, `meta.json`) → `process_export()` → `scene_spec.json` | Marble export / fixture |
| Geometry validation | `validate_scene(version_dir, out_dir)` → report dict | one |
| Robot embodiment | `robots.gr1.make_controller_config(robot, mode)`; `robots.action_space` | GR1 Level A/B/C robots; modes `ik`, `osc`, `joint` |
| Sensors | `sensors.obs_config.ObsConfig(mode)` | `state`, `rgb`, `rgbd`, `multicam` |
| Environment stepping | `HouseholdEnv.reset(seed)/step(a)` (Gymnasium) | one env; scene injected as a dict |
| Demonstrations | `datasets.hdf5.DemoWriter/DemoFile`, `collect.replay_check` | scripted expert |
| Policy training | `Agent.reset(env)/act(obs)`; `save/load_checkpoint` | random, scripted, BC-MLP, BC-CNN, diffusion, PPO |
| Evaluation | `evaluation.rollout.evaluate(spec, task, seeds)` | same harness for all agents |
| Visualisation | `visualization.video.rollout_video`, Streamlit dashboard | — |

## Scene-spec schema (shared by procedural and Marble-derived scenes)

`scene_id, kind, counter_height, counter_size, cup_spawn{x,y}, place_target{x,y,radius}, decor{...,extras[]}, visual{...},
static_geoms{boxes[], hulls[], visual_mesh}, cameras{}` — see `simulation/kitchen_arena.DEFAULT_SCENE`.
Marble supplies **static** collision geometry (`static_geoms`); the interactive work counter, robot and graspable cup are
always inserted by the environment (the spec's "inserted separately"), at a collision-free placement found by
`geometry.process.find_task_placement`.

## Physics vs rendering independence

Physics is MuJoCo only (`mj_step` via RoboSuite). Cameras are used only to produce observations/videos. Gaussian splats
are never loaded by the simulator; MuJoCo does not render them natively. An optional splat viewer is out of scope here.

## Key design decisions (and why)

* **Fixed-base GR1 (Level A)**: the only configuration whose learned manipulation is verified. Level B/C robots instantiate but
  no learned or validated locomotion exists (see `limitations.md`).
* **Reduced 4-D action space** (hand xyz + scalar grip): the 24-D IK action is >80 % constant for these tasks.
* **Boxes, not whole-mesh hulls, for rooms**: MuJoCo collides meshes as convex hulls; see `geometry/process.py`.
* **State policies see privileged cup pose**: documented; vision modes do not (`sensors/obs_config.py`).
