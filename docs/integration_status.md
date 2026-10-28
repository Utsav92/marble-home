# Integration status

Labels: **VERIFIED** = executed here with evidence · **TESTED** = automated tests · **IMPLEMENTED** = code exists, only partly exercised ·
**EXPERIMENTAL** · **BLOCKED** = cannot proceed (reason given) · **NOT IMPLEMENTED**.

| Component | Status | Evidence / notes |
|---|---|---|
| MuJoCo 3.3.7 + RoboSuite 1.5.2 on Windows CPU | VERIFIED | `results/environment.json`; pins explained in `docs/reproducibility.md` |
| GR1 humanoid model (Level A `GR1FixedLowerBody`) | VERIFIED + TESTED | compiles, steps, finite states, joint limits valid (`results/m1/m1_smoke.json`, `tests/test_imports_and_robot.py`) |
| GR1FloatingBody / GR1 (Level B / C robots) | IMPLEMENTED (instantiate + step only) | no locomotion controller; **Level B/C learning: BLOCKED** |
| Whole-body mink IK controller | VERIFIED | vendored from robosuite v1.5.2 (wheel omits it) |
| Three action representations (`ik`, `osc`, `joint`) | `ik`: VERIFIED + used; `osc`/`joint`: IMPLEMENTED (step only, no learning) | `robots/gr1.py` |
| Cameras: head, 2 wrist, RGB + depth | VERIFIED | `tests`, `results/m1/head_rgb.png` |
| Procedural kitchen + cup + counter | VERIFIED | `results/scene_views.png` |
| Task 1-3: ReachCup / PickUpCup / PlaceCup | VERIFIED (scripted expert, BC, diffusion) | `reports/benchmark_*.md` |
| Task 4-7: ClearTable, LoadDishwasher, LaundryTransfer, Multiroom | **NOT IMPLEMENTED** | counter-top static dishwasher geometry exists in `kitchen_arena` but no task/reward |
| Scripted-expert demonstrations, HDF5, replay verification | VERIFIED + TESTED | `data/demonstrations/*/dataset_meta.json` |
| Human teleoperation | NOT IMPLEMENTED | |
| Behaviour cloning (MLP, state) | VERIFIED | `reports/` |
| Behaviour cloning (CNN, images) | see report (EXPERIMENTAL) | `scripts/render_dataset.py`, `configs/bc_cnn.yaml` |
| Diffusion policy (own compact implementation) | VERIFIED (trains, rolls out) | not the official code; see report for performance |
| PPO (SB3), reward diagnostics, BC-initialised actor | VERIFIED (runs, logs) | short budgets; see report |
| Diffusion-checkpoint -> PPO transfer | NOT SUPPORTED (by design) | different architecture; BC-actor distillation used instead (`training/train_rl.py`) |
| Curriculum scheduler (CI-based advance / rollback / log) | IMPLEMENTED + unit-checked | stages 5 (Marble kitchens) and 7 (navigation) **BLOCKED** |
| Domain randomisation (visual / physics / sensor) | VERIFIED (reproducible, expert robust) | `simulation/randomization.py` |
| World API client (generate / poll / get / export / download) | **VERIFIED live** (one capped draft generation, 230 credits) + TESTED offline | `data/marble_raw/registry.json`, `docs/marble_import.md` |
| Real Marble GLB -> MuJoCo static collision | **VERIFIED on one real export** (337k faces -> 2,755 boxes; compiles, steps, collision-free spawn) | `results/geometry/marble_kitchen_001/`; **scale is an assumption** (API returned null `metric_scale_factor`) |
| Marble Gaussian-splat rendering | **NOT IMPLEMENTED** | PLY is downloaded and checksummed only; MuJoCo cannot render splats |
| Marble coverage of 5 categories | **VERIFIED: 5 worlds generated and composed** (1,150 credits, `marble-1.0-draft`, 1 world per category) | all 5 pass geometry validation (`results/geometry/marble_*/`). **No world returned `metric_scale_factor`**, so every scale is an ASSUMPTION: prompt-footprint-area matching for kitchen (1.415), living room (0.836), bedroom (1.038), multiroom (1.0, no stated size); laundry room (1.2) is the MINIMUM scale at which the task fits (at the prompt-derived 0.829 it does not fit; `results/experiments/marble_laundry_scale_feasibility.json`). Living room (1.41 m) and multiroom (1.91 m) come out with implausibly low ceilings under these scales (advisory flags). No policy was trained inside a Marble scene; only the scripted expert was run there. |
| Geometry validation + screenshots | VERIFIED | fixture + real scene |
| Streamlit dashboard (5 views) | VERIFIED (renders, `tests/test_dashboard.py`) | |
| RoboMimic / MimicGen / DROID / OXE / LeRobot data | NOT USED | embodiment/action-space mismatch; `docs/datasets.md` |
| GPU training (Profile B/C) | UNVALIDATED | no GPU available |
| Linux / WSL2 | UNTESTED | |
