# marble-home

Household humanoid manipulation research stack: **World Labs Marble** (scene diversity) + **MuJoCo / RoboSuite** (physics, Fourier GR1
humanoid) + **PyTorch** policies (behaviour cloning, diffusion policy) + **Stable-Baselines3** PPO.

> **Read `docs/limitations.md` first.** Short version: the humanoid is *fixed-base* (Level A); policies control the right hand's
> position + a scalar grip; demonstrations are scripted; **no live Marble generation has ever been run** (no API key) and the
> "Marble-derived" scene is a synthetic fixture; tasks 4-7 (clear table, dishwasher, laundry, multiroom) are not implemented.

Status labels: **VERIFIED** (executed here, evidence in `results/`), **IMPLEMENTED** (written, only partly exercised), **TESTED**
(automated tests), **EXPERIMENTAL**, **BLOCKED** (with the precise blocker). The full table is `docs/integration_status.md`.

## Install

Developed and tested on **Windows 11 + Python 3.11, CPU only** (no NVIDIA GPU). Dependencies are pinned in `requirements/`.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements/cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python -m pip install -e . --no-deps
.\.venv\Scripts\python scripts/check_environment.py
```
Why the pins matter (both verified failures, see `docs/reproducibility.md`): `robosuite==1.5.2` needs `mujoco==3.3.x`
(MuJoCo >= 3.4 removed `MjData.qM`), and the RoboSuite wheel omits its GR1 IK controller, which is vendored here.

**WSL2 / Linux** (UNTESTED here): create a venv, `pip install -r requirements/gpu.txt` (or `cpu.txt`), `pip install -e . --no-deps`;
for headless rendering set `MUJOCO_GL=egl` (NVIDIA) or `osmesa`. Native Windows needs no GL variable.

## Quick start (every command below was executed)

```bash
python scripts/check_environment.py                       # versions, GR1 + render check, integration status
python scripts/test_robot.py --headless                   # M1 smoke test: 100 steps, physics check, expert video
python scripts/generate_worlds.py --config configs/marble/default.yaml   # DRY RUN: prints Marble requests, spends nothing
python scripts/generate_worlds.py --procedural 18         # seeded procedural scenes (5 categories) + scene-level splits
python scripts/import_marble.py --make-fixture kitchen_fixture_001   # SYNTHETIC Marble-like export -> processed scene
python scripts/validate_geometry.py --scene kitchen_fixture_001      # pass/fail + screenshots
python scripts/collect_demos.py --task pick_up_cup --episodes 100 --workers 3
python scripts/prepare_dataset.py --task pick_up_cup --verify 5      # merge, split, stats, replay verification
python scripts/train_bc.py --config configs/bc.yaml
python scripts/train_diffusion.py --config configs/diffusion.yaml
python scripts/train_rl.py --config configs/rl.yaml --set task=reach_cup
python scripts/benchmark.py --config configs/evaluation.yaml --auto
python scripts/render_episode.py --policy best --task pick_up_cup
streamlit run src/marble_home/visualization/dashboard.py
python scripts/run_pipeline.py --profile smoke           # all of the above, tiny budgets, isolated in ./.smoke
python -m pytest                                          # offline tests (paid Marble tests are opt-in: -m network)
```

Live Marble generation spends paid credits and is never started implicitly:
`WORLDLABS_API_KEY=... python scripts/generate_worlds.py --confirm-spend --max-scenes 1`.

## Layout

`src/marble_home/` - `worldlabs/` (API client, manifest/registry), `geometry/` (GLB -> MuJoCo collision, validation, synthetic
fixtures), `simulation/` (kitchen arena/env, Gymnasium wrapper, scene generator, domain randomisation), `robots/` (GR1 adapter,
action space, vendored IK controller), `sensors/`, `tasks/` (success / reward / phase logic), `datasets/`, `policies/`,
`training/`, `evaluation/`, `visualization/`. `scripts/` are thin CLIs. `docs/` has architecture, Marble import contract, datasets,
limitations, reproducibility, integration status. `reports/` is generated from logged results.

## Results

See `reports/final_report.md` (narrative) and `reports/results_tables.md` (generated tables). Every number in them comes from
files under `results/`.
