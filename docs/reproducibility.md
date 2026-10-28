# Reproducibility

## Environment (what was actually used)

Windows 11, Python 3.11.9, Intel i5-8250U (4C/8T), 16 GB RAM, **no NVIDIA GPU** (Profile A). `requirements/base.txt` pins the
combination that was installed and tested:

* `robosuite==1.5.2` requires `MjData.qM`, which `mujoco>=3.4` removed -> **`mujoco==3.3.7`** (the system Python had 3.14).
* The robosuite 1.5.2 wheel **omits `robosuite/examples/`**, so its default GR1 whole-body IK controller is missing. It is
  vendored (MIT, unmodified) in `src/marble_home/robots/vendor/mink_controller.py` and registered by `robots.gr1.register_mink_ik()`.
* `numpy<2` (numba, pulled in by robosuite).

```powershell
python -m venv .venv --system-site-packages     # or without --system-site-packages
.\.venv\Scripts\python -m pip install -r requirements/cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python -m pip install -e . --no-deps
.\.venv\Scripts\python scripts/check_environment.py
```

**WSL2 / Linux** (recommended for GPU training): same commands in a Linux venv; headless rendering needs `MUJOCO_GL=egl`
(NVIDIA) or `osmesa` (CPU). The Windows-native path is what this repo was developed and tested on; the Linux/WSL2 path is **UNTESTED**.

## What every run records

`results/runs/<name>/manifest.json`: git commit (+dirty flag), Python/MuJoCo/RoboSuite/torch/numpy versions, hardware, pip-freeze
hash, seed, duration, checkpoint path + SHA-256, dataset path + SHA-256. Plus `config.json`, `metrics.csv` (plots are drawn from it),
TensorBoard logs in `tb/`. Datasets carry `dataset_meta.json` with checksums, splits and replay-verification results.

## Determinism

* A task *instance* is fully determined by `env.reset(seed=s)` (cup pose, domain-randomisation draw). Evaluation uses seeds
  `900000+i`, disjoint from demonstration seeds (`1000+`, `2000+`, `3000+`), so every policy sees identical instances (paired tests).
* Physics is deterministic given state + actions up to the IK solver's internal state (see `docs/datasets.md`).
* Training uses seeded torch/numpy; CPU multi-threading can still introduce tiny float-order differences, so re-runs match
  statistically, not bit-for-bit.

## Commands that were executed to produce `reports/`

See `reports/final_report.md` ("Commands executed") - generated from the same logs.

## Isolation

`MARBLE_HOME_WORK=<dir>` redirects `data/`, `results/`, `reports/`, `checkpoints/` (used by the pipeline smoke test so it can
never touch real datasets). Datasets are namespaced by `--dataset` so multi-scene or ablation data never mixes with the base set.
