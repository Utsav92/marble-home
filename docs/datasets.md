# Dataset compatibility record (spec §9)

| Dataset | Status | Why |
|---|---|---|
| **Native scripted demos (this repo)** | **USED** | `scripts/collect_demos.py`; GR1, 4-D `ee_abs` action, 20 Hz |
| RoboMimic (Lift/Can/Square/Transport) | **NOT USED** | Franka Panda 7-DoF + parallel gripper, OSC delta-pose actions, different observation keys. No valid mapping to a GR1 hand + 6-channel grip without a retargeting model. |
| MimicGen | **NOT USED** | Same embodiment/action-space mismatch (generated from RoboMimic-style sources) |
| DROID | **NOT USED** | Real Franka, joint-velocity/Cartesian actions, wrist+external cameras at different intrinsics; large (TB-scale) download not attempted |
| Open X-Embodiment | **NOT USED** | Mixed embodiments; per-dataset action normalisation; no humanoid |
| LeRobot | **NOT INTEGRATED** | Possible future export target (`datasets/hdf5.py` → LeRobotDataset) but not implemented |

None of these were downloaded, so no licence, version, episode count or split is claimed for them. Third-party demonstrations
are **not** native humanoid demonstrations and are never mixed with the native data. Integrating one would require, at minimum:
an embodiment retargeting step (EE pose → GR1 hand pose), an action-space conversion, camera-intrinsics matching, and a licence check.

## Native dataset record

Written automatically by `scripts/prepare_dataset.py` to `data/demonstrations/<dataset>/dataset_meta.json`: version
(`marble_home.demo.v1`), source, licence (generated), embodiment, action representation, observation modalities, episode/step
counts, sampling frequency, task labels, split sizes, conversions, scene, simulator versions, SHA-256, replay verification results.

Replay verification is **near-deterministic, not bit-exact**: the IK solver's internal state is not part of the saved MuJoCo
state, giving <1 mrad drift early and ≤7 mrad during contact-rich finger closing (measured). Acceptance: proprio error < 0.02,
cup error < 5 mm and the same success outcome.
