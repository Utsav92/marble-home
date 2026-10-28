# Limitations (read before trusting any number)

1. **Level A only.** All learned policies drive a *fixed-base* GR1 (`GR1FixedLowerBody`): torso, head, two arms, two hands. The legs
   do not move. `GR1FloatingBody` / `GR1` instantiate and step, but there is no validated locomotion controller or gait policy, so
   **Level B (assisted mobile manipulation) and Level C (whole-body control) are BLOCKED / not attempted.** Nothing here shows a
   learned walking humanoid.
2. **The policy controls only the right hand's position and a scalar grip.** Hand orientation and the left arm are held fixed; the
   24-D IK action is expanded from a 4-D policy action. Hand-orientation control and bimanual tasks are not learned.
3. **Scripted demonstrations, not human teleoperation.** The expert uses privileged simulator state (cup pose). Demos are therefore
   unimodal and cleaner than human data, which favours simple BC over diffusion.
4. **State policies see the cup pose (privileged).** Their success says "can imitate a good controller given perfect perception",
   not "can see and grasp". Vision modes hide it; see the report for the (small) image-policy results.
5. **One work-surface layout.** All scenes share the same counter-height work area; scenes differ in furniture, colours, counter
   height (0.82-0.92 m) and spawn region. Generalisation numbers are across *these* variations only.
6. **Marble: five real worlds were generated (1 per category, `marble-1.0-draft`, 1,150 credits) and composed, but every scale is an
   ASSUMPTION** (the API returned `semantics_metadata: null` for all of them): footprint-area matching to the prompt for four rooms and
   the minimum feasible scale (1.2) for the laundry room. Two rooms get implausibly low ceilings under these scales. No policy was
   trained or evaluated inside a Marble scene; only the scripted expert was run there. Only the draft model was tried.
7. **Gaussian splats are not rendered.** MuJoCo has no native splat rendering; no splat viewer was built. Splats are only kept as files.
8. **Household tasks are rigid-body proxies.** `clear_table` (3 cups -> flat tray), `load_dishwasher` (2 cups -> the rack plate of a static
   counter-top dishwasher) and `laundry_basket` (3 cylindrical "laundry rolls" -> a tote) are IMPLEMENTED, TESTED and trained
   (`simulation/layouts.py`). They are multi-item pick-and-place on the same work surface: **no door, rack sliding, washer/dryer, deformable cloth,
   or navigation between rooms**. Two tote walls and the dishwasher side walls are visual-only. Receptacle geometry was tuned until the scripted
   expert passed (item spawn bands, slot spacing and fill order encode what the open GR1 hand physically clears), so the layouts are not neutral.
   `LaundryTransfer` (washer -> dryer) and `MultiroomHousehold` remain **NOT IMPLEMENTED**.
   The three main policies are trained and evaluated at one counter height (0.85 m): the dishwasher policy scored 100 % at 0.85 m and 0-7 % at every
   other height tested (measured: 25 % over 0.82-0.91 m, 2 % at held-out heights), so those claims hold only at the training height. A separate
   dishwasher policy trained on 30 demos at each of 0.82/0.85/0.88/0.91 m reached 82 % on those heights and 89 % on held-out 0.835/0.865/0.90 m
   (interpolation inside the trained span, one seed, 15 episodes per height). Outside that span it fails: 0 % at 0.79 m and 27 % at 0.94 m
   (n=15 each). The other two tasks were not retrained across heights). Only default-geometry scenes were evaluated beyond that. BC seed variance is large (clear_table 63 / 77 / 93 % over seeds).
9. **Physics parameters are RoboSuite defaults** (PD gains, friction, masses) - not identified from hardware. "Sim-to-real" is out
   of scope.
10. **Controller v1 vs v2.** The first results (archived in `checkpoints/v1/`, `results/benchmark/v1/`, `reports/v1/`) used RoboSuite's
    default IK posture weights (waist 10). The expert then failed deterministically in the far corners of the cup-spawn region (81 %
    on a 21-point grid) and placed the cup 3.3-4.0 cm from the target. **v2 (default now)** lets the waist help reach (weight 1):
    expert 100 % on the grid, placement error ~0.4 cm, hand error ~0.7 cm. v1 and v2 numbers are NOT comparable (different robot
    behaviour and, for reach/place, stricter success thresholds: reach 3 cm + 5-step hold, place radius 3 cm).
11. **v2 policies need delta actions.** Trained on the same demos, an absolute-target BC scored 13 % under v2 (tracking lag makes
    "target = measured pose + small lead" too slow) while the delta-action BC scored 100 %. All v2 policies use `ee_delta`.
    The waist swings up to ~0.68 rad (39 deg) when reaching: a large whole-body lean, not a calm reach.
12. **Compute-limited statistics.** Evaluation uses 30 instances per policy (Wilson 95 % intervals are wide: ±~10 pp), 3 seeds for
    the data-scaling study, and short RL runs (tens of thousands of steps). Many differences are not statistically resolvable.
13. **Throughput**: about 17 env-steps/s (state) and 4-7 (with cameras) on the dev CPU. No parallel GPU simulation exists in
    MuJoCo/RoboSuite here; MuJoCo physics was **not** GPU-accelerated.
14. **GPU paths (Profile B/C) are UNVALIDATED.** `requirements/gpu.txt` exists but was never run.
