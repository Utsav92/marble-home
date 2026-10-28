"""Geometry validation for a processed (Marble-derived or fixture) scene: pass/fail report + screenshots."""
from __future__ import annotations

import json
from pathlib import Path

import imageio
import mujoco
import numpy as np

from marble_home.utils import provenance

# volume the right arm / hand must be able to use around the counter (task frame, metres)
TASK_ZONE = {"x": (-0.60, 0.45), "y": (-0.65, 0.65), "z": (0.88, 1.60)}
SPAWN_ZONE_XY = {"x": (-0.75, -0.30), "y": (-0.45, 0.45)}  # where the robot stands (behind the counter edge)


def latest_version_dir(scene_id: str, root: Path | None = None) -> Path:
    from marble_home.utils.paths import DATA

    base = (Path(root) if root else DATA / "processed_worlds") / scene_id
    vs = sorted([p for p in base.glob("v*") if p.is_dir()], key=lambda p: int(p.name[1:]))
    if not vs:
        raise FileNotFoundError(f"no processed versions under {base}; run scripts/import_marble.py first")
    return vs[-1]


def _overlap(lo, hi, zone) -> bool:
    return all(lo[i] < zone[a][1] and hi[i] > zone[a][0] for i, a in enumerate("xyz"))


def validate_scene(version_dir: str | Path, out_dir: str | Path, probe_expert: int = 3, render: bool = True,
                   max_boxes: int | None = None, max_hull_faces: int | None = None) -> dict:
    from marble_home.simulation.gym_env import HouseholdEnv

    vdir, out = Path(version_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    spec = json.loads((vdir / "scene_spec.json").read_text())
    proc = json.loads((vdir / "report.json").read_text())
    checks: dict[str, dict] = {}

    def rec(name, ok, detail, required=True):
        checks[name] = {"pass": bool(ok), "detail": detail, "required": required}

    sg = spec["static_geoms"]
    rec("mesh_loaded_and_processed", proc["steps"]["load"]["faces"] > 0,
        f"{proc['steps']['load']['faces']} raw faces -> {len(sg['boxes'])} boxes + {len(sg['hulls'])} hulls")

    floor = proc["steps"]["floor"]
    frac = floor.get("up_area_fraction", 1.0)
    rec("floor_height_plausible", floor["aligned_within_tol"] and frac >= 0.3,
        f"floor after alignment z = {floor.get('floor_z_after_shift', floor['floor_z']):.4f} m (tol 0.03); raw floor was at {floor.get('floor_z_before_shift', floor['floor_z']):.3f} m "
        f"(shifted {floor.get('shifted_by_m', 0.0):+.3f} m); floor = {frac:.0%} of up-facing area")
    h = proc["room"]["ceiling_height_m"]
    rec("room_dimensions_plausible", 2.0 <= h <= 4.5, f"room {proc['room']['size_xyz_m'][0]:.1f} x {proc['room']['size_xyz_m'][1]:.1f} m, "
        f"height {h:.2f} m (declared dimensions are constraints to verify, not guarantees)", required=False)

    max_boxes = max_boxes or proc.get("limits", {}).get("max_boxes", 3000)
    max_hull_faces = max_hull_faces or proc.get("limits", {}).get("hull_max_faces", 96)
    n_faces = [len(__import__('trimesh').load(h_["file"]).faces) for h_ in sg["hulls"]]
    rec("complexity_within_limits", len(sg["boxes"]) <= max_boxes and (max(n_faces) if n_faces else 0) <= max_hull_faces,
        f"boxes {len(sg['boxes'])}/{max_boxes}, max hull faces {max(n_faces) if n_faces else 0}/{max_hull_faces}")

    zone_hits = []
    for i, b in enumerate(sg["boxes"]):
        lo, hi = np.array(b["pos"]) - np.array(b["half"]), np.array(b["pos"]) + np.array(b["half"])
        if _overlap(lo, hi, TASK_ZONE):
            zone_hits.append(i)
    rec("task_zone_free_of_static_geometry", not zone_hits, f"{len(zone_hits)} Marble boxes intersect the counter workspace {TASK_ZONE}")

    # ---- compile + settle in MuJoCo
    try:
        env = HouseholdEnv(task="pick_up_cup", scene=spec, obs_mode="state")
        env.reset(seed=0)
        rec("mujoco_model_compiles", True, f"{env.env.sim.model._model.ngeom} geoms, {env.env.sim.model._model.nbody} bodies")
    except Exception as e:  # noqa: BLE001
        rec("mujoco_model_compiles", False, f"{type(e).__name__}: {e}")
        provenance.write_json(out / "report.json", {"scene": spec["scene_id"], "checks": checks, "passed": False})
        return {"passed": False, "checks": checks}

    for _ in range(40):
        env.step(np.zeros(4, np.float32))
    e = env.env
    m, d = e.sim.model._model, e.sim.data._data
    # physical penetration: deepest robot contact + robot base height drift + cup resting on the counter
    # (geom bounding spheres are NOT used: they hugely over-estimate for elongated leg/foot meshes)
    pen = [float(d.contact[i].dist) for i in range(d.ncon)
           if int(d.contact[i].geom1) in e._robot_geoms or int(d.contact[i].geom2) in e._robot_geoms]
    deepest = min(pen, default=0.0)
    base_z = float(e.base_pose()[0][2])
    cup_err = abs(e._last_state.cup_pos[2] - e.cup_rest_z)
    rec("no_floor_penetration", deepest > -0.005 and abs(base_z - 0.95) < 0.01 and cup_err < 0.004,
        f"deepest robot contact {deepest * 1000:.1f} mm ({len(pen)} contacts), base z {base_z:.3f} m (nominal 0.95), "
        f"cup rest error {cup_err * 1000:.1f} mm")

    static_ids = {g for g in range(m.ngeom) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(("mb_", "hull_"))}
    bad = set()
    for i in range(d.ncon):
        c = d.contact[i]
        a, b = int(c.geom1), int(c.geom2)
        if (a in e._robot_geoms and b in static_ids) or (b in e._robot_geoms and a in static_ids):
            bad.add((a, b))
    rec("robot_spawn_collision_free", not bad, f"{len(bad)} robot<->Marble-geometry contacts after 40 settle steps")

    # Render BEFORE the reachability probe: the probe creates/closes a second env, which destroys the shared GL context.
    shots = {}
    if render:
        for cam in ("overview", "eval_side", "eval_top", "eval_front3q"):
            try:
                img = env.render(640, 480, camera=cam)
                p = out / f"{cam}.png"
                imageio.imwrite(p, img)
                shots[cam] = str(p)
            except Exception as ex:  # noqa: BLE001
                shots[cam] = f"render failed: {ex}"
        blank = [c for c, pth in shots.items() if not pth.startswith("render failed") and imageio.imread(pth).std() < 5.0]
        must = {"overview", "eval_top"}  # side cameras can legitimately sit inside a wall of the composed room -> advisory
        bad_required = [c for c in blank if c in must] + [c for c, v in shots.items() if c in must and v.startswith("render failed")]
        rec("screenshots_nonblank", not bad_required,
            f"{len(shots)} screenshots rendered; required views overview+top " + ("OK" if not bad_required else f"BLANK/FAILED: {bad_required}")
            + (f"; occluded (advisory): {[c for c in blank if c not in must]}" if [c for c in blank if c not in must] else ""))

    if probe_expert:
        from marble_home.policies.agents import ScriptedAgent
        from marble_home.evaluation.rollout import run_episode

        # Relative test: the scripted expert has its own failures at the edge of the cup-spawn range, so compare against
        # the open procedural kitchen on the SAME seeds. The Marble geometry must not make things worse.
        base_env = HouseholdEnv(task="pick_up_cup", obs_mode="state")
        wins, base_wins, results = 0, 0, []
        for s in range(probe_expert):
            r = run_episode(env, ScriptedAgent(), seed=500 + s)
            rb = run_episode(base_env, ScriptedAgent(), seed=500 + s)
            wins += int(r["success"])
            base_wins += int(rb["success"])
            results.append({"seed": 500 + s, "marble_scene": r["success"], "open_baseline": rb["success"], "invalid_steps": r["invalid_steps"]})
        base_env.close()
        rec("approach_positions_reachable_scripted_pick", wins >= base_wins and wins >= 1,
            f"scripted pick_up_cup: {wins}/{probe_expert} in this scene vs {base_wins}/{probe_expert} in the open procedural kitchen ({results})")
    else:
        rec("approach_positions_reachable_scripted_pick", True, "not run (probe_expert=0)", required=False)

    env.close()

    passed = all(c["pass"] for c in checks.values() if c["required"])
    report = {"scene": spec["scene_id"], "source": spec.get("source"), "passed": passed, "checks": checks, "screenshots": shots,
              "processing_warnings": proc["warnings"], "processing_assumptions": proc["assumptions"]}
    provenance.write_json(out / "report.json", report)
    lines = [f"# Geometry validation: {spec['scene_id']} ({spec.get('source')})", f"**{'PASS' if passed else 'FAIL'}**", ""]
    for n, c in checks.items():
        lines.append(f"- {'PASS' if c['pass'] else 'FAIL'}{'' if c['required'] else ' (advisory)'} `{n}`: {c['detail']}")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    return report
