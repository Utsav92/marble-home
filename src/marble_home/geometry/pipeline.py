"""Import a (downloaded or synthetic) Marble export directory and produce a versioned processed world."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from marble_home.geometry import process as P
from marble_home.utils import provenance
from marble_home.utils.paths import DATA, ensure
from marble_home.worldlabs.manifest import AssetRegistry, SceneRecord, checksum_files

COLLIDER_NAMES = ("collider.glb", "collider_mesh.glb", "mesh.glb")


def read_export_dir(d: Path) -> dict:
    """Locate files + metadata in a raw export dir. Layout documented in docs/marble_import.md."""
    d = Path(d)
    glb = next((d / n for n in COLLIDER_NAMES if (d / n).exists()), None) or next(iter(sorted(d.glob("*.glb"))), None)
    if glb is None:
        raise FileNotFoundError(f"no collision-mesh .glb in {d} (expected one of {COLLIDER_NAMES})")
    world = json.loads((d / "world.json").read_text()) if (d / "world.json").exists() else {}
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    sem = (((world.get("assets") or {}).get("splats") or {}).get("semantics_metadata")) or {}
    return {"glb": glb, "world": world, "meta": meta,
            "scale": meta.get("scale", sem.get("metric_scale_factor")),
            "ground_offset": meta.get("ground_plane_offset", sem.get("ground_plane_offset", 0.0)) or 0.0,
            "up_axis": meta.get("up_axis", "y"), "ply": next(iter(sorted(d.glob("*.ply"))), None),
            "synthetic": bool(world.get("_synthetic"))}


def process_export(input_dir: str | Path, scene_id: str | None = None, category: str = "kitchen", cfg: P.ProcessConfig | None = None,
                   processed_root: Path | None = None, save_visual: bool = True) -> dict:
    t0 = time.time()
    ex = read_export_dir(Path(input_dir))
    scene_id = scene_id or Path(input_dir).name
    cfg = cfg or P.ProcessConfig()
    cfg_in_scale = cfg.scale
    if cfg.scale is None:
        cfg.scale = ex["scale"]
    cfg.ground_plane_offset = ex["ground_offset"] if cfg.ground_plane_offset == 0.0 else cfg.ground_plane_offset
    cfg.up_axis = ex["up_axis"]
    rep = P.ProcessReport()

    raw = P.load_glb(ex["glb"])
    rep.steps["load"] = {"faces": int(len(raw.faces)), "vertices": int(len(raw.vertices)),
                         "raw_extent": (raw.bounds[1] - raw.bounds[0]).tolist()}
    m = P.to_metric_zup(raw, cfg, rep)
    floor_z = P.detect_floor(m, rep, cfg.floor_tol)
    raw_floor = dict(rep.steps["floor"])
    shift = 0.0
    if abs(floor_z) > cfg.floor_tol:
        rep.warnings.append(f"floor at z={floor_z:.3f} m after unit/offset handling (no/incorrect ground_plane_offset); shifting so floor=0")
        m.apply_translation([0, 0, -floor_z])
        shift = -floor_z
    after = P.detect_floor(m, P.ProcessReport(), cfg.floor_tol)
    rep.steps["floor"] = {**raw_floor, "floor_z_before_shift": floor_z, "shifted_by_m": shift, "floor_z_after_shift": after,
                          "aligned_within_tol": abs(after) <= cfg.floor_tol}
    room = P.estimate_room(m)
    m = P.repair(m, rep)
    comps = P.dedupe(P.prune_fragments(P.split_components(m), cfg, rep), rep)
    dec = P.decompose(comps, cfg, rep)
    # the arena already has a physical floor plane: drop floor-slab boxes (top within one voxel of z=0)
    n_before = len(dec["boxes"])
    dec["boxes"] = [b for b in dec["boxes"] if b["pos"][2] + b["half"][2] > cfg.floor_tol + b["pitch"] + 1e-6]
    rep.steps["floor_boxes_dropped"] = n_before - len(dec["boxes"])

    lo, hi = m.bounds
    occ, origin, res = P.occupancy_xy(dec["boxes"], dec["hulls"], ((lo[0], lo[1]), (hi[0], hi[1])))
    placement = P.find_task_placement(occ, origin, res)
    carved = (0, 0)
    if placement is None:  # small / cluttered room: displace the least low furniture instead of failing
        placement = P.find_task_placement_carving(dec["boxes"], dec["hulls"], ((lo[0], lo[1]), (hi[0], hi[1])))
        if placement is None:
            raise RuntimeError("no placement for the robot+counter footprint even when carving low furniture (tall units/walls block every position)")
        rep.warnings.append("no free-floor placement: carved low furniture under the inserted counter/robot footprint")
    boxes_t, hulls_t = P.transform_to_task_frame(dec["boxes"], dec["hulls"], placement["anchor_xy"], placement["yaw_deg"])
    if placement.get("carved"):
        boxes_t, hulls_t, nb, nh = P.carve_footprint(boxes_t, hulls_t)
        carved = (nb, nh)
        rep.steps["carved"] = {"boxes_removed": nb, "hulls_removed": nh}

    reg = AssetRegistry()
    root = Path(processed_root) if processed_root else DATA / "processed_worlds"
    sdir = root / scene_id
    ensure(sdir)
    version = 1 + len([p for p in sdir.glob("v*") if p.is_dir()])
    vdir = sdir / f"v{version}"
    (vdir / "hulls").mkdir(parents=True)
    hull_specs = []
    for h in hulls_t:
        f = vdir / "hulls" / f"{h['name']}.obj"
        h["mesh"].export(f)
        hull_specs.append({"name": h["name"], "file": str(f.resolve())})
    visual = None
    if save_visual and len(m.faces) <= cfg.max_faces_total:
        vm = m.copy()
        _, vhull = P.transform_to_task_frame([], [{"name": "v", "mesh": vm}], placement["anchor_xy"], placement["yaw_deg"])
        vm = vhull[0]["mesh"]
        visual = str((vdir / "visual.obj").resolve())
        vm.export(visual)

    b_lo = np.min([np.array(b["pos"]) - np.array(b["half"]) for b in boxes_t], axis=0) if boxes_t else np.zeros(3)
    b_hi = np.max([np.array(b["pos"]) + np.array(b["half"]) for b in boxes_t], axis=0) if boxes_t else np.zeros(3)
    center = ((b_lo + b_hi) / 2).tolist()
    span = float(max(b_hi[:2] - b_lo[:2], default=4.0))
    spec = {
        "scene_id": scene_id, "kind": category, "source": "marble_import" if not ex["synthetic"] else "synthetic_fixture",
        "decor": {"back_wall": False, "upper_cabinets": False, "dishwasher": False, "dining_table": False, "chairs": 0},
        "static_geoms": {"boxes": [{"pos": b["pos"], "half": b["half"]} for b in boxes_t], "hulls": hull_specs, "visual_mesh": visual},
        # high oblique camera above the room centre (ceiling boxes are not drawn, see kitchen_arena._add_static_geoms)
        "cameras": {"overview": {"pos": [center[0] - 0.9, center[1] - 0.9, max(4.5, span * 1.1 + 1.0)],
                                 "lookat": [center[0], center[1], 0.0]}},
        "marble": {"placement": placement, "room": room, "scale": cfg.scale, "ground_plane_offset": cfg.ground_plane_offset},
    }
    (vdir / "scene_spec.json").write_text(json.dumps(spec, indent=2))
    report = {"scene_id": scene_id, "version": version, "steps": rep.steps, "warnings": rep.warnings, "assumptions": rep.assumptions,
              "placement": placement, "room": room, "synthetic": ex["synthetic"], "seconds": round(time.time() - t0, 2),
              "limits": {"max_boxes": cfg.max_boxes, "hull_max_faces": cfg.hull_max_faces}}
    provenance.write_json(vdir / "report.json", report)

    paths = {"collider_glb": str(ex["glb"]), "splat_ply": str(ex["ply"]) if ex["ply"] else None}
    w = ex["world"]
    rec = SceneRecord(
        scene_id=scene_id, category=category, prompt=((w.get("world_prompt") or {}).get("text_prompt") or ""),
        world_id=w.get("world_id"), model=w.get("model"), api_status="synthetic_fixture" if ex["synthetic"] else "offline_import",
        export_paths={k: v for k, v in paths.items() if v}, checksums=checksum_files(paths), scale=cfg.scale,
        scale_source="fixture" if ex["synthetic"] else ("manual override (CLI)" if (cfg_in_scale is not None and cfg_in_scale != ex["scale"]) else ("api/meta" if ex["scale"] else "assumed 1.0 (API metric_scale_factor was null)")),
        ground_plane_offset=cfg.ground_plane_offset,
        complexity={"raw_faces": rep.steps["load"]["faces"], "boxes": len(boxes_t), "hulls": len(hulls_t), "components": rep.steps["fragments"]["kept"]},
        notes=rep.warnings + rep.assumptions)
    reg.register(rec)
    return {"spec_path": str(vdir / "scene_spec.json"), "spec": spec, "report": report, "version_dir": str(vdir)}
