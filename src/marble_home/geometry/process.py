"""Marble collision-mesh (GLB) -> MuJoCo-compatible static collision geometry.

Pipeline (spec section 5): load -> metric Z-up -> floor -> repair -> prune isolated fragments -> dedupe
-> split -> collision decomposition (voxel-merged AABBs for large/thin structure, convex hulls for compact
objects) -> placement search for the interactive task area.

IMPORTANT: MuJoCo collides a ``mesh`` geom as its CONVEX HULL, so a concave room cannot be passed as one
mesh. Large/thin structure is therefore approximated by axis-aligned boxes (conservative: up to one voxel
of inflation), compact components by hulls.  Gaussian splats are never used for physics.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh

# Robot (base at x=-0.5, torso ~0.3 m radius) + 0.7 x 1.2 m counter at the origin, in the task frame (x toward the counter)
TASK_FOOTPRINT = ((-0.85, 0.40), (-0.65, 0.65))

# glTF is +Y up.  MuJoCo here is +Z up.  (x, y, z)_gltf -> (x, -z, y)_mujoco  (rotation of +90 deg about X)
YUP_TO_ZUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)


@dataclass
class ProcessConfig:
    scale: float | None = None            # raw units -> metres (World API: semantics_metadata.metric_scale_factor)
    ground_plane_offset: float = 0.0      # World API: Y offset (m, AFTER scale) to subtract so ground sits at y=0
    up_axis: str = "y"
    min_component_area: float = 0.03      # m^2; shards with less surface area are dropped (face count is NOT a criterion:
    min_component_extent: float = 0.12    # m; a 12-face wall is legitimate) ... nor are components with a tiny largest side
    voxel: float = 0.05
    max_boxes: int = 3000                 # real scenes are dense; 600 forced an 11 cm pitch on a real Marble room
    hull_extent_max: float = 0.9          # components smaller than this (all sides) may use a convex hull
    hull_max_faces: int = 96
    floor_tol: float = 0.03
    max_faces_total: int = 400_000


@dataclass
class ProcessReport:
    steps: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    assumptions: list = field(default_factory=list)


def load_glb(path) -> trimesh.Trimesh:
    sc = trimesh.load(path, force="scene", process=False)
    meshes = []
    for name, geom in sc.geometry.items():
        if not isinstance(geom, trimesh.Trimesh):
            continue
        for node in sc.graph.nodes_geometry:
            T, gname = sc.graph[node]
            if gname == name:
                g = geom.copy()
                g.apply_transform(T)
                meshes.append(g)
    if not meshes:
        raise ValueError(f"{path}: no triangle geometry found")
    return trimesh.util.concatenate(meshes)


def to_metric_zup(mesh: trimesh.Trimesh, cfg: ProcessConfig, rep: ProcessReport) -> trimesh.Trimesh:
    m = mesh.copy()
    v = np.asarray(m.vertices, dtype=float)
    if cfg.scale is None:
        # No metric_scale_factor available -> heuristic: a room's height should be ~2.4-3.2 m.
        ext = v.max(0) - v.min(0)
        h = ext[1] if cfg.up_axis == "y" else ext[2]
        cfg.scale = 1.0
        rep.assumptions.append("scale unknown (no metric_scale_factor): assumed 1.0 (metres). "
                               f"Extent along up axis = {h:.2f} m -> {'plausible' if 1.8 < h < 6 else 'IMPLAUSIBLE, set scale manually'}")
    v = v * cfg.scale
    if cfg.up_axis == "y":
        v[:, 1] -= cfg.ground_plane_offset
        v = v @ YUP_TO_ZUP.T
    else:
        v[:, 2] -= cfg.ground_plane_offset
    m.vertices = v
    rep.steps["units_frame"] = {"scale": cfg.scale, "ground_plane_offset": cfg.ground_plane_offset, "up_axis_in": cfg.up_axis}
    return m


def detect_floor(mesh: trimesh.Trimesh, rep: ProcessReport, tol: float = 0.03) -> float:
    """Height of the dominant low upward-facing surface (area-weighted histogram of up-facing triangles)."""
    up = mesh.face_normals[:, 2] > 0.9
    if not up.any():
        rep.warnings.append("no upward-facing faces: floor not detected")
        return float(mesh.bounds[0, 2])
    z = mesh.triangles_center[up, 2]
    a = mesh.area_faces[up]
    bins = np.arange(z.min() - 0.01, z.max() + 0.03, 0.02)
    hist, edges = np.histogram(z, bins=bins, weights=a)
    thresh = 0.25 * hist.max()
    idx = int(np.argmax(hist >= thresh))  # lowest bin with substantial area
    floor_z = float(edges[idx] + 0.01)
    rep.steps["floor"] = {"floor_z": floor_z, "area_at_floor_m2": float(hist[idx]), "up_area_fraction": float(a[np.abs(z - floor_z) < 0.10].sum() / max(a.sum(), 1e-9)),  # up-facing area within +-10 cm of the floor
                          "aligned_within_tol": abs(floor_z) <= tol}
    return floor_z


def estimate_room(mesh: trimesh.Trimesh) -> dict:
    lo, hi = mesh.bounds
    return {"aabb_min": lo.tolist(), "aabb_max": hi.tolist(), "size_xyz_m": (hi - lo).tolist(),
            "ceiling_height_m": float(hi[2]), "footprint_m2": float((hi[0] - lo[0]) * (hi[1] - lo[1]))}


def repair(mesh: trimesh.Trimesh, rep: ProcessReport) -> trimesh.Trimesh:
    m = mesh.copy()
    n0 = len(m.faces)
    m.merge_vertices()                      # first: duplicates only become detectable once vertices coincide
    m.update_faces(m.nondegenerate_faces())
    m.update_faces(m.unique_faces())
    m.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(m)
    rep.steps["repair"] = {"faces_before": n0, "faces_after": len(m.faces), "watertight": bool(m.is_watertight)}
    return m


def split_components(mesh: trimesh.Trimesh) -> list[trimesh.Trimesh]:
    """Components by VERTEX connectivity. (Edge-adjacency splitting shatters meshes where separate objects touch,
    because merged corners create non-manifold edges that trimesh excludes from face adjacency.)"""
    labels = trimesh.graph.connected_component_labels(mesh.edges_unique, node_count=len(mesh.vertices))
    face_labels = labels[mesh.faces[:, 0]]
    return [mesh.submesh([np.where(face_labels == lab)[0]], append=True) for lab in np.unique(face_labels)]


def prune_fragments(comps, cfg: ProcessConfig, rep: ProcessReport):
    kept, dropped = [], []
    for c in comps:
        ext = float((c.bounds[1] - c.bounds[0]).max())
        (kept if (c.area >= cfg.min_component_area and ext >= cfg.min_component_extent) else dropped).append(c)
    rep.steps["fragments"] = {"components_in": len(comps), "kept": len(kept), "dropped": len(dropped),
                              "dropped_faces": int(sum(len(c.faces) for c in dropped))}
    return kept


def dedupe(comps, rep: ProcessReport):
    seen, out = set(), []
    for c in comps:
        key = (len(c.faces), tuple(np.round(c.bounds.ravel(), 3)), round(float(c.area), 3))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    rep.steps["duplicates_removed"] = len(comps) - len(out)
    return out


# ------------------------------------------------------------------------------------ decomposition
def _greedy_boxes(occ: np.ndarray) -> list[tuple[int, int, int, int, int, int]]:
    """Greedy maximal-box cover of a boolean 3-D occupancy grid. Returns inclusive index boxes (x0,y0,z0,x1,y1,z1)."""
    occ = occ.copy()
    nx, ny, nz = occ.shape
    boxes = []
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                if not occ[x, y, z]:
                    continue
                x1 = x
                while x1 + 1 < nx and occ[x1 + 1, y, z]:
                    x1 += 1
                y1 = y
                while y1 + 1 < ny and occ[x:x1 + 1, y1 + 1, z].all():
                    y1 += 1
                z1 = z
                while z1 + 1 < nz and occ[x:x1 + 1, y:y1 + 1, z1 + 1].all():
                    z1 += 1
                occ[x:x1 + 1, y:y1 + 1, z:z1 + 1] = False
                boxes.append((x, y, z, x1, y1, z1))
    return boxes


def voxel_boxes(mesh: trimesh.Trimesh, pitch: float, max_boxes: int) -> tuple[list[dict], float]:
    """AABB cover of the mesh surface/volume at the coarsest pitch (growing x1.5) that stays under ``max_boxes``."""
    p = pitch
    for _ in range(6):
        # SURFACE voxels only. Filling a watertight room shell would turn the (empty) interior of the room into a solid block;
        # a 5 cm collision shell is enough for walls and furniture (found on a real closed Marble bedroom mesh).
        vox = mesh.voxelized(p)
        occ = np.asarray(vox.matrix, dtype=bool)
        origin = np.asarray(vox.transform)[:3, 3]  # center of voxel (0,0,0)
        idx = _greedy_boxes(occ)
        if len(idx) <= max_boxes:
            out = []
            for x0, y0, z0, x1, y1, z1 in idx:
                lo = origin + (np.array([x0, y0, z0]) - 0.5) * p
                hi = origin + (np.array([x1, y1, z1]) + 0.5) * p
                out.append({"pos": ((lo + hi) / 2).tolist(), "half": ((hi - lo) / 2).tolist()})
            return out, p
        p *= 1.5
    raise RuntimeError(f"could not cover mesh with <= {max_boxes} boxes even at pitch {p:.2f} m")


def decompose(comps, cfg: ProcessConfig, rep: ProcessReport) -> dict:
    boxes, hulls = [], []
    for i, c in enumerate(comps):
        ext = c.bounds[1] - c.bounds[0]
        compact = ext.max() <= cfg.hull_extent_max
        if compact:
            h = c.convex_hull
            if len(h.faces) <= cfg.hull_max_faces:
                hulls.append({"name": f"hull_{i:03d}", "mesh": h})
                continue
        bx, p = voxel_boxes(c, cfg.voxel, max(8, cfg.max_boxes - len(boxes)))
        for b in bx:
            b["pitch"] = p
        boxes += bx
    rep.steps["decomposition"] = {"boxes": len(boxes), "hulls": len(hulls),
                                  "max_voxel_inflation_m": max([b["pitch"] for b in boxes], default=0.0)}
    return {"boxes": boxes, "hulls": hulls}


# ------------------------------------------------------------------------------------ placement search
def occupancy_xy(boxes: list[dict], hulls, bounds_xy, res: float = 0.05, z_lo=0.12, z_hi=1.9):
    """2-D occupancy of everything between z_lo..z_hi (what a standing robot / counter would hit)."""
    (x0, y0), (x1, y1) = bounds_xy
    nx, ny = int(np.ceil((x1 - x0) / res)), int(np.ceil((y1 - y0) / res))
    occ = np.zeros((nx, ny), bool)

    def mark(lo, hi):
        if hi[2] < z_lo or lo[2] > z_hi:
            return
        i0, i1 = int((lo[0] - x0) // res), int((hi[0] - x0) // res)
        j0, j1 = int((lo[1] - y0) // res), int((hi[1] - y0) // res)
        occ[max(i0, 0):max(i1 + 1, 0), max(j0, 0):max(j1 + 1, 0)] = True

    for b in boxes:
        c, h = np.array(b["pos"]), np.array(b["half"])
        mark(c - h, c + h)
    for hh in hulls:
        mark(hh["mesh"].bounds[0], hh["mesh"].bounds[1])
    return occ, (x0, y0), res


def find_task_placement(occ, origin, res, footprint=TASK_FOOTPRINT, margin=0.10, room_bounds=None):
    """Best rigid placement (anchor xy + yaw in 90-degree steps) of the task footprint (robot + counter) in free floor.

    footprint = ((xmin,xmax),(ymin,ymax)) in the TASK frame (x toward the counter, robot behind at x<0).
    Returns dict(anchor_xy, yaw_deg, clearance_m) or None.
    """
    from scipy.ndimage import distance_transform_edt

    free_dist = distance_transform_edt(~occ) * res  # distance to nearest obstacle (m)
    nx, ny = occ.shape
    best = None
    (fx0, fx1), (fy0, fy1) = footprint
    cx, cy = nx * res / 2, ny * res / 2
    for yaw in (0, 90, 180, 270):
        c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
        corners = np.array([[fx0, fy0], [fx1, fy0], [fx1, fy1], [fx0, fy1]]) @ np.array([[c, s], [-s, c]])
        half = np.abs(corners).max(0) + margin
        # whole footprint AABB (in world) must be obstacle-free
        for i in range(int(half[0] / res) + 1, nx - int(half[0] / res) - 1, 2):
            for j in range(int(half[1] / res) + 1, ny - int(half[1] / res) - 1, 2):
                wx, wy = origin[0] + (i + 0.5) * res, origin[1] + (j + 0.5) * res
                # rotate footprint around its anchor; check by sampling the rectangle
                xs = np.linspace(fx0 - margin, fx1 + margin, 14)
                ys = np.linspace(fy0 - margin, fy1 + margin, 18)
                gx, gy = np.meshgrid(xs, ys)
                px = wx + c * gx - s * gy
                py = wy + s * gx + c * gy
                ii = ((px - origin[0]) // res).astype(int)
                jj = ((py - origin[1]) // res).astype(int)
                if ii.min() < 0 or jj.min() < 0 or ii.max() >= nx or jj.max() >= ny or occ[ii, jj].any():
                    continue
                score = -np.hypot(wx - (origin[0] + cx), wy - (origin[1] + cy)) + 0.5 * float(free_dist[i, j])
                if best is None or score > best[0]:
                    best = (score, wx, wy, yaw, float(free_dist[i, j]))
    if best is None:
        return None
    return {"anchor_xy": [best[1], best[2]], "yaw_deg": best[3], "clearance_m": best[4]}


def transform_to_task_frame(boxes, hulls, anchor_xy, yaw_deg):
    """Marble frame -> env frame (counter at origin). Yaw must be a multiple of 90 degrees so boxes stay axis-aligned."""
    assert yaw_deg % 90 == 0
    th = np.radians(-yaw_deg)  # inverse of the placement yaw
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    R = np.round(R)
    out_b = []
    for b in boxes:
        p = np.array(b["pos"])
        h = np.array(b["half"])
        xy = R @ (p[:2] - np.asarray(anchor_xy))
        hxy = np.abs(R) @ h[:2]
        out_b.append({"pos": [float(xy[0]), float(xy[1]), float(p[2])], "half": [float(hxy[0]), float(hxy[1]), float(h[2])]})
    out_h = []
    for hh in hulls:
        m = hh["mesh"].copy()
        v = np.asarray(m.vertices)
        xy = (R @ (v[:, :2] - np.asarray(anchor_xy)).T).T
        m.vertices = np.column_stack([xy, v[:, 2]])
        if np.linalg.det(R) < 0:
            m.invert()
        out_h.append({"name": hh["name"], "mesh": m})
    return out_b, out_h


def find_task_placement_carving(boxes, hulls, bounds_xy, footprint=TASK_FOOTPRINT, res=0.05, tall_z=1.7):
    """Fallback for small rooms: choose the placement that displaces the LEAST low furniture.

    Boxes reaching above ``tall_z`` (walls, tall units) are never carved; the footprint may only overlap low boxes, which
    the inserted counter/robot area replaces. Returns dict(anchor_xy, yaw_deg, carved_boxes=int, occupied_cells=int) or None.
    """
    low = [b for b in boxes if b["pos"][2] + b["half"][2] < tall_z]
    tall = [b for b in boxes if b["pos"][2] + b["half"][2] >= tall_z]
    occ_low, origin, _ = occupancy_xy(low, [h for h in hulls if h["mesh"].bounds[1][2] < tall_z], bounds_xy, res)
    occ_tall, _, _ = occupancy_xy(tall, [h for h in hulls if h["mesh"].bounds[1][2] >= tall_z], bounds_xy, res)
    nx, ny = occ_low.shape
    (fx0, fx1), (fy0, fy1) = footprint
    best = None
    xs, ys = np.linspace(fx0 - 0.1, fx1 + 0.1, 20), np.linspace(fy0 - 0.1, fy1 + 0.1, 24)
    gx, gy = np.meshgrid(xs, ys)
    for yaw in (0, 90, 180, 270):
        c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
        for i in range(2, nx - 2, 2):
            for j in range(2, ny - 2, 2):
                wx, wy = origin[0] + (i + 0.5) * res, origin[1] + (j + 0.5) * res
                ii = ((wx + c * gx - s * gy - origin[0]) // res).astype(int)
                jj = ((wy + s * gx + c * gy - origin[1]) // res).astype(int)
                if ii.min() < 0 or jj.min() < 0 or ii.max() >= nx or jj.max() >= ny or occ_tall[ii, jj].any():
                    continue
                cost = int(occ_low[ii, jj].sum())
                if best is None or cost < best[0]:
                    best = (cost, wx, wy, yaw)
    if best is None:
        return None
    return {"anchor_xy": [best[1], best[2]], "yaw_deg": best[3], "occupied_cells": best[0], "carved": True}


def carve_footprint(boxes, hulls, footprint=((-0.90, 0.45), (-0.70, 0.70)), tall_z=1.7):
    """In the TASK frame: drop low boxes/hulls that intersect the footprint (replaced by the inserted counter + robot)."""
    (fx0, fx1), (fy0, fy1) = footprint

    def hit(lo, hi):
        return lo[0] < fx1 and hi[0] > fx0 and lo[1] < fy1 and hi[1] > fy0 and hi[2] < tall_z

    kb = [b for b in boxes if not hit(np.array(b["pos"]) - np.array(b["half"]), np.array(b["pos"]) + np.array(b["half"]))]
    kh = [h for h in hulls if not hit(h["mesh"].bounds[0], h["mesh"].bounds[1])]
    return kb, kh, len(boxes) - len(kb), len(hulls) - len(kh)
