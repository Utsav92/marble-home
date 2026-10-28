import json

import numpy as np
import pytest
import trimesh

from marble_home.geometry import process as P
from marble_home.worldlabs.client import (InsufficientCredits, MissingApiKey, WorldLabsClient, WorldLabsError)


# ----------------------------------------------------------------------------- geometry
def test_fixture_unit_and_frame_conversion_puts_floor_at_zero(fixture_export):
    raw = P.load_glb(fixture_export / "collider.glb")
    cfg = P.ProcessConfig(scale=1.25, ground_plane_offset=0.30)
    rep = P.ProcessReport()
    m = P.to_metric_zup(raw, cfg, rep)
    assert abs(P.detect_floor(m, rep)) < 0.01  # fixture built with a known scale/offset: floor must land at z=0
    room = P.estimate_room(m)
    assert room["ceiling_height_m"] == pytest.approx(2.7, abs=0.05)  # +Y-up GLB -> +Z-up metres with scale applied


def test_missing_scale_is_flagged_not_silent(fixture_export):
    raw = P.load_glb(fixture_export / "collider.glb")
    rep = P.ProcessReport()
    P.to_metric_zup(raw, P.ProcessConfig(scale=None), rep)
    assert rep.assumptions and "scale unknown" in rep.assumptions[0]


def test_repair_removes_duplicate_faces_and_fragments_are_pruned(fixture_export):
    raw = P.load_glb(fixture_export / "collider.glb")
    cfg, rep = P.ProcessConfig(scale=1.25, ground_plane_offset=0.30), P.ProcessReport()
    m = P.repair(P.to_metric_zup(raw, cfg, rep), rep)
    assert rep.steps["repair"]["faces_after"] < rep.steps["repair"]["faces_before"]  # the duplicated chair
    kept = P.prune_fragments(P.split_components(m), cfg, rep)
    assert rep.steps["fragments"]["dropped"] >= 6  # the six floating shards
    assert all(float((c.bounds[1] - c.bounds[0]).max()) >= cfg.min_component_extent for c in kept)


def test_touching_objects_do_not_shatter():
    a = trimesh.creation.box(extents=[1, 1, 1])
    b = trimesh.creation.box(extents=[1, 1, 1])
    b.apply_translation([1, 0, 0])  # shares a whole face with a
    m = trimesh.util.concatenate([a, b])
    m.merge_vertices()
    assert len(P.split_components(m)) == 1


def test_voxel_boxes_cover_a_thin_wall_conservatively():
    wall = trimesh.creation.box(extents=[4.0, 0.1, 2.5])
    wall.apply_translation([0, 0, 1.25])
    boxes, pitch = P.voxel_boxes(wall, 0.05, 200)
    lo = np.min([np.array(b["pos"]) - np.array(b["half"]) for b in boxes], 0)
    hi = np.max([np.array(b["pos"]) + np.array(b["half"]) for b in boxes], 0)
    assert np.all(lo <= wall.bounds[0] + 1e-6) and np.all(hi >= wall.bounds[1] - 1e-6)  # never smaller than the wall
    assert np.all(hi - lo <= (wall.bounds[1] - wall.bounds[0]) + 2 * pitch + 1e-6)     # inflated by at most ~1 voxel/side


def test_box_budget_enforced():
    big = trimesh.creation.icosphere(subdivisions=3, radius=2.0)
    boxes, pitch = P.voxel_boxes(big, 0.05, 150)
    assert len(boxes) <= 150 and pitch >= 0.05


def test_placement_search_finds_free_floor_and_refuses_full_rooms():
    occ = np.zeros((120, 100), bool)
    occ[:, :3] = occ[:, -3:] = occ[:3, :] = occ[-3:, :] = True  # perimeter walls
    p = P.find_task_placement(occ, (0.0, 0.0), 0.05)
    assert p is not None and p["clearance_m"] > 0.3
    assert P.find_task_placement(np.ones((120, 100), bool), (0.0, 0.0), 0.05) is None


def test_task_frame_transform_rotates_boxes_by_yaw():
    boxes = [{"pos": [3.0, 1.0, 0.5], "half": [0.5, 0.1, 0.5]}]
    out, _ = P.transform_to_task_frame(boxes, [], anchor_xy=[1.0, 1.0], yaw_deg=90)
    # a point 2 m along +x of the anchor lies along -y... in the 90-degree-rotated task frame: |offset| preserved, half-extents swapped
    assert np.linalg.norm(out[0]["pos"][:2]) == pytest.approx(2.0, abs=1e-6)
    assert out[0]["half"][0] == pytest.approx(0.1) and out[0]["half"][1] == pytest.approx(0.5)


def test_pipeline_outputs_and_registry(processed_fixture):
    spec, rep = processed_fixture["spec"], processed_fixture["report"]
    assert rep["synthetic"] and spec["source"] == "synthetic_fixture"
    assert rep["steps"]["floor"]["aligned_within_tol"] and rep["placement"]["clearance_m"] > 0.2
    assert 0 < len(spec["static_geoms"]["boxes"]) <= 600 and spec["decor"]["back_wall"] is False
    assert (processed_fixture["version_dir"] + "/scene_spec.json")


@pytest.mark.slow
def test_marble_derived_scene_loads_steps_and_expert_picks(processed_fixture):
    """Milestone 3 acceptance (on the SYNTHETIC fixture): compile, step, collision-free spawn, task still solvable."""
    from marble_home.geometry.validate import validate_scene

    rep = validate_scene(processed_fixture["version_dir"], processed_fixture["version_dir"] + "/../_val", probe_expert=2, render=False)
    failed = {k: v["detail"] for k, v in rep["checks"].items() if v["required"] and not v["pass"]}
    assert rep["passed"], failed


# ----------------------------------------------------------------------------- World API client (no network)
class FakeResp:
    def __init__(self, status=200, body=None, chunks=(b"abc",)):
        self.status_code, self._b, self._c, self.text = status, body or {}, chunks, json.dumps(body or {})

    def json(self):
        return self._b

    def iter_content(self, n):
        yield from self._c

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def request(self, method, url, headers=None, json=None, timeout=None):
        self.calls.append((method, url, headers, json))
        return self.responses.pop(0)

    def get(self, url, stream=False, timeout=None):
        self.calls.append(("GET", url, None, None))
        return self.responses.pop(0)


def test_generate_request_schema_matches_official_reference():
    body = WorldLabsClient.build_generate_request("a kitchen", model="marble-1.1", seed=42, display_name="k", tags=["x"])
    assert body == {"model": "marble-1.1", "permission": {"public": False}, "seed": 42, "display_name": "k", "tags": ["x"],
                    "world_prompt": {"type": "text", "text_prompt": "a kitchen"}}
    with pytest.raises(ValueError):
        WorldLabsClient.build_generate_request("x", model="made-up-model")
    with pytest.raises(ValueError):
        WorldLabsClient.build_generate_request("x", seed=2**32)


def test_endpoints_auth_header_and_polling():
    s = FakeSession([FakeResp(200, {"operation_id": "op1", "done": False}),
                     FakeResp(200, {"done": False}), FakeResp(200, {"done": True, "response": {"world_id": "w1"}}),
                     FakeResp(200, {"world_id": "w1", "assets": {"mesh": {"collider_mesh_url": "https://x/y.glb"}}}),
                     FakeResp(200, {"operation_id": "op2", "done": True, "response": {"url": "https://x/s.ply"}})])
    c = WorldLabsClient(api_key="SECRET-KEY", base="https://api.example", session=s)
    assert c.generate("p")["operation_id"] == "op1"
    op = c.wait("op1", poll=0.0)
    assert op["response"]["world_id"] == "w1"
    assert c.get_world("w1")["assets"]["mesh"]["collider_mesh_url"].endswith(".glb")
    c.export("w1", "splats", "ply", resolution="500k")
    urls = [u for _, u, _, _ in s.calls]
    assert urls[0] == "https://api.example/marble/v1/worlds:generate"
    assert urls[1].endswith("/marble/v1/operations/op1")
    assert urls[3].endswith("/marble/v1/worlds/w1")
    assert urls[4].endswith("/marble/v1/worlds/w1:export")
    assert all(h["WLT-Api-Key"] == "SECRET-KEY" for _, _, h, _ in s.calls if h)
    assert s.calls[4][3] == {"asset_type": "splats", "format": "ply", "resolution": "500k"}


def test_missing_key_and_credit_errors_do_not_leak_key(monkeypatch):
    monkeypatch.delenv("WORLDLABS_API_KEY", raising=False)
    with pytest.raises(MissingApiKey):
        WorldLabsClient(session=FakeSession([])).generate("p")
    c = WorldLabsClient(api_key="SECRET-KEY", session=FakeSession([FakeResp(402, {"detail": "no credits"})]))
    with pytest.raises(InsufficientCredits) as e:
        c.generate("p")
    assert "SECRET-KEY" not in str(e.value)
    c = WorldLabsClient(api_key="SECRET-KEY", session=FakeSession([FakeResp(500, {})]))
    with pytest.raises(WorldLabsError) as e:
        c.get_world("w")
    assert "SECRET-KEY" not in str(e.value)


def test_failed_operation_raises_and_download_checksums(tmp_path):
    c = WorldLabsClient(api_key="k", session=FakeSession([FakeResp(200, {"done": True, "error": {"code": 3, "message": "bad"}})]))
    with pytest.raises(WorldLabsError):
        c.wait("op", poll=0.0)
    c = WorldLabsClient(api_key="k", session=FakeSession([FakeResp(200, chunks=(b"hello ", b"world"))]))
    r = c.download("https://x/f.glb", tmp_path / "f.glb")
    import hashlib
    assert r["bytes"] == 11 and r["sha256"] == hashlib.sha256(b"hello world").hexdigest()


@pytest.mark.network
def test_live_generate_is_opt_in():  # pragma: no cover - spends credits; run with: pytest -m network
    import os

    if not os.environ.get("WORLDLABS_API_KEY"):
        pytest.skip("WORLDLABS_API_KEY not set")
    c = WorldLabsClient()
    op = c.generate("A small empty room with a table.", model="marble-1.0-draft", display_name="marble-home-live-test")
    assert op["operation_id"]


def test_closed_room_shell_interior_is_not_filled():
    """Regression: a watertight room shell (walls+floor+ceiling) must stay hollow, not become a solid block."""
    outer = trimesh.creation.box(extents=[4.0, 4.0, 2.6])
    inner = trimesh.creation.box(extents=[3.8, 3.8, 2.4])
    inner.invert()  # inward-facing cavity surface -> closed, hollow shell
    room = trimesh.util.concatenate([outer, inner])
    room.merge_vertices()
    assert room.is_watertight
    boxes, pitch = P.voxel_boxes(room, 0.05, 3000)
    centre = np.array([0.0, 0.0, 0.0])
    for b in boxes:
        lo, hi = np.array(b["pos"]) - np.array(b["half"]), np.array(b["pos"]) + np.array(b["half"])
        assert not (np.all(centre > lo) and np.all(centre < hi)), "room interior was filled with collision boxes"
    vol = sum(8 * np.prod(b["half"]) for b in boxes)
    assert vol < 0.5 * 4.0 * 4.0 * 2.6  # a shell, far less than the solid volume
