# Marble integration: what is verified, what is not

## World API (verified against the official reference, 2026-10-24)

Sources: <https://docs.worldlabs.ai/api> and the reference pages for *worlds/generate*, *operations/get*, *worlds/get*,
*worlds/export*. The blog post and case-study pages carry no endpoint details.

| Item | Value |
|---|---|
| Base URL | `https://api.worldlabs.ai` |
| Auth | header `WLT-Api-Key: <key>` (keys from platform.worldlabs.ai) |
| Generate | `POST /marble/v1/worlds:generate`, body `{model, seed?, display_name?, tags?, permission, world_prompt{type:"text", text_prompt}}` → `{operation_id, done, ...}` |
| Models | `marble-1.0-draft`, `marble-1.0`, `marble-1.1` (default), `marble-1.1-plus` (+ deprecated legacy names) |
| Polling | `GET /marble/v1/operations/{operation_id}` → `{done, error, response, metadata}`; `response` is a `World` / `ExportWorldResult` |
| World | `GET /marble/v1/worlds/{world_id}`; `assets.mesh.collider_mesh_url`, `assets.mesh.{full_res,hq}_mesh_url`, `assets.splats.spz_urls`, `assets.splats.semantics_metadata.{metric_scale_factor, ground_plane_offset}`, `assets.imagery.pano_url` |
| Export | `POST /marble/v1/worlds/{world_id}:export`, body `{asset_type: splats|mesh, format: ply|glb, mesh_variant?, resolution?}` |
| Billing | credits; HTTP 402 = insufficient credits. **Prices and rate limits are not documented in the API reference.** |

Not verified by execution: no API key was available, so **no live call has ever been made**. The client is tested against an
injected fake HTTP session that asserts URLs, headers, bodies, polling, 402 handling and that the key never appears in errors.
`semantics_metadata` may be null for worlds generated before Dec 2025 (then the scale is *assumed* and flagged, never silent).

## Offline import layout

```
data/marble_raw/<scene_id>/
    collider.glb        # download of assets.mesh.collider_mesh_url (any *.glb also accepted)
    world.json          # optional: GET /worlds/{id} response (provides scale + ground offset)
    meta.json           # optional overrides: {"scale": 1.0, "ground_plane_offset": 0.0, "up_axis": "y"}
    *.ply               # optional Gaussian splat (kept, checksummed, never used for physics)
```
`python scripts/import_marble.py --input data/marble_raw/<id>` → `data/processed_worlds/<id>/v<N>/`.
Originals are never modified; every import bumps the version and is recorded (checksums, scale source, complexity) in
`data/marble_raw/registry.json`.

## Frame / unit conventions used

glTF is +Y up. The World API says to subtract `ground_plane_offset` from Y **after** applying `metric_scale_factor`
so the ground sits at y=0. The importer does exactly that, then rotates to MuJoCo +Z up. It then *measures* the floor
(area-weighted histogram of up-facing triangles) and reports it; if it is not at z≈0 the scene is shifted and a warning is written.

## What the geometry stage does and does not guarantee

* Conservative: voxel-merged boxes inflate geometry by up to one voxel (5 cm default).
* Compact components (<0.9 m on every side) become convex hulls; everything else becomes boxes. Whole rooms are never convexified.
* Not handled: interior doors/articulation, thin furniture below the voxel pitch, glass, ceilings as visual, non-Manhattan walls
  (boxes are axis-aligned in the Marble frame; placement yaw is restricted to 90° steps to keep them axis-aligned).

## Status

* Client, importer, geometry processing, validation: **VERIFIED live on five real Marble exports** (see "Live results" below) and
  **TESTED offline** on a synthetic fixture (`geometry/fixtures.py`) plus regression tests for the bugs the real meshes exposed.
* Policies have NOT been trained inside Marble scenes; only the scripted expert was run there (reachability check).


## Live results (2026-10-25): five real worlds

One world per category, `marble-1.0-draft`, 230 credits each (1,150 total). Every response had `semantics_metadata: null`, so no
`metric_scale_factor` / `ground_plane_offset` was available and every scale in `data/marble_raw/registry.json` is **assumed**
(`scale_source` says how). Findings from real meshes that the synthetic fixture could not show:

* Real colliders are 170k-400k faces, one giant connected component, non-watertight or closed; the raw floor is not at y=0.
* A **closed** room shell must never be voxel-filled (it turned a bedroom into a solid block). Fixed + regression test
  `test_closed_room_shell_interior_is_not_filled`.
* The box budget must be ~3000 (600 forced an 11 cm pitch).
* Room size matters: the laundry room (prompted as 3 x 3 m) cannot host a humanoid plus counter at that size.
