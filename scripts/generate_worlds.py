"""Plan / generate / fall back for the five household world categories.

    python scripts/generate_worlds.py --config configs/marble/default.yaml                 # dry run (default): prints requests, no credits
    python scripts/generate_worlds.py --config configs/marble/default.yaml --confirm-spend --max-scenes 1   # LIVE (paid)
    python scripts/generate_worlds.py --procedural 18                                      # seeded procedural fallback specs

Live generation needs WORLDLABS_API_KEY and spends credits; it is never started implicitly.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from marble_home.simulation.scene_gen import CATEGORIES, make_scene, split_scenes
from marble_home.utils import provenance
from marble_home.utils.cli import load_cfg
from marble_home.utils.paths import DATA, ensure
from marble_home.worldlabs.client import MissingApiKey, WorldLabsClient, WorldLabsError
from marble_home.worldlabs.manifest import AssetRegistry, SceneRecord, checksum_files


def plan(cfg) -> list[dict]:
    out = []
    for cat in CATEGORIES:
        for i in range(cfg["scenes_per_category"]):
            seed = cfg["seed0"] + len(out)
            prompt = " ".join(str(cfg["categories"][cat]).split()) + " " + " ".join(str(cfg["common_suffix"]).split())
            out.append({"scene_id": f"marble_{cat}_{i + 1:03d}", "category": cat, "seed": seed, "prompt": prompt, "model": cfg["model"]})
    return out


def run_live(items, cfg, max_scenes):
    client = WorldLabsClient()
    reg = AssetRegistry()
    for it in items[:max_scenes]:
        d = DATA / "marble_raw" / it["scene_id"]
        ensure(d)
        rec = SceneRecord(scene_id=it["scene_id"], category=it["category"], prompt=it["prompt"], seed=it["seed"], model=it["model"], api_status="pending")
        t0 = time.time()
        try:
            op = client.generate(it["prompt"], model=it["model"], seed=it["seed"], display_name=it["scene_id"], tags=["marble-home", it["category"]])
            rec.operation_id = op["operation_id"]
            done = client.wait(op["operation_id"])
            world_id = (done.get("response") or {}).get("world_id")
            world = client.get_world(world_id)
            rec.world_id, rec.api_status = world_id, "done"
            rec.cost_credits = (done.get("cost") or {}).get("total_credits")
            (d / "world.json").write_text(json.dumps(world, indent=2))  # no credentials are ever in this response
            sem = ((world.get("assets") or {}).get("splats") or {}).get("semantics_metadata") or {}
            rec.scale, rec.ground_plane_offset = sem.get("metric_scale_factor"), sem.get("ground_plane_offset")
            rec.scale_source = "api" if rec.scale else "unknown"
            mesh = (world.get("assets") or {}).get("mesh") or {}
            paths = {}
            if cfg["export"]["collider_mesh"] and mesh.get("collider_mesh_url"):
                client.download(mesh["collider_mesh_url"], d / "collider.glb")
                paths["collider_glb"] = str(d / "collider.glb")
            if cfg["export"]["splat_ply"]:
                ex = client.export(world_id, "splats", "ply", resolution=cfg["export"]["splat_resolution"])
                ex = ex if ex.get("done") else client.wait(ex["operation_id"])
                url = (ex.get("response") or {}).get("url")
                if url:
                    client.download(url, d / "splat.ply")
                    paths["splat_ply"] = str(d / "splat.ply")
            rec.export_paths, rec.checksums = paths, checksum_files(paths)
            if not paths.get("collider_glb"):
                rec.notes.append("no collider mesh URL in this response (account tier / model may not include it)")
        except (WorldLabsError, MissingApiKey) as e:
            rec.api_status, rec.notes = "failed", [str(e)[:300]]
            print("FAILED", it["scene_id"], e)
        rec.generation_time_s = time.time() - t0
        reg.register(rec)
        print(it["scene_id"], rec.api_status, "world", rec.world_id, "credits", rec.cost_credits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/marble/default.yaml")
    ap.add_argument("--confirm-spend", action="store_true", help="actually call the paid API")
    ap.add_argument("--max-scenes", type=int, default=1, help="safety cap on live generations per invocation")
    ap.add_argument("--categories", nargs="+", default=None, help="only these categories (e.g. living_room bedroom)")
    ap.add_argument("--per-category", type=int, default=None, help="override scenes_per_category")
    ap.add_argument("--model", default=None, help="override the model in the config (e.g. marble-1.0-draft is cheapest)")
    ap.add_argument("--procedural", type=int, default=0, metavar="N", help="write N procedural fallback scene specs instead")
    args = ap.parse_args()

    if args.procedural:
        d = DATA / "processed_worlds"
        ensure(d)
        n = 0
        for k in range(args.procedural):
            sp = make_scene(CATEGORIES[k % 5], k)
            (d / sp["scene_id"] / "v1").mkdir(parents=True, exist_ok=True)
            (d / sp["scene_id"] / "v1" / "scene_spec.json").write_text(json.dumps(sp, indent=2))
            n += 1
        splits = split_scenes()
        provenance.write_json(DATA / "splits" / "scene_splits.json", {k: [s["scene_id"] for s in v] for k, v in splits.items()})
        print(f"wrote {n} procedural scene specs under {d}; scene-level splits -> data/splits/scene_splits.json")
        return

    cfg = load_cfg(args.config)
    if args.model:
        cfg["model"] = args.model
    if args.per_category:
        cfg["scenes_per_category"] = args.per_category
    items = plan(cfg)
    if args.categories:
        items = [it for it in items if it["category"] in args.categories]
    client = WorldLabsClient()
    print(f"{len(items)} planned generations (model {cfg['model']}); API key {'present' if client.has_key else 'ABSENT'}")
    if not args.confirm_spend:
        reg = AssetRegistry()
        for it in items:
            body = WorldLabsClient.build_generate_request(it["prompt"], model=it["model"], seed=it["seed"], display_name=it["scene_id"])
            reg.register(SceneRecord(scene_id=it["scene_id"], category=it["category"], prompt=it["prompt"], seed=it["seed"], model=it["model"], api_status="dry_run"))
            print(f"[dry-run] POST /marble/v1/worlds:generate  scene={it['scene_id']} seed={it['seed']} prompt[:70]={it['prompt'][:70]!r}")
            print("          body keys:", sorted(body))
        print("\nNo credits were spent. To generate for real: set WORLDLABS_API_KEY and add --confirm-spend [--max-scenes N].")
        print("Offline alternative: place an exported GLB under data/marble_raw/<id>/ and run scripts/import_marble.py;")
        print("or use the procedural fallback: python scripts/generate_worlds.py --procedural 18")
        return
    if not client.has_key:
        raise SystemExit("WORLDLABS_API_KEY is not set; refusing to start a live generation")
    run_live(items, cfg, args.max_scenes)


if __name__ == "__main__":
    main()
