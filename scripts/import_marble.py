"""Import a downloaded Marble export directory -> processed MuJoCo-ready scene (versioned).

    python scripts/import_marble.py --input data/marble_raw/kitchen_001
    python scripts/import_marble.py --make-fixture kitchen_fixture_001     # SYNTHETIC stand-in (no Marble account needed)

Expected raw layout (see docs/marble_import.md): collider.glb  [world.json]  [meta.json]  [*.ply]
Originals are never modified. Output: data/processed_worlds/<scene_id>/v<N>/{scene_spec.json,report.json,hulls/,visual.obj}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from marble_home.geometry.fixtures import write_fixture
from marble_home.geometry.pipeline import process_export
from marble_home.geometry.process import ProcessConfig
from marble_home.utils.paths import DATA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="raw export directory")
    ap.add_argument("--make-fixture", metavar="SCENE_ID", help="write a SYNTHETIC export to data/marble_raw/<id> first")
    ap.add_argument("--scene-id", default=None)
    ap.add_argument("--category", default="kitchen")
    ap.add_argument("--scale", type=float, default=None, help="override metric scale (raw units -> metres)")
    ap.add_argument("--voxel", type=float, default=0.05)
    args = ap.parse_args()
    if args.make_fixture:
        inp = write_fixture(DATA / "marble_raw" / args.make_fixture, scene_id=args.make_fixture)
        print("wrote SYNTHETIC fixture:", inp)
    elif args.input:
        inp = Path(args.input)
    else:
        ap.error("--input or --make-fixture required")
    cfg = ProcessConfig(scale=args.scale, voxel=args.voxel)
    out = process_export(inp, scene_id=args.scene_id or Path(inp).name, category=args.category, cfg=cfg)
    rep = out["report"]
    print(json.dumps({"scene_spec": out["spec_path"], "steps": rep["steps"], "placement": rep["placement"],
                      "warnings": rep["warnings"], "assumptions": rep["assumptions"]}, indent=1)[:3000])


if __name__ == "__main__":
    main()
