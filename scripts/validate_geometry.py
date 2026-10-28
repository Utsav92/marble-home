"""Validate a processed scene: pass/fail checks + screenshots -> results/geometry/<scene>/{report.json,report.md,*.png}

    python scripts/validate_geometry.py --scene kitchen_fixture_001
"""
from __future__ import annotations

import argparse

from marble_home.geometry.validate import latest_version_dir, validate_scene
from marble_home.utils.paths import RESULTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--probe-expert", type=int, default=3, help="run the scripted pick this many times to test reachability")
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()
    vdir = latest_version_dir(args.scene)
    rep = validate_scene(vdir, RESULTS / "geometry" / args.scene, args.probe_expert, not args.no_render)
    for n, c in rep["checks"].items():
        print(f"{'PASS' if c['pass'] else 'FAIL'}{'' if c['required'] else ' (advisory)'}  {n}: {c['detail']}")
    print("OVERALL:", "PASS" if rep["passed"] else "FAIL", "->", RESULTS / "geometry" / args.scene / "report.md")
    raise SystemExit(0 if rep["passed"] else 1)


if __name__ == "__main__":
    main()
