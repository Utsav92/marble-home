"""Dump one still per scene (at ~60% of its length) into video/_preview/ to check layout before the full render."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compose as C  # noqa: E402

(C.HERE / "_preview").mkdir(exist_ok=True)
dur = C.jl(C.AUD / "durations.json")["durations"]
ctx = {"title_bg": C.Seq(C.FOOT / "world_marble_kitchen_001.mp4"),
       "worlds": [C.Seq(C.FOOT / f"world_{n}.mp4") for n in ("marble_kitchen_001", "marble_living_room_001", "marble_bedroom_001",
                                                           "marble_laundry_room_001", "marble_multiroom_apartment_001")],
       "meta": {}, "seq": {}}
for nm in ("hero_expert", "hero_bc", "laundry_bc", "dish_bc", "fail_bc", "hgt_single", "hgt_multi"):
    ctx["meta"][nm] = C.meta(nm)
    for cam in ("eval_close", "eval_top"):
        ctx["seq"][f"{nm}_{cam}"] = C.Seq(C.FOOT / f"{nm}_{cam}.mp4")
ctx.update(hero_speed=1.4, tasks_speed={"hero_bc": 2.0, "laundry_bc": 2.0, "dish_bc": 1.4}, fail_speed=1.5, height_speed=2.0,
           failure_lines=C.jl(C.HERE / "failure_lines.json") or [])
which = sys.argv[1:] or [k for k, _, _ in C.SCENES]
for k, f, lb in C.SCENES:
    if k not in which or k not in dur:
        continue
    D = dur[k] + 1.2
    fr = float(sys.argv[0] and 0.6)
    img = f(D * fr, D, ctx)
    if k != "title":
        C.chrome(img, 1, 9, lb, 0.5)
    img.save(C.HERE / "_preview" / f"{k}.png")
    print("saved", k)
