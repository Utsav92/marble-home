"""Compose the research-style video from real footage + real result files + the narration audio.

    python video/compose.py [--preview N]   -> results/videos/marble_home_research.mp4   (--preview N: write only every N-th scene frame to a tmp mp4)
Inputs: video/footage/*.mp4|json (captured by capture.py / splat_flyover.py), results/experiments/*.json, results/benchmark/pick_up_cup.json,
video/audio/*.mp3 + durations.json (narration.py). Every number drawn on screen is read from those files at build time.
"""
from __future__ import annotations

import json
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FOOT, AUD = HERE / "footage", HERE / "audio"
EXP = ROOT / "results" / "experiments"
W, H, FPS = 1920, 1080, 30

BG, PANEL, LINE = (10, 13, 10), (21, 26, 21), (52, 62, 52)
GREEN, WHITE, GRAY, DIM = (118, 185, 0), (241, 245, 241), (150, 162, 150), (96, 108, 96)
RED, AMBER, BLUE = (240, 92, 92), (244, 184, 64), (96, 170, 235)


# ------------------------------------------------------------------ drawing helpers
@lru_cache(maxsize=64)
def font(size, head=False, bold=False):
    path = "C:/Windows/Fonts/bahnschrift.ttf" if head else "C:/Windows/Fonts/segoeui.ttf"
    f = ImageFont.truetype(path, size)
    if head:
        try:
            f.set_variation_by_name("Bold" if bold else "Regular")
        except Exception:
            pass
    return f


def text(d, xy, s, size=28, fill=WHITE, head=False, bold=False, anchor="la"):
    d.text(xy, s, font=font(size, head, bold), fill=fill, anchor=anchor)


def fade(c, a):
    return tuple(int(BG[i] + (c[i] - BG[i]) * a) for i in range(3))


def panel(d, box, fill=PANEL, outline=LINE, r=16, width=2):
    d.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=width)


def ease(u):
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


class Seq:
    """Sequential reader with random-forward access (frames are held while the index does not advance)."""

    def __init__(self, path):
        self.path = path
        self.r = imageio.get_reader(path)
        self.n = self.r.count_frames()
        self.i, self.cur = -1, None

    def frame(self, i):
        i = int(min(max(i, 0), self.n - 1))
        if i < self.i:
            self.r.close()
            self.r = imageio.get_reader(self.path)
            self.i, self.cur = -1, None
        while self.i < i:
            self.cur = self.r.get_next_data()
            self.i += 1
        return self.cur


def paste(img, frame, box, radius=12, border=None):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    im = Image.fromarray(frame)
    s = max(w / im.width, h / im.height)  # cover
    im = im.resize((int(im.width * s) + 1, int(im.height * s) + 1), Image.LANCZOS)
    cx, cy = (im.width - w) // 2, (im.height - h) // 2
    im = im.crop((cx, cy, cx + w, cy + h))
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, w - 1, h - 1), radius=radius, fill=255)
    img.paste(im, (x0, y0), mask)
    if border:
        ImageDraw.Draw(img).rounded_rectangle(box, radius=radius, outline=border, width=2)


def chrome(img, idx, total, label, t_global):
    d = ImageDraw.Draw(img)
    d.rectangle((60, 44, 72, 56), fill=GREEN)
    text(d, (86, 36), "marble-home", 26, WHITE, head=True, bold=True)
    text(d, (260, 38), f"{idx:02d} / {total:02d}   {label.upper()}", 22, GRAY, head=True)
    d.rectangle((0, H - 6, int(W * t_global), H), fill=GREEN)


# ------------------------------------------------------------------ data
def jl(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


HH = jl(EXP / "household_tasks.json")["rows"]
TASKS = [("clear_table", "Clear table"), ("laundry_basket", "Laundry basket"), ("load_dishwasher", "Load dishwasher")]


def hh(task, policy):
    return sorted([r for r in HH if r["task"] == task and r["policy"] == policy], key=lambda r: r.get("seed", -1))


def pooled(task):
    g = hh(task, "bc_mlp")
    k = int(round(sum(r["success"] * r["n"] for r in g)))
    n = sum(r["n"] for r in g)
    return k, n


def meta(name):
    return jl(FOOT / f"{name}.json")


# ------------------------------------------------------------------ scenes
def scene_title(t, D, ctx):
    bg = ctx["title_bg"].frame(int(min(t / D, 0.999) * 80))
    img = Image.new("RGB", (W, H), BG)
    paste(img, bg, (0, 0, W, H), radius=0)
    arr = np.asarray(img).astype(np.float32)
    grad = np.linspace(0.92, 0.35, W)[None, :, None]
    img = Image.fromarray((arr * (1 - grad) * 1.15).clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    a = ease(t / 1.2)
    d.rectangle((120, 330, 130 + int(260 * a), 336), fill=GREEN)
    text(d, (120, 360), "marble-home", 128, fade(WHITE, a), head=True, bold=True)
    text(d, (124, 520), "Learning household manipulation for a humanoid", 46, fade(WHITE, ease((t - 0.4) / 1.2)), head=True)
    text(d, (124, 580), "in generated worlds", 46, fade(WHITE, ease((t - 0.6) / 1.2)), head=True)
    text(d, (124, 700), "Fourier GR1  |  MuJoCo + RoboSuite  |  World Labs Marble  |  CPU-only", 28, fade(GRAY, ease((t - 1.2) / 1.2)))
    text(d, (124, 748), "Every number on screen is read from the result files.", 28, fade(GREEN, ease((t - 1.8) / 1.2)))
    return img


def scene_worlds(t, D, ctx):
    names = ["marble_kitchen_001", "marble_living_room_001", "marble_bedroom_001", "marble_laundry_room_001", "marble_multiroom_apartment_001"]
    labels = ["Kitchen", "Living room", "Bedroom", "Laundry room", "Apartment"]
    k = min(int(t / D * 5), 4)
    u = (t / D * 5) - k
    fr = ctx["worlds"][k].frame(int(u * 89))
    img = Image.new("RGB", (W, H), BG)
    paste(img, fr, (60, 110, W - 60, H - 150), radius=18, border=LINE)
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(ov).rounded_rectangle((60, H - 276, 900, H - 152), radius=0, fill=(8, 10, 8, 175))
    img = Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB")
    d = ImageDraw.Draw(img)
    text(d, (96, H - 262), f"0{k + 1}  {labels[k]}", 54, WHITE, head=True, bold=True)
    text(d, (98, H - 196), "real Marble splat file (497,664 splats), drawn as points", 26, GRAY)
    for i, lb in enumerate(labels):
        x = 60 + i * 360
        d.rectangle((x, H - 112, x + 336, H - 106), fill=GREEN if i == k else LINE)
        text(d, (x, H - 98), lb, 22, WHITE if i == k else DIM)
    panel(d, (W - 700, 130, W - 80, 214), fill=(14, 18, 14), outline=AMBER)
    text(d, (W - 680, 142), "scale ASSUMED", 28, AMBER, head=True, bold=True)
    text(d, (W - 680, 178), "API returned no metric scale", 22, GRAY)
    return img


def scene_pipeline(t, D, ctx):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    steps = [
        ("Marble world", "splat + collision mesh", "497,664 splats / world\nGLB collider -> physics"),
        ("Task layout", "GR1 + receptacles", "20 Hz, whole-body IK\nrigid items + fixtures"),
        ("Scripted expert", "privileged state", "100 successful demos\n(75-99% of attempts)"),
        ("Imitation (BC)", "MLP 256 x 256", "69-d state -> 4-d action\ndelta xyz + grip"),
        ("Held-out eval", "same seeds for all", "30 episodes x 3 seeds\nWilson 95% intervals"),
    ]
    text(d, (90, 130), "From a generated room to an evaluated policy", 44, WHITE, head=True, bold=True)
    bw, gap, x0, y0 = 300, 56, 98, 300
    for i, (a, b, c) in enumerate(steps):
        p = ease((t - 0.6 - i * 1.5) / 0.8)
        if p <= 0:
            continue
        x = x0 + i * (bw + gap)
        panel(d, (x, y0, x + bw, y0 + 330), outline=fade(GREEN if i in (2, 3) else LINE, p), fill=fade(PANEL, p))
        text(d, (x + 24, y0 + 26), f"0{i + 1}", 22, fade(GREEN, p), head=True, bold=True)
        text(d, (x + 24, y0 + 62), a, 30, fade(WHITE, p), head=True, bold=True)
        text(d, (x + 24, y0 + 112), b, 24, fade(GRAY, p))
        d.line((x + 24, y0 + 168, x + bw - 24, y0 + 168), fill=fade(LINE, p), width=2)
        for j, ln in enumerate(c.split("\n")):
            text(d, (x + 24, y0 + 190 + j * 36), ln, 20, fade(GRAY, p))
        if i < 4:
            ax = x + bw + 12
            d.line((ax, y0 + 165, ax + gap - 24, y0 + 165), fill=fade(GREEN, p), width=4)
            d.polygon([(ax + gap - 24, y0 + 153), (ax + gap - 4, y0 + 165), (ax + gap - 24, y0 + 177)], fill=fade(GREEN, p))
    text(d, (90, 740), "Observations: proprioception (56-d) + goal (2-d) + privileged item pose and phase. No pixels in these policies.", 28, fade(GRAY, ease((t - 8) / 1)))
    text(d, (90, 786), "Success = every item on its slot (3 cm, upright, settled, released), still there at the end, at most 5 invalid contact steps.", 28,
         fade(GRAY, ease((t - 9) / 1)))
    return img


def telemetry_panel(img, d, box, tel, t_idx, n_items, title, color):
    x0, y0, x1, y1 = box
    panel(d, box)
    text(d, (x0 + 28, y0 + 18), title, 30, color, head=True, bold=True)
    cur = tel[min(t_idx, len(tel) - 1)]
    done = cur["items_done"]
    for i in range(n_items):
        cx = x0 + 44 + i * 64
        d.ellipse((cx - 18, y0 + 92, cx + 18, y0 + 128), outline=GREEN, width=3, fill=GREEN if i < done else None)
    text(d, (x0 + 44 + n_items * 64, y0 + 92), f"{done}/{n_items} placed", 26, WHITE)
    text(d, (x1 - 28, y0 + 22), f"step {cur['t']:>4d}", 26, GRAY, anchor="ra")
    text(d, (x0 + 28, y0 + 152), "hand-to-grasp error (m)", 22, GRAY)
    gx0, gy0, gx1, gy1 = x0 + 28, y0 + 184, x1 - 28, y1 - 22
    d.rectangle((gx0, gy0, gx1, gy1), outline=LINE, width=1)
    err = np.array([min(r["eef_error"], 0.5) for r in tel])
    n = len(err)
    pts = [(gx0 + (gx1 - gx0) * i / max(1, n - 1), gy1 - (gy1 - gy0) * err[i] / 0.5) for i in range(0, min(t_idx + 1, n), 2)]
    if len(pts) > 1:
        d.line(pts, fill=color, width=3)
    d.text((gx1 - 6, gy0 + 2), "0.5", font=font(18), fill=DIM, anchor="ra")
    if cur["success"]:
        text(d, (x1 - 28, y0 + 92), "SUCCESS", 30, GREEN, head=True, bold=True, anchor="ra")


def run_view(ctx, name, t, speed):
    m = ctx["meta"][name]
    i = int(t * 20 * speed)
    return m, i


def scene_hero(t, D, ctx):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    speed = ctx["hero_speed"]
    cols = [("hero_expert", "Scripted expert (privileged state)", GRAY), ("hero_bc", "Learned policy (BC-MLP, seed 2)", GREEN)]
    tt = max(0.0, t - 0.5)
    for c, (name, title, colr) in enumerate(cols):
        m, i = run_view(ctx, name, tt, speed)
        x0 = 60 + c * 930
        paste(img, ctx["seq"][name + "_eval_close"].frame(i), (x0, 110, x0 + 900, 616), border=LINE)
        paste(img, ctx["seq"][name + "_eval_top"].frame(i), (x0 + 600, 124, x0 + 888, 286), radius=8, border=colr)
        text(d, (x0 + 18, 590), f"playback {speed}x", 20, WHITE, anchor="ls")
        telemetry_panel(img, d, (x0, 640, x0 + 900, 1040), m["telemetry"], min(i, len(m["telemetry"]) - 1), m["n_items"], title, colr)
    text(d, (W // 2, 70), "clear_table  -  same instance (seed 900001), identical start", 26, GRAY, anchor="ma")
    return img


def scene_tasks(t, D, ctx):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    speed = ctx["tasks_speed"]
    items = [("hero_bc", "Clear table", "clear_table", "3 cups -> flat tray", "learned policy, seed 2"),
             ("laundry_bc", "Laundry basket", "laundry_basket", "3 rolls -> tote  (rigid, no cloth)", "learned policy, seed 0"),
             ("dish_bc", "Load dishwasher", "load_dishwasher", "2 cups -> rack  (static, no door)", "learned policy, seed 0")]
    tt = max(0.0, t - 0.4)
    for c, (name, ttl, task, sub, who) in enumerate(items):
        x0 = 60 + c * 612
        m = ctx["meta"][name]
        i = int(tt * 20 * speed[name])
        paste(img, ctx["seq"][name + "_eval_close"].frame(i), (x0, 120, x0 + 588, 450), border=LINE)
        paste(img, ctx["seq"][name + "_eval_top"].frame(i), (x0 + 392, 130, x0 + 578, 235), radius=6, border=GREEN)
        text(d, (x0 + 14, 440), f"playback {speed[name]}x", 18, WHITE, anchor="ls")
        panel(d, (x0, 470, x0 + 588, 1030))
        text(d, (x0 + 26, 490), ttl, 38, WHITE, head=True, bold=True)
        text(d, (x0 + 26, 542), sub, 24, GRAY)
        text(d, (x0 + 26, 580), who + "  |  seed 900001", 22, DIM)
        k, n = pooled(task)
        lo, hi = wilson(k, n)
        ex = hh(task, "scripted_expert")[0]
        text(d, (x0 + 26, 650), "learned, pooled over 3 seeds", 22, GRAY)
        text(d, (x0 + 26, 682), f"{k / n:.0%}", 76, GREEN, head=True, bold=True)
        text(d, (x0 + 26, 770), f"95% CI [{lo:.0%}, {hi:.0%}]   n = {n}", 24, GRAY)
        text(d, (x0 + 26, 830), "scripted expert", 22, GRAY)
        text(d, (x0 + 26, 862), f"{ex['success']:.0%}", 48, WHITE, head=True, bold=True)
        sd = " / ".join(f"{r['success']:.0%}" for r in hh(task, "bc_mlp"))
        text(d, (x0 + 26, 960), f"learned, per seed: {sd}", 24, GRAY)
        cur = m["telemetry"][min(i, len(m["telemetry"]) - 1)]
        text(d, (x0 + 560, 870), f"{cur['items_done']}/{m['n_items']} placed", 30, GREEN if cur["success"] else WHITE, anchor="ra", head=True)
        if cur["success"]:
            text(d, (x0 + 560, 910), "SUCCESS", 26, GREEN, anchor="ra", head=True, bold=True)
    return img


def axes_pct(d, box, ticks=(0, 25, 50, 75, 100)):
    x0, y0, x1, y1 = box
    for tk in ticks:
        y = y1 - (y1 - y0) * tk / 100
        d.line((x0, y, x1, y), fill=LINE, width=1)
        text(d, (x0 - 14, y), f"{tk}%", 20, DIM, anchor="rm")


def scene_results(t, D, ctx):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    text(d, (90, 110), "Held-out success, 30 episodes per bar", 40, WHITE, head=True, bold=True)
    text(d, (90, 164), "whiskers: Wilson 95% interval    dark = scripted expert    green = learned policy, one bar per training seed", 22, GRAY)
    box = (170, 260, 1520, 900)
    axes_pct(d, box)
    gw = (box[2] - box[0]) / 3
    grow = ease((t - 0.5) / 2.0)
    for gi, (task, label) in enumerate(TASKS):
        bars = [("expert", hh(task, "scripted_expert")[0])] + [(f"s{r['seed']}", r) for r in hh(task, "bc_mlp")]
        bw = 74
        gx = box[0] + gi * gw + (gw - (4 * bw + 3 * 18)) / 2
        for bi, (nm, r) in enumerate(bars):
            x = gx + bi * (bw + 18)
            val = r["success"] * grow
            top = box[3] - (box[3] - box[1]) * val
            col = (70, 80, 70) if nm == "expert" else GREEN
            d.rounded_rectangle((x, top, x + bw, box[3]), radius=6, fill=col)
            lo, hi = r["ci95"]
            if grow > 0.98:
                ymid = lambda v: box[3] - (box[3] - box[1]) * v
                cx = x + bw / 2
                d.line((cx, ymid(hi), cx, ymid(lo)), fill=WHITE, width=3)
                d.line((cx - 9, ymid(hi), cx + 9, ymid(hi)), fill=WHITE, width=3)
                d.line((cx - 9, ymid(lo), cx + 9, ymid(lo)), fill=WHITE, width=3)
                text(d, (cx, box[3] + 12), nm, 20, GRAY, anchor="ma")
                text(d, (cx, ymid(hi) - 8), f"{r['success']:.0%}", 22, WHITE, anchor="mb")
        text(d, (box[0] + gi * gw + gw / 2, box[3] + 58), label, 30, WHITE, head=True, bold=True, anchor="ma")
        k, n = pooled(task)
        text(d, (box[0] + gi * gw + gw / 2, box[3] + 100), f"pooled {k / n:.0%}  ({k}/{n})", 24, GREEN, anchor="ma")
    p = ease((t - 4.5) / 1.0)
    panel(d, (1580, 260, 1860, 900), fill=fade(PANEL, p), outline=fade(LINE, p))
    for j, (ttl, body) in enumerate([("Seed spread", "clear table: 63% to 93%\nsame data, same test set"),
                                     ("Reading", "gap to the expert is\nno wider than the seed\nto seed spread"),
                                     ("Caveat", "state-based policies,\none counter height,\nrigid proxies")]):
        y = 290 + j * 200
        text(d, (1604, y), ttl, 26, fade(GREEN, p), head=True, bold=True)
        for q, ln in enumerate(body.split("\n")):
            text(d, (1604, y + 40 + q * 30), ln, 21, fade(GRAY, p))
    return img


def scene_findings(t, D, ctx):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    bench = {r["name"]: r for r in jl(ROOT / "results" / "benchmark" / "pick_up_cup.json")["rows"]}
    ab = bench["bc_mlp_v2_pick_up_cup_s0"]["summary"]["success_rate"]
    de = bench["bc_mlp_v2delta_pick_up_cup_s0"]["summary"]["success_rate"]
    prop = [r["success"] for r in jl(EXP / "proprio_only_v2.json")["rows"]]
    cnn = [jl(EXP / f"images_v2_{a}_{b}.json")["success"] for a in ("none", "visual") for b in ("none", "visual")]
    ddim = jl(EXP / "ddim_steps_v2.json")["rows"]
    lat = {r["ddim_steps"]: r["min_ms"] for r in jl(EXP / "latency_v2.json")["rows"] if r["policy"] == "diffusion"}
    PW, PX = 570, [90, 690, 1290]
    heads = [("1  Action space", "same demos, same network"), ("2  Pixels vs blind", "pick_up_cup, no cup info vs CNN"),
             ("3  Diffusion steps", "one checkpoint, 5 step budgets")]
    for i in range(3):
        p = ease((t - 0.3 - i * 3.2) / 0.8)
        x = PX[i]
        panel(d, (x, 120, x + PW, 1010), fill=fade(PANEL, p), outline=fade(LINE, p))
        text(d, (x + 28, 142), heads[i][0], 34, fade(WHITE, p), head=True, bold=True)
        text(d, (x + 28, 192), heads[i][1], 22, fade(GRAY, p))
    # panel 1
    p = ease((t - 0.8) / 1.6)
    x = PX[0]
    for j, (lb, v, col) in enumerate([("absolute\ntargets", ab, RED), ("delta\nactions", de, GREEN)]):
        bx = x + 110 + j * 220
        top = 880 - 520 * v * p
        d.rounded_rectangle((bx, top, bx + 120, 880), radius=8, fill=col)
        text(d, (bx + 60, top - 10), f"{v * p:.0%}", 40, WHITE, head=True, bold=True, anchor="mb")
        for q, ln in enumerate(lb.split("\n")):
            text(d, (bx + 60, 896 + q * 28), ln, 22, GRAY, anchor="ma")
    text(d, (x + 28, 972), "controller lag breaks 'pose -> target' regression", 20, DIM)
    # panel 2
    p = ease((t - 3.5) / 1.6)
    x = PX[1]
    gx0, gx1, gy0, gy1 = x + 90, x + PW - 40, 300, 880
    ymax = 0.30
    yy = lambda v: gy1 - (gy1 - gy0) * v / ymax
    for tk in (0, 10, 20, 30):
        d.line((gx0, yy(tk / 100), gx1, yy(tk / 100)), fill=LINE, width=1)
        text(d, (gx0 - 14, yy(tk / 100)), f"{tk}%", 20, DIM, anchor="rm")
    lo_b, hi_b = min(prop), max(prop)
    d.rectangle((gx0, yy(hi_b), gx1, yy(lo_b)), fill=fade(AMBER, 0.16 * p))
    text(d, (gx1 - 8, yy(hi_b) - 6), f"blind range {lo_b:.0%} to {hi_b:.0%}", 20, fade(AMBER, p), anchor="rb")
    for j, (lb, vals, col) in enumerate([("blind\n(3 seeds)", prop, AMBER), ("CNN on pixels\n(4 cells)", cnn, BLUE)]):
        cx = gx0 + 120 + j * 230
        order = sorted(range(len(vals)), key=lambda q: vals[q])
        for rank, q in enumerate(order):
            ox = (rank - (len(vals) - 1) / 2) * 34 * p
            d.ellipse((cx + ox - 11, yy(vals[q]) - 11, cx + ox + 11, yy(vals[q]) + 11), fill=col)
            if p > 0.95:
                text(d, (cx + ox, yy(vals[q]) - 16), f"{vals[q]:.0%}", 16, WHITE, anchor="mb")
        for q, ln in enumerate(lb.split("\n")):
            text(d, (cx, 906 + q * 26), ln, 20, GRAY, anchor="ma")
    text(d, (x + 28, 972), "CNN overlaps the blind range and is never better", 20, DIM)
    # panel 3
    p = ease((t - 6.7) / 1.6)
    x = PX[2]
    gx0, gx1, gy0, gy1 = x + 80, x + PW - 30, 300, 880
    axes_pct(d, (gx0, gy0, gx1, gy1))
    bwid = 62
    for j, r in enumerate(ddim):
        bx = gx0 + 20 + j * 92
        top = gy1 - (gy1 - gy0) * r["success"] * p
        d.rounded_rectangle((bx, top, bx + bwid, gy1), radius=6, fill=GREEN if r["success"] > 0 else LINE)
        text(d, (bx + bwid / 2, top - 8), f"{r['success'] * p:.0%}", 22, WHITE, anchor="mb")
        text(d, (bx + bwid / 2, gy1 + 10), f"{r['ddim_steps']}", 22, WHITE, anchor="ma")
        text(d, (bx + bwid / 2, gy1 + 38), f"{lat[r['ddim_steps']]:.0f}ms", 18, DIM, anchor="ma")
    ybc = gy1 - (gy1 - gy0) * 1.0
    d.line((gx0, ybc, gx1, ybc), fill=AMBER, width=3)
    text(d, (gx1, ybc - 6), "plain BC-MLP 100%", 20, AMBER, anchor="rb")
    text(d, (x + 28, 972), "DDIM steps; ms = CPU latency per 8-step plan", 20, DIM)
    return img


def scene_failure(t, D, ctx):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    m = ctx["meta"]["fail_bc"]
    speed = ctx["fail_speed"]
    i = int(max(0.0, t - 0.5) * 20 * speed)
    paste(img, ctx["seq"]["fail_bc_eval_close"].frame(i), (60, 110, 1260, 785), border=LINE)
    paste(img, ctx["seq"]["fail_bc_eval_top"].frame(i), (900, 124, 1248, 320), radius=8, border=RED)
    text(d, (78, 770), f"playback {speed}x", 20, WHITE, anchor="ls")
    telemetry_panel(img, d, (60, 810, 1260, 1040), m["telemetry"], min(i, len(m["telemetry"]) - 1), m["n_items"],
                    "Learned policy (seed 0), instance 900000", RED)
    panel(d, (1300, 110, 1860, 1040))
    text(d, (1330, 134), "A real failure", 38, RED, head=True, bold=True)
    for j, ln in enumerate(ctx["failure_lines"]):
        text(d, (1330, 206 + j * 40), ln, 25, WHITE if j == 0 else GRAY)
    return img


def height_data():
    """(height, successes, n) per policy from the three multi-height result files; nothing typed in."""
    ev = jl(EXP / "multiheight_load_dishwasher.json")
    ex = jl(EXP / "multiheight_extrap_load_dishwasher.json")
    pr = jl(EXP / "multiheight_probe_load_dishwasher.json")
    n = ev["n_per_height"]
    out = {"single": [], "multi": []}
    for pol, key in (("bc_single_height", "single"), ("bc_multi_height", "multi")):
        for split in ("train_heights", "test_heights"):
            for h, v in ev["per_policy"][pol][split]["per_height"].items():
                out[key].append((float(h), int(round(v * n)), n, split == "train_heights"))
    for r in ex["rows"]:
        out["multi"].append((r["counter_height"], int(round(r["success"] * r["n"])), r["n"], False))
    pe = [(r["counter_height"], r["expert_success"]) for r in pr["rows"]]
    return out, ev, ex, pe


def scene_heights(t, D, ctx):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    data, ev, ex, pe = height_data()
    speed = ctx["height_speed"]
    i = int(max(0.0, t - 0.5) * 20 * speed)
    for j, (nm, ttl, col) in enumerate([("hgt_single", "Trained at 0.85 m only", RED), ("hgt_multi", "Trained at 4 heights", GREEN)]):
        x0 = 60 + j * 620
        paste(img, ctx["seq"][f"{nm}_eval_close"].frame(i), (x0, 110, x0 + 600, 440), border=col)
        m = ctx["meta"][nm]
        text(d, (x0, 456), ttl, 26, col, head=True, bold=True)
        done = min(i, len(m["telemetry"]) - 1)
        text(d, (x0, 490), f"counter 0.865 m (held out)   items {m['telemetry'][done]['items_done']}/{m['n_items']}", 21, GRAY)
    text(d, (60, 1040), f"playback {speed}x, same instance (seed {ctx['meta']['hgt_single']['seed']})", 18, DIM)
    # chart
    gx0, gy0, gx1, gy1 = 150, 620, 1230, 965
    lo, hi = 0.78, 0.95
    xm = lambda h: gx0 + (gx1 - gx0) * (h - lo) / (hi - lo)
    ym = lambda v: gy1 - (gy1 - gy0) * v
    p_sp = ease((t - 1.0) / 1.0)
    d.rectangle((xm(min(ev["train_heights"])), gy0, xm(max(ev["train_heights"])), gy1), fill=fade(GREEN, 0.10 * p_sp))
    text(d, ((xm(0.82) + xm(0.91)) / 2, ym(0.5)), "trained span 0.82 to 0.91 m", 20, fade(GREEN, p_sp), anchor="mm")
    for tk in (0, 25, 50, 75, 100):
        d.line((gx0, ym(tk / 100), gx1, ym(tk / 100)), fill=LINE, width=1)
        text(d, (gx0 - 14, ym(tk / 100)), f"{tk}%", 20, DIM, anchor="rm")
    for h in (0.79, 0.82, 0.85, 0.88, 0.91, 0.94):
        text(d, (xm(h), gy1 + 12), f"{h:.2f}", 20, GRAY, anchor="ma")
    text(d, ((gx0 + gx1) / 2, gy1 + 44), "counter height (m)", 20, DIM, anchor="ma")
    pe_p = ease((t - 2.0) / 1.0)
    d.line((gx0, ym(1.0), gx1, ym(1.0)), fill=fade(AMBER, pe_p), width=2)
    text(d, (gx0 + 6, ym(1.0) - 6), f"scripted expert {min(v for _, v in pe):.0%} at all {len(pe)} heights probed", 19, fade(AMBER, pe_p), anchor="lb")
    for key, col, dx, t0 in (("single", RED, -7, 3.0), ("multi", GREEN, 7, 6.0)):
        pk = ease((t - t0) / 1.2)
        for h, k, n, intrain in data[key]:
            x = xm(h) + dx
            lo_c, hi_c = wilson(k, n)
            c = fade(col, pk)
            d.line((x, ym(hi_c), x, ym(lo_c)), fill=c, width=2)
            r = 9
            if intrain:
                d.ellipse((x - r, ym(k / n) - r, x + r, ym(k / n) + r), fill=c)
            else:
                d.ellipse((x - r, ym(k / n) - r, x + r, ym(k / n) + r), outline=c, width=3)
    ly = gy0 - 62
    d.ellipse((gx0 + 10, ly, gx0 + 28, ly + 18), fill=GRAY)
    text(d, (gx0 + 38, ly - 2), "filled: height in training set", 19, GRAY)
    d.ellipse((gx0 + 330, ly, gx0 + 348, ly + 18), outline=GRAY, width=3)
    text(d, (gx0 + 358, ly - 2), "ring: held-out height", 19, GRAY)
    text(d, (gx0 + 640, ly - 2), "n = 15 per point, Wilson 95%", 19, GRAY)
    # right panel
    p = ease((t - 7.5) / 1.0)
    panel(d, (1300, 110, 1860, 1040), fill=fade(PANEL, p), outline=fade(LINE, p))
    text(d, (1330, 134), "Counter height", 38, fade(GREEN, p), head=True, bold=True)
    ev_p = ev["per_policy"]
    rows = [("single-height policy", f"{ev_p['bc_single_height']['train_heights']['success']:.0%} on training heights", RED),
            ("", f"{ev_p['bc_single_height']['test_heights']['success']:.0%} on held-out heights", RED),
            ("multi-height policy", f"{ev_p['bc_multi_height']['train_heights']['success']:.0%} on training heights", GREEN),
            ("", f"{ev_p['bc_multi_height']['test_heights']['success']:.0%} on held-out heights (inside span)", GREEN),
            ("", " / ".join(f"{r['success']:.0%}" for r in ex["rows"]) + "  at 0.79 / 0.94 m, outside span", AMBER)]
    y = 210
    for a, b, col in rows:
        if a:
            y += 14
            text(d, (1330, y), a, 25, fade(WHITE, p), head=True, bold=True)
            y += 38
        text(d, (1330, y), b, 22, fade(col, p))
        y += 36
    y += 30
    for ln in ["30 demos at each of", "0.82 / 0.85 / 0.88 / 0.91 m.", "One seed, one task (dishwasher).", "It interpolates; it does", "not extrapolate."]:
        text(d, (1330, y), ln, 22, fade(GRAY, p))
        y += 32
    return img


def scene_limits(t, D, ctx):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    text(d, (120, 130), "What this does and does not show", 56, WHITE, head=True, bold=True)
    left = [("Privileged state", "policies see item poses, not pixels; a CNN on pixels did not beat a blind policy"),
            ("Rigid proxies", "no door, sliding rack, washer, or cloth; layouts tuned until the expert passed"),
            ("Counter height", "one training height fails elsewhere; four heights fix the dishwasher only inside the span"),
            ("Assumed scale", "Marble returned no metric scale; rooms use assumed sizes"),
            ("Incomplete", "domain-randomisation grid: 1 of 32 evaluations finished")]
    for j, (a, b) in enumerate(left):
        p = ease((t - 0.6 - j * 1.4) / 0.7)
        y = 260 + j * 126
        d.rectangle((120, y + 6, 132, y + 18), fill=fade(AMBER, p))
        text(d, (160, y - 8), a, 34, fade(WHITE, p), head=True, bold=True)
        text(d, (160, y + 36), b, 24, fade(GRAY, p))
    p = ease((t - 8) / 1.0)
    panel(d, (1240, 250, 1800, 880), fill=fade(PANEL, p), outline=fade(LINE, p))
    text(d, (1274, 276), "Next", 40, fade(GREEN, p), head=True, bold=True)
    for j, ln in enumerate(["perception instead of poses", "articulated doors and racks", "cloth and washer / dryer",
                            "heights beyond the span, other tasks", "teleoperated, multimodal demos"]):
        text(d, (1274, 350 + j * 70), ln, 27, fade(WHITE, p))
    text(d, (120, 960), "reports/final_report.md   |   results/experiments/*.json   |   62 tests pass", 24, fade(GRAY, ease((t - 9) / 1)))
    return img


SCENES = [("title", scene_title, "Title"), ("worlds", scene_worlds, "Generated worlds"), ("pipeline", scene_pipeline, "Method"),
          ("hero", scene_hero, "Clear table"), ("tasks", scene_tasks, "Three tasks"), ("results", scene_results, "Results"),
          ("findings", scene_findings, "Findings"), ("failure", scene_failure, "Failure analysis"), ("heights", scene_heights, "Counter height"),
          ("limits", scene_limits, "Limits and next steps")]


def build(preview=0):
    dur = jl(AUD / "durations.json")["durations"]
    scenes = [(k, f, lb) for k, f, lb in SCENES if k in dur]
    ctx = {"title_bg": Seq(FOOT / "world_marble_kitchen_001.mp4"),
           "worlds": [Seq(FOOT / f"world_{n}.mp4") for n in ("marble_kitchen_001", "marble_living_room_001", "marble_bedroom_001",
                                                           "marble_laundry_room_001", "marble_multiroom_apartment_001")],
           "meta": {}, "seq": {}}
    for nm in ("hero_expert", "hero_bc", "laundry_bc", "dish_bc", "fail_bc", "hgt_single", "hgt_multi"):
        ctx["meta"][nm] = meta(nm)
        for cam in ("eval_close", "eval_top"):
            ctx["seq"][f"{nm}_{cam}"] = Seq(FOOT / f"{nm}_{cam}.mp4")
    ctx["hero_speed"] = 1.4
    ctx["tasks_speed"] = {"hero_bc": 2.0, "laundry_bc": 2.0, "dish_bc": 1.4}
    ctx["fail_speed"] = 1.5
    ctx["height_speed"] = 2.0
    ctx["failure_lines"] = (jl(HERE / "failure_lines.json") or [])
    lead, tail = 0.5, 0.7
    starts, total = [], 0.0
    for k, f, lb in scenes:
        starts.append(total)
        total += dur[k] + lead + tail
    out = (ROOT / "results" / "videos" / ("marble_home_research_v2.mp4" if not preview else "_preview.mp4"))
    silent = HERE / "_silent.mp4"
    wr = imageio.get_writer(silent, fps=FPS, macro_block_size=1, quality=8)
    for si, (k, f, lb) in enumerate(scenes):
        D = dur[k] + lead + tail
        n = int(D * FPS)
        step = preview if preview else 1
        for fi in range(0, n, step):
            t = fi / FPS
            img = f(t, D, ctx)
            if k != "title":
                chrome(img, si + 1, len(scenes), lb, (starts[si] + t) / total)
            a = min(1.0, t / 0.25, (D - t) / 0.25)
            if a < 1.0:
                img = Image.fromarray((np.asarray(img).astype(np.float32) * max(a, 0.0)).astype(np.uint8))
            wr.append_data(np.asarray(img))
        print("scene", k, round(D, 1), "s", flush=True)
    wr.close()
    # audio: each narration starts `lead` seconds into its scene
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ff, "-y", "-i", str(silent)]
    flt = []
    for j, (k, f, lb) in enumerate(scenes):
        cmd += ["-i", str(AUD / f"{k}.mp3")]
        ms = int((starts[j] + lead) * 1000)
        flt.append(f"[{j + 1}:a]adelay={ms}|{ms}[a{j}]")
    flt.append("".join(f"[a{j}]" for j in range(len(scenes))) + f"amix=inputs={len(scenes)}:normalize=0[aout]")
    cmd += ["-filter_complex", ";".join(flt), "-map", "0:v", "-map", "[aout]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k", "-shortest", str(out)]
    subprocess.run(cmd, check=True, capture_output=True)
    print("wrote", out, round(total, 1), "s")


if __name__ == "__main__":
    pv = 0
    if "--preview" in sys.argv:
        pv = int(sys.argv[sys.argv.index("--preview") + 1])
    build(pv)
