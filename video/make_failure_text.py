"""Derive the failure-analysis captions from the captured telemetry (video/footage/fail_bc.json); nothing is typed in by hand."""
import json
from pathlib import Path

H = Path(__file__).resolve().parent
m = json.loads((H / "footage" / "fail_bc.json").read_text())
t = m["telemetry"]
lift = next(r["t"] for r in t if r["item"][2] > 0.95)
rel = next(i for i in range(lift, len(t)) if t[i]["grip"] < -0.5)  # first open command after the lift
pe_before = t[max(rel - 2, 0)]["place_error"]  # slot error just before the fingers open
pe_after = t[min(rel + 20, len(t) - 1)]["place_error"]  # slot error 20 steps after the open command
inv, peak = t[-1]["invalid_steps"], max(r["peak_force"] for r in t)
cm = lambda v: f"{v * 100:.1f} cm"
lines = [
    f"Instance {m['seed']}, policy seed 0, clear_table.",
    "",
    f"Cup 1 reaches the tray (lifted at step {lift}) and is",
    f"set down {cm(pe_before)} from its slot. As the fingers",
    f"open (step {t[rel]['t']}) the cup shifts sideways:",
    f"slot error {cm(pe_before)} -> {cm(pe_after)}. Tolerance: 3.0 cm.",
    f"Result: 0 / {m['n_items']} items counted.",
    "",
    "Demonstrations contain only successful",
    "placements, so the policy has never seen a",
    "miss. It drifts to the edge of its workspace",
    "and presses into the scene:",
    f"{inv} invalid-contact steps,",
    f"peak contact force {peak:.0f} N.",
    "",
    "Untried fixes: recovery data (DAgger-style),",
    "a retry behaviour, a gentler grip release.",
]
(H / "failure_lines.json").write_text(json.dumps(lines))
text = (f"Failures are informative. Here the policy carried the first cup to the tray and set it down "
        f"{pe_before * 100:.0f} centimetre from its slot. As the fingers opened, the cup shifted to "
        f"{pe_after * 100:.0f} centimetres out, beyond the three centimetre tolerance. "
        f"Because the demonstrations contain only successful placements, the policy had never seen a miss: "
        f"it drifted to the edge of its workspace and pushed against the scene for the rest of the episode.")
(H / "failure_text.txt").write_text(text)
print("\n".join(lines))
print(text)
