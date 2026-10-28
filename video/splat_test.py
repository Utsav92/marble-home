import sys

import cv2
import numpy as np

sys.path.insert(0, "D:/marble-home/video")
from splat_render import load_splat, render

world = sys.argv[1] if len(sys.argv) > 1 else "marble_living_room_001"
pos, rgb, sw = load_splat(f"D:/marble-home/data/marble_raw/{world}/splat.ply")
print(len(pos), "splats kept; extents", np.percentile(pos, [2, 98], axis=0).round(2).tolist())
tiles = []
for up_sign in (-1.0, 1.0):  # y-down (OpenCV-style) vs y-up
    up = np.array([0, up_sign, 0.0])
    c = np.median(pos, axis=0)
    for ang in (0.0, 2.0):
        eye = c + np.array([0.0, 0.0, 0.0])
        tgt = eye + np.array([np.sin(ang), 0.0, np.cos(ang)])
        img = render(pos, rgb, sw, eye, tgt, up, W=640, H=360, fov_deg=80)
        cv2.putText(img, f"up_sign={up_sign} ang={ang}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        tiles.append(img)
sheet = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
cv2.imwrite(f"D:/marble-home/video/_splat_{world}.png", cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
