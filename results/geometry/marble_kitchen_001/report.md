# Geometry validation: marble_kitchen_001 (marble_import)
**PASS**

- PASS `mesh_loaded_and_processed`: 337646 raw faces -> 2339 boxes + 0 hulls
- PASS `floor_height_plausible`: floor after alignment z = -0.0000 m (tol 0.03); raw floor was at -1.179 m (shifted +1.179 m); floor = 71% of up-facing area
- PASS (advisory) `room_dimensions_plausible`: room 4.7 x 4.2 m, height 3.16 m (declared dimensions are constraints to verify, not guarantees)
- PASS `complexity_within_limits`: boxes 2339/3000, max hull faces 0/96
- PASS `task_zone_free_of_static_geometry`: 0 Marble boxes intersect the counter workspace {'x': (-0.6, 0.45), 'y': (-0.65, 0.65), 'z': (0.88, 1.6)}
- PASS `mujoco_model_compiles`: 3829 geoms, 92 bodies
- PASS `no_floor_penetration`: deepest robot contact 0.0 mm (0 contacts), base z 0.950 m (nominal 0.95), cup rest error 0.5 mm
- PASS `robot_spawn_collision_free`: 0 robot<->Marble-geometry contacts after 40 settle steps
- PASS `screenshots_nonblank`: 4 screenshots rendered; required views overview+top OK; occluded (advisory): ['eval_side']
- PASS `approach_positions_reachable_scripted_pick`: scripted pick_up_cup: 2/3 in this scene vs 2/3 in the open procedural kitchen ([{'seed': 500, 'marble_scene': True, 'open_baseline': True, 'invalid_steps': 0}, {'seed': 501, 'marble_scene': True, 'open_baseline': True, 'invalid_steps': 0}, {'seed': 502, 'marble_scene': False, 'open_baseline': False, 'invalid_steps': 0}])
