# Geometry validation: kitchen_fixture_001 (synthetic_fixture)
**PASS**

- PASS `mesh_loaded_and_processed`: 288 raw faces -> 24 boxes + 6 hulls
- PASS `floor_height_plausible`: detected floor z = -0.0000 m (tol 0.03)
- PASS (advisory) `room_dimensions_plausible`: room 5.2 x 4.2 m, height 2.70 m (declared dimensions are constraints to verify, not guarantees)
- PASS `complexity_within_limits`: boxes 24/600, max hull faces 12/96
- PASS `task_zone_free_of_static_geometry`: 0 Marble boxes intersect the counter workspace {'x': (-0.6, 0.45), 'y': (-0.75, 0.75), 'z': (0.88, 1.6)}
- PASS `mujoco_model_compiles`: 189 geoms, 92 bodies
- PASS `no_floor_penetration`: deepest robot contact 0.0 mm (0 contacts), base z 0.950 m (nominal 0.95), cup rest error 0.5 mm
- PASS `robot_spawn_collision_free`: 0 robot<->Marble-geometry contacts after 40 settle steps
- PASS `screenshots_nonblank`: 4 screenshots rendered
- PASS `approach_positions_reachable_scripted_pick`: scripted pick_up_cup: 3/4 in this scene vs 3/4 in the open procedural kitchen ([{'seed': 500, 'marble_scene': True, 'open_baseline': True, 'invalid_steps': 0}, {'seed': 501, 'marble_scene': True, 'open_baseline': True, 'invalid_steps': 0}, {'seed': 502, 'marble_scene': False, 'open_baseline': False, 'invalid_steps': 0}, {'seed': 503, 'marble_scene': True, 'open_baseline': True, 'invalid_steps': 0}])
