"""Multi-item household tasks (clear_table / load_dishwasher / laundry_basket) and the CNN checkpoint action_mode regression."""
import inspect

import numpy as np
import pytest

from marble_home.simulation.kitchen_arena import merge_scene
from marble_home.simulation.layouts import apply_layout
from marble_home.tasks import definitions as TD


@pytest.mark.parametrize("task", list(TD.MULTI_ITEM))
def test_layout_shapes(task):
    sc = apply_layout(task, merge_scene(None))
    n = TD.MULTI_ITEM[task]
    assert len(sc["slots"]) == n and len(sc["item_spawn"]) == n and len(sc["item_rgba"]) >= n
    xs = [s["x"] for s in sc["slots"]]
    assert all(a - b >= 0.11 for a, b in zip(xs, xs[1:])), "slots must be >= 11 cm apart, filled far-to-near (wrist trails toward -x)"
    # each item's approach corridor must be clear of the next item (spawn bands strictly increasing in y with a gap)
    ys = [sp["y"] for sp in sc["item_spawn"]]
    assert all(b[0] - a[1] >= 0.03 for a, b in zip(ys, ys[1:]))
    assert apply_layout(task, sc) is sc  # idempotent
    assert apply_layout("pick_up_cup", merge_scene(None))["place_target"] == merge_scene(None)["place_target"]


def test_single_item_tasks_unchanged():
    cfg = TD.default_task_config("place_cup")
    assert cfg.n_items == 1 and cfg.horizon == 320 and cfg.task not in TD.MULTI_ITEM


def test_multi_task_config():
    cfg = TD.default_task_config("clear_table")
    assert cfg.n_items == 3 and cfg.horizon >= 3 * 300


@pytest.fixture(scope="module")
def multi_env():
    from marble_home.simulation.gym_env import HouseholdEnv

    e = HouseholdEnv(task="load_dishwasher", seed=0)
    e.reset(seed=3)
    yield e
    e.close()


def test_items_spawn_and_hand_over(multi_env):
    """Teleport item 0 onto its slot with the 'was lifted' flag set: the env must count it and hand over to item 1."""
    e = multi_env
    e.reset(seed=3)
    k = e.env
    assert k.n_items == 2 and k.active == 0 and k._items_done == 0
    p0, p1 = (np.array(k.sim.data.body_xpos[b]) for b in k.item_body_ids)
    assert np.linalg.norm(p0[:2] - p1[:2]) > 0.066, "items must not overlap at spawn"
    sl = k.slots[0]
    jn = k.items[0].joints[0]
    k.sim.data.set_joint_qpos(jn, np.array([sl["x"], sl["y"], k.table_z + sl["dz"] + 0.0455, 1, 0, 0, 0]))
    k.sim.forward()
    k._was_lifted = True
    act = np.zeros(4, np.float32)
    act[3] = -1.0
    info = {}
    for _ in range(k.task_cfg.place_hold_steps + 5):
        _, _, _, _, info = e.step(act)
        if k.active == 1:
            break
    assert k._items_done == 1 and k.active == 1, "finished item must be counted and the next item activated"
    assert not info["success"], "one of two items placed is not task success"
    assert np.allclose(k.target_xy, [k.slots[1]["x"], k.slots[1]["y"]])
    assert k.cup_rest_z > k.table_z + 0.08, "dishwasher rack slots are raised"


def test_final_check_requires_finished_items_to_stay_put(multi_env):
    e = multi_env
    e.reset(seed=4)
    k = e.env
    k._items_done = 1  # pretend item 0 was placed ...
    assert not k._finished_items_ok()  # ... but it is still at its spawn point, not on the slot
    sl = k.slots[0]
    k.sim.data.set_joint_qpos(k.items[0].joints[0], np.array([sl["x"], sl["y"], k.table_z + sl["dz"] + 0.0455, 1, 0, 0, 0]))
    k.sim.forward()
    assert k._finished_items_ok()


def test_cnn_checkpoint_records_action_mode():
    """Regression: train_bc_images once omitted action_mode, so delta-trained CNNs were evaluated with absolute actions (0%)."""
    from marble_home.training import train_bc as tb

    src = inspect.getsource(tb.train_bc_images)
    assert '"action_mode": dataset_action_mode(ds)' in src
