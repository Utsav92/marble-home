import json
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")


@pytest.fixture(scope="session")
def env_state():
    """One shared state-obs environment (building a GR1 env takes several seconds)."""
    from marble_home.simulation.gym_env import HouseholdEnv

    env = HouseholdEnv(task="pick_up_cup", obs_mode="state", seed=0)
    env.reset(seed=0)
    yield env
    env.close()


@pytest.fixture(scope="session")
def env_rgbd():
    from marble_home.simulation.gym_env import HouseholdEnv

    env = HouseholdEnv(task="pick_up_cup", obs_mode="rgbd", image_hw=64, seed=0)
    env.reset(seed=0)
    yield env
    env.close()


@pytest.fixture(scope="session")
def fixture_export(tmp_path_factory):
    from marble_home.geometry.fixtures import write_fixture

    return write_fixture(tmp_path_factory.mktemp("raw") / "kitchen_fixture_test")


@pytest.fixture(scope="session")
def processed_fixture(fixture_export, tmp_path_factory):
    from marble_home.geometry.pipeline import process_export

    root = tmp_path_factory.mktemp("proc")
    import marble_home.worldlabs.manifest as mf

    # keep the registry out of the real data dir
    orig = mf.AssetRegistry.__init__

    def init(self, root_=None):
        orig(self, tmp_path_factory.mktemp("reg"))

    mf.AssetRegistry.__init__ = init
    try:
        return process_export(fixture_export, scene_id="kitchen_fixture_test", processed_root=root)
    finally:
        mf.AssetRegistry.__init__ = orig
