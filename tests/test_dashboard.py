from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "src" / "marble_home" / "visualization" / "dashboard.py"


@pytest.mark.parametrize("view", ["Training Dashboard", "Benchmark Dashboard", "World Browser", "Experiment Explorer", "Simulation Viewer"])
def test_dashboard_view_renders_without_exception(view):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    at.sidebar.radio[0].set_value(view).run()
    assert not at.exception, [e.value for e in at.exception]
