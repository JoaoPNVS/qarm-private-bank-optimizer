import importlib
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")
MODULES = ["data", "estimators", "optimizer", "benchmarks", "backtest", "risk", "profiling", "report"]
TIMEOUT = 180


@pytest.mark.parametrize("name", MODULES)
def test_modules_import(name):
    importlib.import_module(f"src.{name}")


def run_app(widgets: dict | None = None) -> AppTest:
    """Run the app, then set each {(widget_kind, key): value} in order, rerunning after each one
    (widgets of a profile only exist once that profile is selected)."""
    at = AppTest.from_file(APP_PATH).run(timeout=TIMEOUT)
    for (kind, key), value in (widgets or {}).items():
        getattr(at, kind)(key=key).set_value(value)
        at.run(timeout=TIMEOUT)
    return at


def test_default_app_runs_cleanly():
    at = run_app()
    assert not at.exception and not at.error
    assert not at.warning  # also catches Streamlit deprecation notices
    assert "All active constraints are satisfied" in at.success[0].value
    assert any(m.label == "Sharpe ratio" for m in at.metric)


@pytest.mark.parametrize("profile", ["Conservative", "Balanced", "Growth", "Dynamic"])
def test_every_profile_runs(profile):
    at = run_app({("selectbox", "profile"): profile})
    assert not at.exception and not at.error


def test_incompatible_constraints_show_message():
    # 70% in bonds needs at least four bond funds at 20%; at 10% per fund the five funds give 50%
    at = run_app({("selectbox", "profile"): "Conservative", ("slider", "Conservative.etf"): 10})
    assert not at.exception
    assert any("cannot all be satisfied" in e.value for e in at.error)
