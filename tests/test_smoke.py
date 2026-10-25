import importlib
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")
MODULES = ["data", "estimators", "optimizer", "benchmarks", "backtest", "risk", "profiling", "report"]


@pytest.mark.parametrize("name", MODULES)
def test_modules_import(name):
    importlib.import_module(f"src.{name}")


def test_app_runs_without_exception():
    at = AppTest.from_file(APP_PATH).run(timeout=60)
    assert not at.exception and not at.error
