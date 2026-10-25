import importlib

import pytest

MODULES = ["data", "estimators", "optimizer", "benchmarks", "bonds", "backtest", "risk", "profiling", "report"]


@pytest.mark.parametrize("name", MODULES)
def test_modules_import(name):
    importlib.import_module(f"src.{name}")
