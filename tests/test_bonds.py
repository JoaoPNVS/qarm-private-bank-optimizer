import numpy as np
import pandas as pd
import pytest

from src import bonds, data


@pytest.fixture(scope="module")
def universe():
    return data.load_universe()


def test_rating_scale():
    assert bonds.rating_from_score(3.0) == "AA" and bonds.rating_from_score(10.76) == "BB+"
    assert bonds.rating_from_score(np.nan) == "n/a"


def test_sleeve_summary_weighted_averages(universe):
    w = pd.Series({"SHY": 0.20, "TLT": 0.10, "HYG": 0.10, "AAPL": 0.60})
    s = bonds.sleeve_summary(w, universe)
    assert np.isclose(s["share"], 0.40)
    u = universe.loc[["SHY", "TLT", "HYG"]]
    rel = pd.Series([0.5, 0.25, 0.25], index=u.index)
    assert np.isclose(s["duration"], (rel * u["duration"]).sum())
    assert np.isclose(s["ig_share"], 0.75) and np.isclose(s["hy_share_of_portfolio"], 0.10)
    assert np.isclose(s["by_maturity"].sum(), 1) and list(s["by_maturity"].index) == bonds.MATURITY_ORDER
    assert s["by_issuer"]["Treasury"] == pytest.approx(0.75)
    assert s["avg_rating"] == bonds.rating_from_score((rel * u["rating_score"]).sum())
    assert np.isclose(s["funds"]["Share of bonds"].sum(), 1)


def test_empty_sleeve(universe):
    s = bonds.sleeve_summary(pd.Series({"AAPL": 1.0}), universe)
    assert s["share"] == 0 and np.isnan(s["duration"]) and s["avg_rating"] == "n/a"


def test_solver_dust_is_not_a_bond_portfolio(universe):
    w = pd.Series({"AAPL": 0.99999, "FLOT": 0.000005, "SHY": 0.000005})
    assert bonds.sleeve_summary(w, universe)["share"] == 0
