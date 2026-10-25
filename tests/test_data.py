import numpy as np
import pandas as pd
import pytest

from src import data

WINDOW = 60


@pytest.fixture(scope="module")
def universe():
    return data.load_universe()


@pytest.fixture(scope="module")
def returns():
    return data.load_returns()


def test_universe_composition(universe):
    counts = universe["asset_class"].value_counts()
    assert counts["Equity"] == 150 and counts["Bond"] == 3 and counts["Cash"] == 1
    stocks = universe[universe["asset_class"] == "Equity"]
    assert (stocks["listed_currency"] == "USD").all()
    assert not {"GOOG", "FOX", "NWS"} & set(universe.index)        # one share class per company
    assert stocks["sector"].nunique() >= 10 and (stocks["sector"] != "").all()
    assert (universe.loc[universe["asset_class"] == "Bond", "listed_currency"] == "CHF").all()


def test_returns_match_universe(returns, universe):
    assert list(returns.columns) == list(universe.index)
    assert returns.index.is_monotonic_increasing and returns.index.is_month_end.all()
    assert np.isfinite(returns.stack()).all()
    assert returns.abs().max().max() < 2.0


def test_no_gaps_after_inception(returns):
    for t in returns:
        s = returns[t].loc[returns[t].first_valid_index():]
        assert s.notna().all(), f"{t} has gaps"


def test_enough_assets_eligible_today(returns):
    eligible = returns.tail(WINDOW).notna().all()
    assert eligible.sum() >= 120  # recent IPOs and spin-offs wait for 60 months


def test_cash_equals_rf(returns):
    rf = data.load_rf()
    both = pd.concat([returns[data.CASH_TICKER], rf], axis=1).dropna()
    assert len(both) > 200 and np.allclose(both.iloc[:, 0], both.iloc[:, 1])


def test_usd_stock_converted_to_chf(returns):
    px = pd.read_parquet(data.DATA_DIR / "prices_local.parquet")["AAPL"]
    fx = data.load_fx()["USDCHF=X"]
    expected = (px * fx).pct_change(fill_method=None)
    both = pd.concat([returns["AAPL"], expected], axis=1).dropna()
    assert np.allclose(both.iloc[:, 0], both.iloc[:, 1])


def test_benchmark_spy_in_chf(returns):
    spy = data.load_benchmark_returns()
    assert spy.index.equals(returns.index)
    assert spy.notna().sum() > 250
    assert 0.10 < spy.std() * np.sqrt(12) < 0.25


def test_constant_rf_fallback(returns):
    assert np.isclose(data.constant_rf(returns.index, 0.01).iloc[0], 0.01 / 12)


def test_missing_file_gives_clear_error(tmp_path):
    with pytest.raises(data.DataError, match="download_data.py"):
        data.load_rf(tmp_path)
