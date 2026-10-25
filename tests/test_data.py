import numpy as np
import pandas as pd
import pytest

from src import data

MIN_MONTHS = 15 * 12


@pytest.fixture(scope="module")
def universe():
    return data.load_universe()


@pytest.fixture(scope="module")
def returns():
    return data.load_returns()


def test_universe_tags(universe):
    assert len(universe) == 35
    assert set(universe.loc[["EEM", "AAXJ", "EMB"], "currency"]) == {"EM-FX"}
    assert universe.loc[["EEM", "AAXJ", "EMB"], "non_chf"].all()
    assert universe.loc[["EEM", "AAXJ", "EMB"], "is_EM"].all()
    assert not universe.loc[universe["currency"] == "CHF", "non_chf"].any()
    assert set(universe["listed_currency"]) == {"CHF", "USD"}


def test_returns_match_universe(returns, universe):
    assert list(returns.columns) == list(universe.index)
    assert returns.index.is_monotonic_increasing and returns.index.is_month_end.all()
    assert np.isfinite(returns.stack()).all()
    assert returns.abs().max().max() < 0.6


def test_every_asset_has_15_years(returns):
    counts = returns.notna().sum()
    assert (counts >= MIN_MONTHS).all(), counts[counts < MIN_MONTHS]


def test_no_gaps_after_inception(returns):
    for t in returns:
        s = returns[t].loc[returns[t].first_valid_index():]
        assert s.notna().all(), f"{t} has gaps"


def test_cash_equals_rf(returns):
    rf = data.load_rf()
    both = pd.concat([returns[data.CASH_TICKER], rf], axis=1).dropna()
    assert len(both) > MIN_MONTHS
    assert np.allclose(both.iloc[:, 0], both.iloc[:, 1])


def test_usd_asset_converted_to_chf(returns):
    px = pd.read_parquet(data.DATA_DIR / "prices_local.parquet")["SPY"]
    fx = data.load_fx()["USDCHF=X"]
    expected = (px * fx).pct_change(fill_method=None)
    both = pd.concat([returns["SPY"], expected], axis=1).dropna()
    assert np.allclose(both.iloc[:, 0], both.iloc[:, 1])


def test_common_start(returns):
    assert data.common_start(returns, list(returns.columns)) == pd.Timestamp("2011-03-31")


def test_constant_rf_fallback(returns):
    rf = data.constant_rf(returns.index, 0.01)
    assert np.isclose(rf.iloc[0], 0.01 / 12)


def test_missing_file_gives_clear_error(tmp_path):
    with pytest.raises(data.DataError, match="download_data.py"):
        data.load_rf(tmp_path)
