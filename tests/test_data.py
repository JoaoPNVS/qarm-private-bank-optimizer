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
    assert counts["Equity"] == 153 and counts["Bond"] == 15 and counts["Cash"] == 1
    assert (universe["listed_currency"] == "USD").all()             # no currency conversion anywhere
    assert not {"GOOG", "FOX", "NWS"} & set(universe.index)        # one share class per company
    stocks = universe[universe["asset_class"] == "Equity"]
    assert stocks["sector"].nunique() >= 10 and (stocks["sector"] != "").all()


def test_fixed_assets_and_tags(universe):
    etfs = universe[universe["vehicle"] == "ETF"]
    bond_etfs = {"BIL", "SHY", "IEI", "IEF", "TLT", "IGSB", "IGIB", "VCLT", "HYG", "MUB", "MBB", "EMB", "BNDX",
                 "TIP", "FLOT"}
    assert set(etfs.index) == {"VGK", "EWJ", "EEM", "GLD"} | bond_etfs
    assert set(universe.index[universe["is_EM"]]) == {"EEM", "PDD", "MELI"}
    assert set(universe.loc[universe["sector"] == "International"].index) == {"VGK", "EWJ", "EEM"}
    bonds = universe[universe["asset_class"] == "Bond"]
    assert set(bonds.index) == bond_etfs
    assert set(bonds["bond_type"]) == {"Treasury", "Corporate IG", "High Yield", "Municipal", "Mortgage-backed",
                                       "EM USD", "International hedged", "TIPS", "Floating rate"}
    assert universe.loc["GLD", "asset_class"] == "Commodity"
    assert (universe.loc[universe["vehicle"] == "Stock", "asset_class"] == "Equity").all()
    assert len(universe[universe["vehicle"] == "Stock"]) == 150


def test_returns_match_universe(returns, universe):
    assert list(returns.columns) == list(universe.index)
    assert returns.index.is_monotonic_increasing and returns.index.is_month_end.all()
    assert returns.index[0] == pd.Timestamp("1998-02-28")
    assert np.isfinite(returns.stack()).all()


def test_no_gaps_after_inception(returns):
    for t in returns:
        s = returns[t].loc[returns[t].first_valid_index():]
        assert s.notna().all(), f"{t} has gaps"


def test_enough_assets_eligible_today(returns):
    assert returns.tail(WINDOW).notna().all().sum() >= 120  # recent IPOs and spin-offs wait for 60 months


def test_cash_earns_lagged_t_bill(returns):
    rf = data.load_rf()
    irx = pd.read_parquet(data.DATA_DIR / "rf.parquet")["irx_pct"]
    assert np.allclose(rf.iloc[1:], irx.shift(1).dropna().loc[rf.index[1:]] / 1200)
    assert np.allclose(returns[data.CASH_TICKER], rf.reindex(returns.index))
    assert 0 <= rf.max() * 12 < 0.07   # T-bill between 0% and 7% since 1998


def test_returns_are_price_returns_in_usd(returns):
    px = pd.read_parquet(data.DATA_DIR / "prices.parquet")["AAPL"]
    both = pd.concat([returns["AAPL"], px.pct_change(fill_method=None)], axis=1).dropna()
    assert np.allclose(both.iloc[:, 0], both.iloc[:, 1])


def test_benchmarks(returns):
    b = data.load_benchmark_returns()
    assert list(b.columns) == ["SPY", "AGG"] and b.index.equals(returns.index)
    assert b["AGG"].first_valid_index() == pd.Timestamp("2003-10-31")
    assert 0.12 < b["SPY"].std() * np.sqrt(12) < 0.20 and b["AGG"].std() * np.sqrt(12) < 0.07


def test_missing_file_gives_clear_error(tmp_path):
    with pytest.raises(data.DataError, match="download_data.py"):
        data.load_rf(tmp_path)


def test_bond_characteristics(universe):
    b = universe[universe["asset_class"] == "Bond"]
    assert b["duration"].between(0, 20).all() and (b["wam"] >= b["duration"] - 1e-9).all()
    assert set(b.index[b["credit_quality"] == "HY"]) == {"HYG", "EMB"}
    assert set(b["maturity_bucket"]) == {"Ultra-short", "Short", "Intermediate", "Long"}
    assert b.loc["TLT", "duration"] > b.loc["IEF", "duration"] > b.loc["IEI", "duration"] > b.loc["SHY", "duration"]
    assert b.loc["HYG", "rating_score"] > b.loc["IGIB", "rating_score"] > b.loc["IEF", "rating_score"]  # worse = higher
    assert b["char_source"].str.startswith("https://").all() and (b["char_as_of"] >= "2026-09-30").all()
    assert universe.loc[universe["asset_class"] != "Bond", "duration"].isna().all()
