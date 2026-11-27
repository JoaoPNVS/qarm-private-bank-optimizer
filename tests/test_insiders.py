import numpy as np
import pandas as pd
import pytest

from src import backtest as btm, benchmarks as bm, data, estimators as est, insiders as ins


@pytest.fixture(scope="module")
def market():
    return data.load_insider_market()


@pytest.fixture(scope="module")
def trades():
    return data.load_insider_trades()


@pytest.fixture(scope="module")
def signal(market):
    return ins.market_signal(market)


@pytest.fixture(scope="module")
def base():
    universe, returns = data.load_universe(), data.load_returns()
    window = est.estimation_window(returns, returns.index[-1])
    return est.estimate(window, data.load_rf(), est.shrinkage_groups(universe),
                        benchmarks=data.load_benchmark_returns(), benchmark_class=bm.BENCHMARK_CLASS)


# --- data --------------------------------------------------------------------------------------

def test_market_data_is_monthly_and_covers_the_backtest(market):
    idx = market.index
    assert idx.is_monotonic_increasing and (idx == idx + pd.offsets.MonthEnd(0)).all()
    assert len(pd.date_range(idx[0], idx[-1], freq="ME").difference(idx)) == 0   # no missing month
    assert idx[0] <= pd.Timestamp("2006-01-31") and idx[-1] >= data.load_returns().index[-1]
    assert (market["buy_companies"] > 0).all() and (market["sell_companies"] > 0).all()


def test_trades_are_our_stocks_and_public_before_the_last_price(trades):
    universe = data.load_universe()
    stocks = set(universe.index[universe["vehicle"] == "Stock"])
    assert set(trades["ticker"]) <= stocks
    assert trades["filing_date"].max() <= data.load_returns().index[-1]
    assert not trades.duplicated(["accession", "ticker"]).any()
    assert (trades["trade_date"].dropna() <= trades.loc[trades["trade_date"].notna(), "filing_date"]).mean() > 0.99


def test_predecessor_registrants_are_mapped(trades):
    googl = trades[trades["ticker"] == "GOOGL"]
    assert googl["filing_date"].min() < pd.Timestamp("2007-01-01")      # Google Inc. before Alphabet
    assert not trades.loc[trades["ticker"] == "MRK", "issuer_name"].str.contains("SCHERING", case=False).any()
    meta = data.load_insider_metadata()
    assert set(meta["no_filings"]) <= {"ASML", "CCEP", "FER", "SHOP", "TRI", "NBIS", "PDD", "ARM"}  # foreign issuers


def test_missing_insider_files_raise_a_clear_error(tmp_path):
    with pytest.raises(data.DataError, match="download_insiders"):
        data.load_insider_market(tmp_path)
    with pytest.raises(data.DataError, match="download_insiders"):
        data.load_insider_metadata(tmp_path)


# --- signal ------------------------------------------------------------------------------------

def test_signal_is_point_in_time(market, signal):
    for cut in ["2012-01-31", "2020-03-31", "2024-06-30"]:
        partial = ins.market_signal(market.loc[:cut])
        assert np.allclose(partial.loc[cut], signal.loc[cut], equal_nan=True)


def test_breadth_peaks_at_market_lows(signal):
    assert signal["buy_share"].idxmax() in pd.date_range("2008-10-31", "2009-04-30", freq="ME")
    assert signal.loc["2009-03-31", "z"] > 1 and signal.loc["2020-05-31", "z"] > 1


def test_signal_at_caps_and_goes_stale(signal):
    month, share, z = ins.signal_at(signal, pd.Timestamp("2009-03-31"))
    assert month == pd.Timestamp("2009-03-31") and abs(z) <= ins.Z_CAP and 0 < share < 1
    assert ins.signal_at(signal, signal.index[-1] + pd.DateOffset(months=ins.STALE_MONTHS + 1)) == (None, None, None)
    assert ins.signal_at(signal, pd.Timestamp("2006-06-30")) == (None, None, None)   # z not defined yet


# --- view --------------------------------------------------------------------------------------

def test_view_is_equities_vs_bonds(base, signal):
    v = ins.build_view(base.prior_mu, signal.index[-1], signal)
    assert list(v.P.index) == [ins.MARKET_VIEW]
    row = v.P.iloc[0]
    assert row["SPY"] == 1 and row["AGG"] == -1 and row.abs().sum() == 2
    assert np.isclose(v.Q.iloc[0], base.prior_mu["SPY"] - base.prior_mu["AGG"] + ins.KAPPA * v.z)


def test_single_view_weighs_confidence_exactly(base, signal):
    date = signal["z"].abs().idxmax()
    for c in (0.25, 0.5, 0.75):
        e, v = ins.with_insider_view(base, date, signal, c)
        post_spread = e.benchmark_mu["SPY"] - e.benchmark_mu["AGG"]
        prior_spread = base.prior_mu["SPY"] - base.prior_mu["AGG"]
        assert np.isclose(post_spread, (1 - c) * prior_spread + c * v.Q.iloc[0])
        assert e.method == "insider" and e.cov is base.cov


def test_insider_buying_favours_equities(base, signal):
    bullish = signal["z"].idxmax()
    e, v = ins.with_insider_view(base, bullish, signal, 0.5)
    assert v.z > 0
    universe = data.load_universe()
    eq = universe.index[universe["asset_class"] == "Equity"].intersection(e.mu.index)
    bonds = universe.index[universe["asset_class"] == "Bond"].intersection(e.mu.index)
    move = e.mu - base.prior_mu.reindex(e.mu.index)
    # equities gain more than bonds (bonds may still gain a little: since 2022 they co-move with stocks)
    assert move[eq].mean() > max(move[bonds].mean(), 0)
    assert e.benchmark_mu["SPY"] - e.benchmark_mu["AGG"] > base.prior_mu["SPY"] - base.prior_mu["AGG"]


def test_no_view_without_data_and_errors(base, signal):
    late = signal.index[-1] + pd.DateOffset(months=ins.STALE_MONTHS + 2)
    e, v = ins.with_insider_view(base, late, signal, 0.5)
    assert v.P.empty and np.allclose(e.mu, base.prior_mu.reindex(e.mu.index))
    with pytest.raises(est.EstimationError, match="between 0 and"):
        ins.with_insider_view(base, signal.index[-1], signal, ins.MAX_CONFIDENCE + 0.05)
    plain = est.estimate(est.estimation_window(data.load_returns(), base.end), data.load_rf(),
                         est.shrinkage_groups(data.load_universe()))
    with pytest.raises(est.EstimationError, match="benchmark"):
        ins.with_insider_view(plain, signal.index[-1], signal, 0.5)


def test_backtest_views_use_only_past_filings(market, signal):
    """A trade at the close of April 2016 uses filings up to March 2016 only (one month of lag)."""
    d = pd.Timestamp("2016-04-30")
    returns = data.load_returns()
    e = btm.estimates_by_date(returns, data.load_rf(), est.shrinkage_groups(data.load_universe()), [d],
                              benchmarks=data.load_benchmark_returns())
    assert e[d].end == pd.Timestamp("2016-03-31")
    full = btm.with_insider_view(e, signal, 0.5)[d]
    past = btm.with_insider_view(e, ins.market_signal(market.loc[:"2016-03-31"]), 0.5)[d]
    assert np.allclose(full.mu, past.mu)


# --- evidence ----------------------------------------------------------------------------------

def test_evidence_tables(signal, trades):
    bench = data.load_benchmark_returns()
    q = ins.market_evidence(signal, (bench["SPY"] - bench["AGG"]).dropna())
    assert len(q) == 5 and q["months"].sum() > 150
    assert q["next_return"].iloc[-1] > q["next_return"].iloc[:-1].max()   # broadest buying: best next year
    ev = ins.purchase_event_study(trades, data.load_returns(), data.load_universe())
    assert list(ev.index) == ["3 months", "6 months", "12 months"] and (ev["events"] > 1000).all()
    assert ev[["t_universe", "t_sector"]].abs().max().max() < 3    # no reliable stock-level signal
    recent = ins.recent_purchases(trades, data.load_returns().index[-1])
    assert (recent["direction"] == "buy").all() and recent["filing_date"].is_monotonic_decreasing
