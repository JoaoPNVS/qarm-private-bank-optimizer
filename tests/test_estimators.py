import numpy as np
import pandas as pd
import pytest

from src import data, estimators as est


@pytest.fixture(scope="module")
def window():
    ret = data.load_returns()
    return est.estimation_window(ret, ret.index[-1], 60), data.load_rf()


@pytest.fixture(scope="module")
def estimates(window):
    return est.estimate(*window)


def test_reference_weights_valid():
    uni = data.load_universe()
    assert np.isclose(sum(est.REFERENCE_WEIGHTS.values()), 1.0)
    assert set(est.REFERENCE_WEIGHTS) <= set(uni.index)
    by_class = pd.Series(est.REFERENCE_WEIGHTS).groupby(uni["asset_class"]).sum()
    assert np.isclose(by_class["Equity"], 0.50) and np.isclose(by_class["Bond"], 0.40)


def test_cov_psd_and_preserves_volatility(window, estimates):
    cov = estimates.cov
    assert np.allclose(cov, cov.T)
    assert np.linalg.eigvalsh(cov.values).min() > -1e-10
    risky = [c for c in cov if c != data.CASH_TICKER]
    ex = est.excess_returns(*window)
    assert np.allclose(np.diag(cov.loc[risky, risky]), ex[risky].var() * 12)
    assert (cov.loc[data.CASH_TICKER] == 0).all()
    assert 0 < estimates.shrinkage < 1


def test_equilibrium_prices_reference_at_target_sharpe(estimates):
    w, mu, cov = estimates.w_ref, estimates.mu, estimates.cov
    sharpe = w @ mu / np.sqrt(w @ cov @ w)
    assert np.isclose(sharpe, est.REFERENCE_SHARPE)
    assert estimates.mu[data.CASH_TICKER] == 0
    assert (estimates.mu.drop(data.CASH_TICKER) > 0).all()


def test_shrunk_mean_limits(window):
    ex = est.excess_returns(*window)
    raw = est.shrunk_mean(ex, 0.0)
    assert np.allclose(raw.drop(data.CASH_TICKER), ex.drop(columns=data.CASH_TICKER).mean() * 12)
    full = est.shrunk_mean(ex, 1.0).drop(data.CASH_TICKER)
    assert np.allclose(full, full.iloc[0])


@pytest.fixture(scope="module")
def universe():
    return data.load_universe()


def test_no_views_gives_equilibrium_exactly(window, estimates, universe):
    for views in (None, [], [est.View("off", {"ticker": "GLD"}, 0.05, active=False)]):
        e = est.estimate(*window, views=views, universe=universe)
        assert e.mu.equals(estimates.mu)


def test_full_confidence_view_holds_exactly(window, universe):
    v = est.View("CH - US = 2%", {"asset_class": "Equity", "region": "Switzerland"}, 0.02, 1.0,
                 short={"asset_class": "Equity", "region": "US"})
    e = est.estimate(*window, views=[v], universe=universe)
    assert np.isclose(e.views.loc[v.name, "posterior"], 0.02)


def test_views_move_posterior_toward_q(window, universe):
    e = est.estimate(*window, views=est.HOUSE_VIEWS, universe=universe)
    t = e.views
    assert len(t) == len(est.HOUSE_VIEWS)
    # posterior lies between prior and the view for each (non-conflicting) house view
    lo, hi = np.minimum(t["prior"], t["view"]), np.maximum(t["prior"], t["view"])
    assert ((t["posterior"] >= lo - 1e-9) & (t["posterior"] <= hi + 1e-9)).all()
    assert e.mu[data.CASH_TICKER] == 0


def test_lower_confidence_moves_less(window, universe):
    def post(c):
        v = est.View("gold", {"ticker": "GLD"}, 0.08, c)
        return est.estimate(*window, views=[v], universe=universe).mu["GLD"]
    assert est.estimate(*window).mu["GLD"] < post(0.2) < post(0.8) < 0.08 + 1e-12


def test_view_basket_weights(universe):
    w_ref = est.reference_weights(list(universe.index))
    b = est.basket({"asset_class": "Equity", "region": "Switzerland"}, universe, w_ref)
    assert np.isclose(b.sum(), 1) and np.isclose(b["CSSMI.SW"], 0.8)  # 8% / (8% + 2%)
    assert np.isclose(est.basket({"ticker": "NESN.SW"}, universe, w_ref)["NESN.SW"], 1)  # no ref weight
    with pytest.raises(est.EstimationError):
        est.basket({"region": "Mars"}, universe, w_ref)


def test_views_rejected_with_shrunk_mean(window, universe):
    with pytest.raises(est.EstimationError, match="equilibrium"):
        est.estimate(*window, method="shrunk_mean", views=est.HOUSE_VIEWS, universe=universe)


def test_window_errors():
    ret = data.load_returns()
    with pytest.raises(est.EstimationError, match="Incomplete"):
        est.estimation_window(ret, pd.Timestamp("2011-06-30"), 36)
    with pytest.raises(est.EstimationError, match="Unknown"):
        est.estimate(ret.iloc[-60:], data.load_rf(), method="magic")
