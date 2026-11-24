import numpy as np
import pandas as pd
import pytest

from src import data, estimators as est


@pytest.fixture(scope="module")
def universe():
    return data.load_universe()


@pytest.fixture(scope="module")
def window():
    ret = data.load_returns()
    return est.estimation_window(ret, ret.index[-1], 60), data.load_rf()


@pytest.fixture(scope="module")
def estimates(window, universe):
    return est.estimate(*window, est.shrinkage_groups(universe))


def test_cov_psd_and_preserves_volatility(window, estimates):
    cov = estimates.cov
    assert np.allclose(cov, cov.T)
    assert np.linalg.eigvalsh(cov.values).min() > -1e-10
    risky = [c for c in cov if c != data.CASH_TICKER]
    ex = est.excess_returns(*window)
    assert np.allclose(np.diag(cov.loc[risky, risky]), ex[risky].var() * 12)
    assert (cov.loc[data.CASH_TICKER] == 0).all()
    assert 0 < estimates.cov_shrinkage < 1


def test_shrinkage_groups(universe):
    g = est.shrinkage_groups(universe)
    assert g["AAPL"] == "Equity" and g["GLD"] == "Commodity" and g["BIL"] == "Cash" and g["FLOT"] == "Cash"
    assert g["SHY"] == "Bond Short" and g["IEF"] == "Bond Intermediate" and g["TLT"] == "Bond Long"


def test_shrinkage_toward_group_mean(window, universe):
    ex = est.excess_returns(*window)
    cls = est.shrinkage_groups(universe)
    raw = est.shrunk_mean(ex, cls, 0.0)
    assert np.allclose(raw.drop(data.CASH_TICKER), ex.drop(columns=data.CASH_TICKER).mean() * 12)
    full = est.shrunk_mean(ex, cls, 1.0)
    for c in set(cls.reindex(full.index)) - {"Cash"}:
        members = full[cls.reindex(full.index) == c]
        assert np.allclose(members, members.iloc[0])  # every asset gets its class average
    half = est.shrunk_mean(ex, cls, 0.5)
    assert np.allclose(half, (raw + full) / 2)
    assert half[data.CASH_TICKER] == 0


def test_short_bonds_not_dragged_by_long_bonds(window, universe):
    # whole-class shrinkage pulled short funds toward the 2022 losses of long bonds
    ex = est.excess_returns(*window)
    by_bucket = est.shrunk_mean(ex, est.shrinkage_groups(universe))
    whole_class = est.shrunk_mean(ex, universe["asset_class"])
    assert by_bucket["SHY"] > whole_class["SHY"] and by_bucket["TLT"] < whole_class["TLT"]


def test_shrinkage_keeps_bonds_below_equities(estimates, universe):
    cls = universe["asset_class"].reindex(estimates.mu.index)
    assert estimates.mu[cls == "Bond"].mean() < estimates.mu[cls == "Equity"].mean()


def test_eligibility_drops_short_histories():
    ret = data.load_returns()
    early = est.estimation_window(ret, pd.Timestamp("2015-06-30"), 60)
    later = est.estimation_window(ret, pd.Timestamp("2018-06-30"), 60)
    assert "META" not in early.columns and "META" in later.columns  # IPO 2012-05
    assert "AAPL" in early.columns and not early.isna().any().any()


def test_window_errors():
    ret = data.load_returns()
    with pytest.raises(est.EstimationError, match="Only"):
        est.estimation_window(ret, pd.Timestamp("1996-06-30"), 60)
    with pytest.raises(est.EstimationError, match="Shrinkage"):
        est.shrunk_mean(ret.iloc[-60:].fillna(0), est.shrinkage_groups(data.load_universe()), 1.5)


# --- Black-Litterman -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bl_base(window, universe):
    from src import benchmarks as bm
    return est.estimate(*window, est.shrinkage_groups(universe), benchmarks=data.load_benchmark_returns(),
                        benchmark_class=bm.BENCHMARK_CLASS)


def test_equilibrium_prior_is_reverse_optimisation(bl_base):
    jc, pi = bl_base.joint_cov, bl_base.prior_mu
    expected = est.BL_DELTA * (0.6 * jc["SPY"] + 0.4 * jc["AGG"])
    assert np.allclose(pi, expected.reindex(pi.index))
    assert pi[data.CASH_TICKER] == 0
    bonds = data.load_universe().query("asset_class == 'Bond'").index.intersection(pi.index)
    assert (pi[bonds] > 0).all()                       # bonds earn a premium in equilibrium...
    assert (bl_base.sample_mu[bonds] < 0).all()        # ...although they lost to cash over the window


def test_zero_confidence_gives_the_prior(bl_base):
    e = est.with_view_confidence(bl_base, 0.0)
    assert np.allclose(e.mu, bl_base.prior_mu.reindex(e.mu.index))
    assert np.allclose(e.benchmark_mu, bl_base.prior_mu[["SPY", "AGG"]])
    assert e.method == "black_litterman" and e.view_confidence == 0.0
    assert e.cov is bl_base.cov and e.joint_cov is bl_base.joint_cov   # nothing re-estimated


def test_uncorrelated_assets_blend_linearly():
    idx = ["A", "B", data.CASH_TICKER]
    cov = pd.DataFrame(np.diag([0.04, 0.01, 0.0]), index=idx, columns=idx)
    prior, sample = pd.Series([0.05, 0.02, 0.0], index=idx), pd.Series([0.20, -0.04, 0.0], index=idx)
    for c in (0.1, 0.25):
        mu = est.black_litterman_mu(prior, sample, cov, c)
        assert np.allclose(mu, (1 - c) * prior + c * sample)


def test_correlated_views_reinforce_each_other(bl_base):
    c = 0.10
    post = est.with_view_confidence(bl_base, c)
    spy = (post.benchmark_mu["SPY"] - bl_base.prior_mu["SPY"]) / (bl_base.sample_mu["SPY"] - bl_base.prior_mu["SPY"])
    assert spy > 3 * c              # far more than the 10% an isolated view would get


def test_more_confidence_moves_toward_history(bl_base):
    gap = lambda c: (est.with_view_confidence(bl_base, c).mu - bl_base.sample_mu.reindex(bl_base.mu.index)).abs().mean()
    assert gap(0.0) > gap(0.10) > gap(0.25)


def test_black_litterman_errors(window, universe, bl_base):
    plain = est.estimate(*window, est.shrinkage_groups(universe))
    with pytest.raises(est.EstimationError, match="benchmark"):
        est.with_view_confidence(plain, 0.1)
    with pytest.raises(est.EstimationError, match="between 0 and"):
        est.with_view_confidence(bl_base, est.MAX_VIEW_CONFIDENCE + 0.05)


def test_general_black_litterman_matches_one_view_per_asset(bl_base):
    pi, cov = bl_base.prior_mu, bl_base.joint_cov
    sample = bl_base.sample_mu.reindex(pi.index)
    risky = [t for t in pi.index if t != data.CASH_TICKER]
    P = pd.DataFrame(np.eye(len(risky)), index=risky, columns=risky)
    for c in (0.05, 0.10, 0.25):
        general = est.black_litterman_posterior(pi, cov, P, sample[risky], c)
        assert np.allclose(general, est.black_litterman_mu(pi, sample, cov, c))


def test_relative_view_blends_the_spread():
    idx = ["A", "B", data.CASH_TICKER]
    cov = pd.DataFrame(np.diag([0.04, 0.01, 0.0]), index=idx, columns=idx)
    prior = pd.Series([0.05, 0.02, 0.0], index=idx)
    P = pd.DataFrame([[1.0, -1.0, 0.0]], index=["A over B"], columns=idx)
    for c in (0.1, 0.5):
        mu = est.black_litterman_posterior(prior, cov, P, pd.Series({"A over B": 0.10}), c)
        assert np.isclose(mu["A"] - mu["B"], (1 - c) * 0.03 + c * 0.10)   # isolated view: weight c
        assert mu[data.CASH_TICKER] == 0


def test_general_black_litterman_edge_cases(bl_base):
    pi, cov = bl_base.prior_mu, bl_base.joint_cov
    P = pd.DataFrame([[1.0, -1.0]], index=["v"], columns=["SPY", "AGG"])
    q = pd.Series({"v": 0.2})
    assert np.allclose(est.black_litterman_posterior(pi, cov, P, q, 0.0), pi)
    assert np.allclose(est.black_litterman_posterior(pi, cov, P.iloc[:0], q.iloc[:0], 0.5), pi)
    with pytest.raises(est.EstimationError, match="below 100%"):
        est.black_litterman_posterior(pi, cov, P, q, 1.0)
    cash = pd.DataFrame([[1.0]], index=["c"], columns=[data.CASH_TICKER])
    with pytest.raises(est.EstimationError, match="cash"):
        est.black_litterman_posterior(pi, cov, cash, pd.Series({"c": 0.1}), 0.5)
