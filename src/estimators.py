"""Expected-return and covariance estimators.

Conventions: monthly CHF returns in, ANNUALISED EXCESS returns (over the CHF risk-free rate)
and annualised covariance out. CASH_CHF is the risk-free asset: zero excess return, zero
variance, zero covariance.

Default expected returns = equilibrium-implied returns (reverse optimisation, Black-Litterman
prior): mu = delta * Sigma * w_ref. Raw historical means are far too noisy to be fed to a
mean-variance optimiser: it concentrates on whatever did best in the window. Over 2011-2026 the
highest raw means are QQQ, Sika and ABB, and a long-only optimiser puts ~70% in QQQ alone; with
the default 60-month window the winners change (UBS, ABB, gold take ~97%). Equilibrium returns are, by
construction, the returns that make the reference portfolio optimal, so the optimiser starts
from a diversified allocation and only deviates from it because of the client's constraints.
The shrunk historical mean (toward the cross-sectional grand mean) is kept as an option.

House views (Black-Litterman): the CIO's views are blended with the equilibrium prior,
    mu_BL = Pi + tau Sigma P' (tau P Sigma P' + Omega)^-1 (Q - P Pi),
with one row of P per view (absolute: a basket; relative: long basket minus short basket).
Baskets select assets by ticker or by tags (region, asset class, ...) and are weighted by the
reference portfolio inside the basket (equal weights if the basket holds none of it).
View uncertainty is set from the confidence c in (0, 1]:
    Omega_k = (1/c_k - 1) * tau * p_k' Sigma p_k
so c = 50% gives the classic He-Litterman choice Omega = tau P Sigma P', c = 100% imposes the
view exactly and c -> 0 ignores it. With Omega proportional to tau, tau cancels out of mu_BL;
it is kept (TAU = 0.05) only for the standard notation. The covariance is left unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from src.data import CASH_TICKER

MONTHS_PER_YEAR = 12
TAU = 0.05

# Reference (market) portfolio for reverse optimisation: a global balanced 50/40/10 portfolio
# for a CHF investor. Equity weights follow approximate relative market capitalisation (US
# largest, EM ~10% of equities) with Switzerland represented by index ETFs only (single stocks
# are already inside the SMI); bonds are half Swiss government, half global; 10% real assets.
REFERENCE_WEIGHTS = {
    # Equity 50%
    "SPY": 0.22, "IWM": 0.02, "EZU": 0.05, "EWU": 0.02, "EWJ": 0.03, "EPP": 0.01, "EEM": 0.05,
    "CSSMI.SW": 0.08, "CSSMIM.SW": 0.02,
    # Bonds 40%
    "CSBGC3.SW": 0.05, "CSBGC7.SW": 0.08, "CSBGC0.SW": 0.07,
    "IEF": 0.05, "TLT": 0.02, "TIP": 0.02, "LQD": 0.04, "HYG": 0.01, "EMB": 0.02, "BWX": 0.04,
    # Real assets 10%
    "GLD": 0.04, "DBC": 0.01, "VNQ": 0.015, "RWX": 0.015, "SRECHA.SW": 0.02,
}

# Sharpe ratio assumed for the reference portfolio; sets the market risk aversion
# delta = Sharpe / sigma_ref (so that the reference earns Sharpe * sigma_ref in excess).
REFERENCE_SHARPE = 0.35


class EstimationError(ValueError):
    """Not enough clean data to estimate the inputs."""


@dataclass
class View:
    """A CIO view. Absolute: E[r_long] = value. Relative: E[r_long] - E[r_short] = value.
    Values are annualised excess returns; selectors map a universe column (or "ticker") to a
    value or list of values, e.g. {"asset_class": "Equity", "region": "Switzerland"}."""
    name: str
    long: dict
    value: float
    confidence: float = 0.5
    short: dict | None = None
    active: bool = True


# Default "House View" for the demo (switch off with active=False or pass views=[]).
HOUSE_VIEWS = [
    View("Swiss equities outperform US equities by 1% p.a.",
         {"asset_class": "Equity", "region": "Switzerland"}, 0.010, 0.50,
         short={"asset_class": "Equity", "region": "US"}),
    View("Eurozone and Japanese equities outperform US equities by 0.5% p.a.",
         {"ticker": ["EZU", "EWJ"]}, 0.005, 0.30, short={"asset_class": "Equity", "region": "US"}),
    View("Gold earns 3% p.a. above SARON", {"ticker": "GLD"}, 0.030, 0.30),
    View("Swiss government bonds earn 0.5% p.a. above SARON",
         {"asset_class": "Bond", "region": "Switzerland"}, 0.005, 0.50),
]


@dataclass
class Estimates:
    mu: pd.Series          # annualised expected excess returns (posterior if views are used)
    cov: pd.DataFrame      # annualised covariance
    method: str
    delta: float           # market risk aversion implied by the reference portfolio
    w_ref: pd.Series       # reference portfolio used for reverse optimisation
    shrinkage: float       # Ledoit-Wolf intensity on correlations
    start: pd.Timestamp
    end: pd.Timestamp
    prior: pd.Series | None = None                      # equilibrium returns before views
    views: pd.DataFrame = field(default_factory=pd.DataFrame)  # view, Q, prior and posterior


def estimation_window(returns: pd.DataFrame, end: pd.Timestamp, months: int,
                      tickers: list[str] | None = None) -> pd.DataFrame:
    """Last `months` monthly returns up to `end` (inclusive) for assets with a complete history."""
    sub = returns.loc[:end, tickers or list(returns.columns)].tail(months)
    if len(sub) < months:
        raise EstimationError(f"Only {len(sub)} months available before {end:%Y-%m}, need {months}.")
    incomplete = sub.columns[sub.isna().any()].tolist()
    if incomplete:
        raise EstimationError(f"Incomplete history in the estimation window for {incomplete}.")
    return sub


def excess_returns(returns: pd.DataFrame, rf: pd.Series) -> pd.DataFrame:
    rf = rf.reindex(returns.index)
    if rf.isna().any():
        raise EstimationError("Risk-free rate missing for some months of the estimation window.")
    ex = returns.sub(rf, axis=0)
    if CASH_TICKER in ex:
        ex[CASH_TICKER] = 0.0
    return ex


def _risky(columns) -> list[str]:
    return [c for c in columns if c != CASH_TICKER]


def ledoit_wolf_cov(excess: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Annualised covariance: Ledoit-Wolf shrinkage applied to the CORRELATION matrix.

    Returns are standardised (each asset divided by its sample volatility) before fitting
    sklearn's LedoitWolf, whose target is a scaled identity. On standardised data that target
    is "all correlations zero, all variances equal to one", so the shrinkage only pulls
    correlations toward zero; the result is rescaled by the sample volatilities, which are
    therefore preserved exactly.
    Why not shrink the raw covariance: its identity target has the AVERAGE variance of the
    universe on the diagonal, which inflates low-risk assets. On the 60 months to 2026-09 the
    Swiss govt 0-3y ETF would go from 1.6% to 5.3% volatility (3-7y: 3.4% -> 6.0%), making CHF
    bonds look as risky as gold to the optimiser.
    """
    risky = _risky(excess.columns)
    x = excess[risky]
    vol = x.std(ddof=1)
    if (vol <= 0).any():
        raise EstimationError(f"Zero volatility for {vol[vol <= 0].index.tolist()}.")
    z = (x - x.mean()) / vol
    lw = LedoitWolf(assume_centered=True).fit(z.values)
    d = np.sqrt(np.diag(lw.covariance_))
    corr = lw.covariance_ / np.outer(d, d)
    cov = pd.DataFrame(corr * np.outer(vol, vol) * MONTHS_PER_YEAR, index=risky, columns=risky)
    cov = cov.reindex(index=excess.columns, columns=excess.columns, fill_value=0.0)
    return cov, float(lw.shrinkage_)


def reference_weights(tickers: list[str]) -> pd.Series:
    """REFERENCE_WEIGHTS restricted to `tickers` and renormalised to 100%."""
    w = pd.Series(REFERENCE_WEIGHTS, dtype=float).reindex(tickers).fillna(0.0)
    if w.sum() <= 0:
        raise EstimationError("None of the selected assets belongs to the reference portfolio.")
    return w / w.sum()


def implied_risk_aversion(cov: pd.DataFrame, w_ref: pd.Series,
                          sharpe: float = REFERENCE_SHARPE) -> float:
    sigma = float(np.sqrt(w_ref @ cov @ w_ref))
    return sharpe / sigma


def equilibrium_returns(cov: pd.DataFrame, w_ref: pd.Series, delta: float) -> pd.Series:
    """Reverse optimisation: the excess returns for which w_ref is the mean-variance optimum."""
    return (delta * cov @ w_ref).rename("mu")


def shrunk_mean(excess: pd.DataFrame, shrinkage: float = 0.5) -> pd.Series:
    """Sample mean shrunk toward the grand mean of risky assets (James-Stein style).

    shrinkage = 0 gives raw historical means, 1 gives the same return for every risky asset.
    """
    if not 0 <= shrinkage <= 1:
        raise EstimationError("Shrinkage must be between 0 and 1.")
    m = excess.mean() * MONTHS_PER_YEAR
    risky = _risky(excess.columns)
    mu = (1 - shrinkage) * m + shrinkage * m[risky].mean()
    if CASH_TICKER in mu:
        mu[CASH_TICKER] = 0.0
    return mu.rename("mu")


def basket(selector: dict, universe: pd.DataFrame, w_ref: pd.Series) -> pd.Series:
    """Weights of a view basket over w_ref.index (reference-weighted, equal if no reference weight)."""
    tickers = w_ref.index
    mask = pd.Series(True, index=tickers)
    for col, val in selector.items():
        values = list(val) if isinstance(val, (list, tuple, set)) else [val]
        if col == "ticker":
            mask &= tickers.isin(values)
        elif col in universe.columns:
            mask &= universe.loc[tickers, col].isin(values)
        else:
            raise EstimationError(f"Unknown view selector '{col}'.")
    if not mask.any():
        raise EstimationError(f"View basket {selector} matches no selected asset.")
    w = w_ref[mask] if w_ref[mask].sum() > 0 else mask[mask].astype(float)
    return (w / w.sum()).reindex(tickers, fill_value=0.0)


def view_matrices(views: list[View], universe: pd.DataFrame, w_ref: pd.Series):
    """P (views x assets), Q and confidences for the active views."""
    active = [v for v in views if v.active]
    rows = []
    for v in active:
        if not 0 < v.confidence <= 1:
            raise EstimationError(f"Confidence of view '{v.name}' must be in (0, 100%].")
        p = basket(v.long, universe, w_ref)
        if v.short is not None:
            p = p - basket(v.short, universe, w_ref)
        rows.append(p)
    names = [v.name for v in active]
    P = pd.DataFrame(rows, index=names, columns=w_ref.index)
    return P, pd.Series([v.value for v in active], index=names), \
        pd.Series([v.confidence for v in active], index=names)


def black_litterman(prior: pd.Series, cov: pd.DataFrame, P: pd.DataFrame, Q: pd.Series,
                    confidence: pd.Series, tau: float = TAU) -> pd.Series:
    """Posterior expected returns (formula and Omega convention in the module docstring)."""
    if P.empty:
        return prior.copy()
    S, Pm = cov.values, P.values
    psp = Pm @ S @ Pm.T
    omega = np.diag((1 / confidence.values - 1) * tau * np.diag(psp))
    try:
        adj = np.linalg.solve(tau * psp + omega, Q.values - Pm @ prior.values)
    except np.linalg.LinAlgError as exc:
        raise EstimationError("Views are redundant or contradictory at 100% confidence.") from exc
    return pd.Series(prior.values + tau * S @ Pm.T @ adj, index=prior.index, name="mu")


def estimate(returns: pd.DataFrame, rf: pd.Series, method: str = "equilibrium",
             mean_shrinkage: float = 0.5, views: list[View] | None = None,
             universe: pd.DataFrame | None = None) -> Estimates:
    """Expected excess returns and covariance from one estimation window of monthly returns.
    `views` (Black-Litterman) apply to the equilibrium prior only and need `universe` tags."""
    if returns.isna().any().any():
        raise EstimationError("The estimation window contains missing returns.")
    ex = excess_returns(returns, rf)
    cov, lw_shrinkage = ledoit_wolf_cov(ex)
    w_ref = reference_weights(list(ex.columns))
    delta = implied_risk_aversion(cov, w_ref)
    view_table = pd.DataFrame()
    if method == "equilibrium":
        prior = equilibrium_returns(cov, w_ref, delta)
        mu = prior
        if views and any(v.active for v in views):
            if universe is None:
                raise EstimationError("Views need the universe tags.")
            P, Q, conf = view_matrices(views, universe, w_ref)
            mu = black_litterman(prior, cov, P, Q, conf)
            view_table = pd.DataFrame({"view": Q, "confidence": conf,
                                       "prior": P.values @ prior.values, "posterior": P.values @ mu.values})
    elif method == "shrunk_mean":
        if views and any(v.active for v in views):
            raise EstimationError("House views can only be combined with equilibrium returns.")
        prior, mu = None, shrunk_mean(ex, mean_shrinkage)
    else:
        raise EstimationError(f"Unknown expected-return method '{method}'.")
    if CASH_TICKER in mu:
        mu[CASH_TICKER] = 0.0
    return Estimates(mu, cov, method, delta, w_ref, lw_shrinkage, ex.index[0], ex.index[-1],
                     prior, view_table)
