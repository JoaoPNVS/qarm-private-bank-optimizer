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
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from src.data import CASH_TICKER

MONTHS_PER_YEAR = 12

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
class Estimates:
    mu: pd.Series          # annualised expected excess returns
    cov: pd.DataFrame      # annualised covariance
    method: str
    delta: float           # market risk aversion implied by the reference portfolio
    w_ref: pd.Series       # reference portfolio used for reverse optimisation
    shrinkage: float       # Ledoit-Wolf intensity on correlations
    start: pd.Timestamp
    end: pd.Timestamp


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
    """Annualised covariance: Ledoit-Wolf shrinkage applied to the correlation matrix.

    Returns are standardised before fitting, so the shrinkage pulls correlations toward zero
    while each asset keeps its own sample volatility. (Shrinking the raw covariance toward a
    scaled identity would inflate the volatility of low-risk CHF bonds several-fold.)
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


def estimate(returns: pd.DataFrame, rf: pd.Series, method: str = "equilibrium",
             mean_shrinkage: float = 0.5) -> Estimates:
    """Expected excess returns and covariance from one estimation window of monthly returns."""
    if returns.isna().any().any():
        raise EstimationError("The estimation window contains missing returns.")
    ex = excess_returns(returns, rf)
    cov, lw_shrinkage = ledoit_wolf_cov(ex)
    w_ref = reference_weights(list(ex.columns))
    delta = implied_risk_aversion(cov, w_ref)
    if method == "equilibrium":
        mu = equilibrium_returns(cov, w_ref, delta)
    elif method == "shrunk_mean":
        mu = shrunk_mean(ex, mean_shrinkage)
    else:
        raise EstimationError(f"Unknown expected-return method '{method}'.")
    return Estimates(mu, cov, method, delta, w_ref, lw_shrinkage, ex.index[0], ex.index[-1])
