"""Expected-return and covariance estimators.

Conventions: monthly CHF returns in, ANNUALISED EXCESS returns (over the CHF risk-free rate)
and annualised covariance out. CASH_CHF is the risk-free asset: zero excess return, zero
variance, zero covariance.

Eligibility: an asset is used at a date only if it has a complete return history over the
whole estimation window (default 60 months) ending at that date. Recent IPOs therefore enter
the optimisation 60 months after their first monthly return.

Expected returns: historical mean over the window, shrunk toward the grand mean of the asset's
own class (equities toward the average equity, bonds toward the average bond):
    mu_i = (1 - a) * m_i + a * m_class(i)
Raw 60-month means are mostly noise (a 25%-volatility stock has a standard error of about 11%
p.a. on a 5-year mean); a long-only optimiser would pile into the luckiest past winners.
Shrinking halfway (a = 0.5 by default) keeps the ranking of past performance but halves the
dispersion. Shrinking within each class matters: toward the global mean, low-risk bonds would
inherit equity-like returns and absorb the portfolio.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from src.data import CASH_TICKER

MONTHS_PER_YEAR = 12
DEFAULT_WINDOW = 60
DEFAULT_MEAN_SHRINKAGE = 0.5


class EstimationError(ValueError):
    """Not enough clean data to estimate the inputs."""


@dataclass
class Estimates:
    mu: pd.Series          # annualised expected excess returns (investable assets)
    cov: pd.DataFrame      # annualised covariance (investable assets)
    mean_shrinkage: float  # weight on the class grand mean
    cov_shrinkage: float   # Ledoit-Wolf intensity on correlations
    start: pd.Timestamp
    end: pd.Timestamp
    benchmark_mu: float = np.nan              # expected excess return of the benchmark series (SPY)
    joint_cov: pd.DataFrame | None = None     # covariance of investable assets + benchmark series


def estimation_window(returns: pd.DataFrame, end: pd.Timestamp, months: int = DEFAULT_WINDOW,
                      tickers: list[str] | None = None) -> pd.DataFrame:
    """Last `months` monthly returns up to `end` (inclusive), restricted to the assets with a
    complete history over the window (the eligible assets at that date)."""
    sub = returns.loc[:end, tickers or list(returns.columns)].tail(months)
    if len(sub) < months:
        raise EstimationError(f"Only {len(sub)} months available before {end:%Y-%m}, need {months}.")
    sub = sub.loc[:, sub.notna().all()]
    if sub.shape[1] == 0 or list(sub.columns) == [CASH_TICKER]:
        raise EstimationError(f"No investable asset has {months} months of history at {end:%Y-%m}.")
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
    therefore preserved exactly. With more assets than months the sample covariance is
    singular; the shrunk matrix is always well conditioned.
    Why not shrink the raw covariance: its identity target has the AVERAGE variance of the
    universe on the diagonal, which inflates low-risk assets (a Swiss govt 0-3y ETF would go
    from ~1.6% to ~5% volatility next to equities).
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


def shrunk_mean(excess: pd.DataFrame, asset_class: pd.Series,
                shrinkage: float = DEFAULT_MEAN_SHRINKAGE) -> pd.Series:
    """Annualised sample mean shrunk toward the grand mean of each asset class.

    shrinkage = 0 gives raw historical means, 1 gives every asset its class average.
    """
    if not 0 <= shrinkage <= 1:
        raise EstimationError("Shrinkage must be between 0 and 1.")
    m = excess.mean() * MONTHS_PER_YEAR
    cls = asset_class.reindex(m.index)
    target = m.groupby(cls).transform("mean")
    mu = (1 - shrinkage) * m + shrinkage * target
    if CASH_TICKER in mu:
        mu[CASH_TICKER] = 0.0
    return mu.rename("mu")


def estimate(returns: pd.DataFrame, rf: pd.Series, asset_class: pd.Series,
             mean_shrinkage: float = DEFAULT_MEAN_SHRINKAGE,
             benchmark: pd.Series | None = None) -> Estimates:
    """Expected excess returns and covariance from one estimation window of monthly returns.

    `benchmark` (e.g. SPY in CHF, not investable) is estimated jointly so the portfolio and its
    benchmark are compared with the same method: its covariances come from the same Ledoit-Wolf
    fit and its mean is shrunk toward the equity grand mean (computed on investable stocks only).
    """
    if returns.isna().any().any():
        raise EstimationError("The estimation window contains missing returns.")
    ex = excess_returns(returns, rf)
    cov, lw_shrinkage = ledoit_wolf_cov(ex)
    mu = shrunk_mean(ex, asset_class, mean_shrinkage)
    out = Estimates(mu, cov, mean_shrinkage, lw_shrinkage, ex.index[0], ex.index[-1])
    if benchmark is not None:
        b = benchmark.reindex(ex.index)
        if b.isna().any():
            raise EstimationError("Benchmark returns missing in the estimation window.")
        name = str(benchmark.name)
        b_ex = (b - rf.reindex(ex.index)).rename(name)
        out.joint_cov, _ = ledoit_wolf_cov(pd.concat([ex, b_ex], axis=1))
        eq_mean = (ex.mean() * MONTHS_PER_YEAR)[asset_class.reindex(ex.columns) == "Equity"].mean()
        out.benchmark_mu = float((1 - mean_shrinkage) * b_ex.mean() * MONTHS_PER_YEAR + mean_shrinkage * eq_mean)
    return out
