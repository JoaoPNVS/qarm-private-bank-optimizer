"""Benchmarks: SPY, and one SPY / US-bond mix per profile (USD, not investable).

Profile benchmark = equity_share x SPY + (1 - equity_share) x AGG (US aggregate bonds),
rebalanced monthly. Equity shares: Conservative 30%, Balanced 60%, Growth 80%, Dynamic 100%
(see profiling.SPECS). AGG has returns from 2003-10, so mixed benchmarks start there; SPY
starts in 1998-02.

Each profile's volatility cap is calibrated on its benchmark: the benchmark's historical
volatility rounded UP to the next full percent, so the optimised portfolio takes the same risk
as the benchmark it is compared with (tested in tests/test_benchmarks.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.estimators import Estimates

EQUITY_LEG, BOND_LEG = "SPY", "AGG"
BENCHMARK_CLASS = {EQUITY_LEG: "Equity", BOND_LEG: "Bond"}


def profile_benchmark_returns(equity_share: float, benchmarks: pd.DataFrame) -> pd.Series:
    """Monthly returns of the SPY / AGG mix (NaN-free, starts when every needed leg exists)."""
    if equity_share >= 1:
        return benchmarks[EQUITY_LEG].dropna().rename("Benchmark")
    mix = equity_share * benchmarks[EQUITY_LEG] + (1 - equity_share) * benchmarks[BOND_LEG]
    return mix.dropna().rename("Benchmark")


def _weights(equity_share: float, estimates: Estimates) -> pd.Series:
    if estimates.joint_cov is None:
        raise ValueError("Estimates were computed without the benchmark series.")
    w = pd.Series(0.0, index=estimates.joint_cov.index)
    w[EQUITY_LEG] = equity_share
    w[BOND_LEG] = 1 - equity_share
    return w


def ex_ante_stats(equity_share: float, estimates: Estimates) -> dict:
    """Expected excess return and volatility of the profile benchmark, same estimators as the portfolio."""
    w = _weights(equity_share, estimates)
    mu = pd.concat([estimates.mu, estimates.benchmark_mu]).reindex(w.index)
    ret = float(w @ mu)
    vol = float(np.sqrt(w @ estimates.joint_cov @ w))
    return {"expected_return": ret, "volatility": vol, "sharpe": ret / vol if vol > 0 else np.nan}


def tracking_error(weights: pd.Series, equity_share: float, estimates: Estimates) -> float:
    """Ex-ante tracking error of a portfolio (investable weights) versus the profile benchmark."""
    b = _weights(equity_share, estimates)
    a = weights.reindex(b.index).fillna(0.0) - b
    return float(np.sqrt(max(a @ estimates.joint_cov @ a, 0.0)))
