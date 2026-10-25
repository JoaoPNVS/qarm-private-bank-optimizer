"""Benchmarks: SPY converted to CHF, and one SPY / Swiss-bond mix per profile.

Profile benchmark = equity_share x SPY (CHF) + (1 - equity_share) x Swiss government bonds 3-7y
(CSBGC7.SW), rebalanced monthly. Equity shares: Conservative 30%, Balanced 60%, Growth 80%,
Dynamic 100% (see profiling.SPECS). The bond leg has returns from 2008-02, so mixed benchmarks
start there; SPY (CHF) starts in 2003-10.

Each profile's volatility cap is calibrated on its benchmark: the benchmark's historical
volatility (2008-02 to 2026-09) rounded UP to the next full percent, so the optimised portfolio
takes the same risk as the benchmark it is compared with (tested in tests/test_benchmarks.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.estimators import Estimates

BOND_LEG = "CSBGC7.SW"


def profile_benchmark_returns(equity_share: float, spy: pd.Series, returns: pd.DataFrame) -> pd.Series:
    """Monthly returns of the SPY / CHF-bond mix (NaN-free, starts when every needed leg exists)."""
    if equity_share >= 1:
        return spy.dropna().rename("Benchmark")
    mix = equity_share * spy + (1 - equity_share) * returns[BOND_LEG]
    return mix.dropna().rename("Benchmark")


def _weights(equity_share: float, estimates: Estimates) -> pd.Series:
    if estimates.joint_cov is None:
        raise ValueError("Estimates were computed without the benchmark series.")
    spy = estimates.joint_cov.columns[-1]
    w = pd.Series(0.0, index=estimates.joint_cov.index)
    w[spy] = equity_share
    if equity_share < 1:
        if BOND_LEG not in w.index:
            raise ValueError(f"{BOND_LEG} has no complete history in the estimation window.")
        w[BOND_LEG] = 1 - equity_share
    return w


def ex_ante_stats(equity_share: float, estimates: Estimates) -> dict:
    """Expected excess return and volatility of the profile benchmark, same estimators as the portfolio."""
    w = _weights(equity_share, estimates)
    mu = pd.concat([estimates.mu, pd.Series({w.index[-1]: estimates.benchmark_mu})])
    ret = float(w @ mu.reindex(w.index))
    vol = float(np.sqrt(w @ estimates.joint_cov @ w))
    return {"expected_return": ret, "volatility": vol, "sharpe": ret / vol if vol > 0 else np.nan}


def tracking_error(weights: pd.Series, equity_share: float, estimates: Estimates) -> float:
    """Ex-ante tracking error of a portfolio (investable weights) versus the profile benchmark."""
    b = _weights(equity_share, estimates)
    a = weights.reindex(b.index).fillna(0.0) - b
    return float(np.sqrt(max(a @ estimates.joint_cov @ a, 0.0)))
