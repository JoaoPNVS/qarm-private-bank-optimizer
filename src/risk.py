"""Performance metrics and Monte Carlo wealth simulation (block bootstrap)."""
from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12
DUST = 1e-4  # solver residual weights ignored when rebuilding a portfolio's history


def drawdowns(returns: pd.Series) -> pd.Series:
    wealth = (1 + returns).cumprod()
    return wealth / wealth.cummax() - 1


def performance_metrics(returns: pd.Series, rf: pd.Series) -> dict:
    """Annualised return (geometric), volatility, Sharpe (mean excess return / volatility) and
    maximum drawdown of a monthly return series."""
    r = returns.dropna()
    if len(r) < 2:
        return {"annual_return": np.nan, "volatility": np.nan, "sharpe": np.nan, "max_drawdown": np.nan}
    excess = r - rf.reindex(r.index).fillna(0.0)
    vol = r.std() * np.sqrt(MONTHS_PER_YEAR)
    return {"annual_return": (1 + r).prod() ** (MONTHS_PER_YEAR / len(r)) - 1,
            "volatility": vol,
            "sharpe": excess.mean() * MONTHS_PER_YEAR / vol if vol > 0 else np.nan,
            "max_drawdown": drawdowns(r).min()}


def portfolio_history(weights: pd.Series, returns: pd.DataFrame) -> pd.Series:
    """Monthly returns of today's weights (rebalanced monthly) over the months where every held
    asset has a return, i.e. since the youngest holding started trading."""
    w = weights[weights > DUST]
    w = w / w.sum()
    r = returns[w.index].dropna()
    if r.empty:
        raise ValueError("The held assets have no common return history.")
    return (r @ w).rename("Portfolio")


def block_bootstrap(history: pd.DataFrame, horizon_months: int, n_sims: int = 5000, block: int = 12,
                    seed: int = 42) -> dict[str, np.ndarray]:
    """Simulated monthly returns (n_sims x horizon) for each column of `history`, built from
    blocks of `block` consecutive historical months drawn with replacement. All columns use the
    SAME blocks, so a portfolio and its benchmark are simulated on the same scenarios. Blocks
    keep the short-term dependence of returns (momentum, volatility clusters, crises)."""
    h = history.dropna()
    if len(h) < 2 * block:
        raise ValueError(f"At least {2 * block} months of common history are needed, got {len(h)}.")
    rng = np.random.default_rng(seed)
    n_blocks = -(-horizon_months // block)
    starts = rng.integers(0, len(h) - block + 1, size=(n_sims, n_blocks))
    idx = (starts[..., None] + np.arange(block)).reshape(n_sims, -1)[:, :horizon_months]
    return {c: h[c].values[idx] for c in h.columns}


def wealth_simulation(history: pd.DataFrame, initial_wealth: float, years: int, n_sims: int = 5000,
                      block: int = 12, seed: int = 42) -> dict:
    """For each column: wealth percentiles (5 / 50 / 95%) month by month, final-wealth percentiles
    and the probability of ending below the initial wealth."""
    sims = block_bootstrap(history, years * MONTHS_PER_YEAR, n_sims, block, seed)
    out = {}
    for name, r in sims.items():
        paths = initial_wealth * np.cumprod(1 + r, axis=1)
        paths = np.hstack([np.full((n_sims, 1), initial_wealth), paths])
        bands = pd.DataFrame(np.percentile(paths, [5, 50, 95], axis=0).T, columns=["p5", "p50", "p95"])
        final = paths[:, -1]
        out[name] = {"bands": bands, "median": float(np.median(final)), "p5": float(np.percentile(final, 5)),
                     "p95": float(np.percentile(final, 95)), "prob_loss": float((final < initial_wealth).mean())}
    return out
