"""Performance metrics and Monte Carlo wealth simulation (block bootstrap)."""
from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12
TAIL = 0.05  # value at risk / expected shortfall level (worst 5% of simulated years)


def drawdowns(returns: pd.Series) -> pd.Series:
    """Drawdowns from the highest wealth reached, including the initial capital."""
    wealth = (1 + returns).cumprod()
    return wealth / wealth.cummax().clip(lower=1.0) - 1


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
                      block: int = 12, seed: int = 42, year_loss_limit: float = 0.10) -> dict:
    """For each column: wealth percentiles (5 / 50 / 95%) month by month, final-wealth percentiles
    and two risk measures for the client:
    - prob_year_loss: share of simulated years (12-month periods) losing more than `year_loss_limit`
      (the client's stated maximum one-year loss in the questionnaire);
    - dd_median / dd_p5: the largest fall from a peak during the horizon (maximum drawdown) in a
      typical scenario (median) and in a bad one (5% worst scenarios), as negative fractions;
    - year_var95 / year_es95: one-year value at risk and expected shortfall (CVaR) at 95%, as
      returns: with the simulated one-year returns sorted and k = 5% of them, the k-th worst return
      and the average of the k worst. Expected shortfall is a coherent risk measure (subadditive: it
      never penalises diversification); value at risk is not.
    prob_loss (ending below the initial wealth) is kept for reference.
    Every risk measure is computed on the growth of 1 dollar, so it does not depend on the wealth
    (many simulated years are the same historical block, and rescaling by the wealth would move
    some of these ties across the 5% cut-off through rounding)."""
    sims = block_bootstrap(history, years * MONTHS_PER_YEAR, n_sims, block, seed)
    out = {}
    for name, r in sims.items():
        growth = np.hstack([np.ones((n_sims, 1)), np.cumprod(1 + r, axis=1)])
        paths = initial_wealth * growth
        bands = pd.DataFrame(np.percentile(paths, [5, 50, 95], axis=0).T, columns=["p5", "p50", "p95"])
        final = paths[:, -1]
        yearly = growth[:, MONTHS_PER_YEAR::MONTHS_PER_YEAR] / growth[:, :-1:MONTHS_PER_YEAR] - 1
        max_dd = (growth / np.maximum.accumulate(growth, axis=1) - 1).min(axis=1)
        worst = np.sort(yearly, axis=None)[:max(1, int(np.ceil(TAIL * yearly.size)))]
        out[name] = {"bands": bands, "median": float(np.median(final)), "p5": float(np.percentile(final, 5)),
                     "p95": float(np.percentile(final, 95)), "prob_loss": float((growth[:, -1] < 1).mean()),
                     "prob_year_loss": float((yearly < -year_loss_limit).mean()),
                     "dd_median": float(np.median(max_dd)), "dd_p5": float(np.percentile(max_dd, 5)),
                     "year_var95": float(worst[-1]), "year_es95": float(worst.mean())}
    return out
