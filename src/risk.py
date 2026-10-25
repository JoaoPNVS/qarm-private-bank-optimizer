"""Performance and risk metrics (Monte Carlo wealth simulation to come)."""
from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12


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
