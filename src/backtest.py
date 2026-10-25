"""Rolling-window backtest engine.

At each rebalancing date t (a month-end), the expected returns and covariance are estimated on
the 60 months up to and including t (only assets with a complete window are eligible), the
portfolio is re-optimised with the client constraints, and trades are executed at the close of
t. Between two rebalancings the weights drift with the assets' returns. Transaction costs are
`tc` (default 10 bps) per unit traded, charged on the month after the trade; the first
rebalancing pays the cost of building the portfolio from cash. The optimiser also sees the cost
(L1 penalty), so it does not trade for negligible gains.

If a rebalancing is infeasible (e.g. after the client tightened a constraint), the portfolio
keeps its drifted weights and the event is recorded; an infeasible FIRST rebalancing is an error.

Start: the default constraints need five eligible bond funds (70% bonds at most 15% per fund for
the Conservative profile, with a short enough average duration); IEI, IGSB and IGIB reach 60
months in 2012-01, so every profile starts at the 2012-01-31 rebalancing, with returns from
2012-02. Rebalancing dates are every 3, 6 or 12 months from the start (semi-annual and annual
dates are subsets of the quarterly ones).

Ex-ante versus realised risk: each rebalancing records the volatility promised by the
optimiser; the realised volatility is the annualised volatility of the strategy over the next 12
months (NaN when fewer than 12 months remain).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src import estimators as est, optimizer as opt

FREQ_MONTHS = {"Quarterly": 3, "Semi-annual": 6, "Annual": 12}
DEFAULT_START = pd.Timestamp("2012-01-31")
DEFAULT_TC = 0.001
REALISED_HORIZON = 12


class BacktestError(RuntimeError):
    """The backtest cannot start (e.g. constraints infeasible at the first rebalancing)."""


@dataclass
class BacktestResult:
    returns: pd.Series              # monthly net strategy returns
    weights: pd.DataFrame           # month-end weights after drift (first row = first target)
    rebalances: pd.DataFrame        # per rebalancing: ex-ante return/vol, turnover, cost, status
    events: list[str] = field(default_factory=list)


def rebalance_dates(index: pd.DatetimeIndex, start: pd.Timestamp, freq: str) -> list[pd.Timestamp]:
    if freq not in FREQ_MONTHS:
        raise BacktestError(f"Unknown rebalancing frequency '{freq}'.")
    step = FREQ_MONTHS[freq]
    dates = list(index[index >= start][::step])
    if len(dates) < 2:
        raise BacktestError("Not enough history after the start date for a backtest.")
    return dates


def estimates_by_date(returns: pd.DataFrame, rf: pd.Series, groups: pd.Series,
                      dates: list[pd.Timestamp], window: int = est.DEFAULT_WINDOW,
                      mean_shrinkage: float = est.DEFAULT_MEAN_SHRINKAGE) -> dict:
    """Estimates at each date (independent of the client, so they can be shared across profiles)."""
    return {d: est.estimate(est.estimation_window(returns, d, window), rf, groups, mean_shrinkage)
            for d in dates}


def run_backtest(returns: pd.DataFrame, rf: pd.Series, universe: pd.DataFrame,
                 constraints: list[opt.Constraint], freq: str = "Quarterly", tc: float = DEFAULT_TC,
                 start: pd.Timestamp = DEFAULT_START, window: int = est.DEFAULT_WINDOW,
                 estimates: dict | None = None) -> BacktestResult:
    rebals = rebalance_dates(returns.index, start, freq)
    estimates = estimates or {}
    tickers = list(returns.columns)
    months = returns.index[returns.index >= rebals[0]]
    w = pd.Series(0.0, index=tickers)
    w_rows, ret_rows, reb_rows, events = {}, {}, [], []
    pending_cost, invested = 0.0, False

    for i, t in enumerate(months):
        if t in rebals:
            e = estimates.get(t) or est.estimate(est.estimation_window(returns, t, window), rf,
                                                 est.shrinkage_groups(universe))
            d = opt.ProblemData(e.mu, e.cov, universe, w_prev=w if invested else None, tc=tc)
            res = opt.solve(d, constraints, diagnose=not invested)
            if res.ok:
                target = res.weights.reindex(tickers).fillna(0.0)
                turnover = float((target - w).abs().sum())
                pending_cost = tc * turnover
                reb_rows.append({"date": t, "status": "ok", "expected_return": res.expected_return,
                                 "ex_ante_vol": res.volatility, "turnover": turnover, "cost": pending_cost,
                                 "initial": not invested, "n_positions": int((target > 0.005).sum()),
                                 "n_eligible": len(e.mu)})
                w, invested = target, True
            elif not invested:
                n_bonds = int((universe["asset_class"].reindex(e.mu.index) == "Bond").sum())
                raise BacktestError(
                    f"Constraints are infeasible at the first rebalancing ({t:%B %Y}). At that date only "
                    f"{len(e.mu)} assets, including {n_bonds} bond funds, had {window} months of history, "
                    f"so settings that work today may not work then. {res.message}")
            else:
                events.append(f"{t:%Y-%m}: rebalancing skipped, weights kept ({res.message})")
                reb_rows.append({"date": t, "status": "skipped", "turnover": 0.0, "cost": 0.0, "initial": False})
        w_rows[t] = w.copy()
        if i + 1 < len(months):
            nxt = months[i + 1]
            r = returns.loc[nxt].reindex(tickers).fillna(0.0)
            gross = float(w @ r)
            ret_rows[nxt] = (1 + gross) * (1 - pending_cost) - 1
            pending_cost = 0.0
            w = w * (1 + r) / (1 + gross)

    strat = pd.Series(ret_rows, name="Strategy").sort_index()
    reb = pd.DataFrame(reb_rows).set_index("date")
    reb["realised_vol"] = [
        strat.loc[strat.index > t].head(REALISED_HORIZON).std() * np.sqrt(12)
        if (strat.index > t).sum() >= REALISED_HORIZON else np.nan for t in reb.index]
    return BacktestResult(strat, pd.DataFrame(w_rows).T, reb, events)


def annual_turnover(result: BacktestResult) -> float:
    """Average yearly turnover (buys + sells), excluding the initial investment."""
    reb = result.rebalances
    years = len(result.returns) / 12
    return float(reb.loc[~reb["initial"], "turnover"].sum() / years) if years > 0 else np.nan
