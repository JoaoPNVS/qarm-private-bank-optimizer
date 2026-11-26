"""Rolling-window backtest engine.

Timing (no look-ahead, one month of implementation lag). At each rebalancing date t (a
month-end), the decision uses information up to the PREVIOUS month-end t-1 only: expected returns
and covariance are estimated on the 60 months up to and including t-1 (only assets with a complete
window are eligible), and the insider view uses the filings published by t-1. The portfolio is
re-optimised with the client constraints and the trades are executed at the close of t, so the
new weights earn the returns of month t+1. The month between the data and the trade leaves time
to compute the portfolio and is conservative: deciding with data up to t and trading at that
same close would assume the month-end prices are known before the close (tested: it adds about
+0.01 to +0.03 to the Sharpe ratio of every profile and makes drawdowns slightly smaller). Between two rebalancings the weights drift with the assets' returns. Transaction costs are
asset-specific (universe.csv tc_bps, by liquidity tier): sum_i c_i |w_i - w_prev_i|, charged on
the month after the trade; the first rebalancing pays the cost of building the portfolio from
cash. With `penalize_costs` (default) the optimiser also sees these costs (L1 penalty), so it
does not trade for negligible gains; without it costs are only charged afterwards.

If a rebalancing is infeasible (e.g. after the client tightened a constraint), the portfolio
keeps its drifted weights and the event is recorded; an infeasible FIRST rebalancing is an error.

Start: the default constraints need five eligible bond funds (70% bonds at most 15% per fund for
the Conservative profile, with a short enough average duration); IEI, IGSB and IGIB reach 60
months in 2012-01, so with the one-month lag every profile starts at the 2012-04-30 rebalancing
(the first quarter-end month whose decision data, to 2012-03, has them), with returns from
2012-05. Rebalancing dates are every 3, 6 or 12 months from the start (semi-annual and annual
dates are subsets of the quarterly ones).

Equal-weight benchmark (`run_equal_weight`): same universe, same eligibility and timing (hence the
same survivorship bias), same dates and costs, equal weights on stocks and bond funds split by the
profile's bond floor. It is a naive reference, not the strategy without optimisation: it ignores
the expected returns, the volatility budget and the bond-sleeve, sector and cash limits (its bonds
are longer and riskier than several profiles allow), so the gap to the strategy mixes the effect of
the optimisation with these differences in allocation and constraints.

Ex-ante versus realised risk: each rebalancing records the volatility estimated by the
optimiser; this is not a guarantee. The realised volatility is the annualised volatility of the
strategy over the next 12 months (NaN when fewer than 12 months remain).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src import estimators as est, insiders as ins, optimizer as opt

FREQ_MONTHS = {"Quarterly": 3, "Semi-annual": 6, "Annual": 12}
DEFAULT_START = pd.Timestamp("2012-04-30")
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


def information_date(index: pd.DatetimeIndex, t: pd.Timestamp) -> pd.Timestamp:
    """The last month-end whose data a decision traded at the close of `t` may use: the month before t."""
    past = index[index < t]
    if len(past) == 0:
        raise BacktestError(f"No data before {t:%Y-%m} to decide the rebalancing.")
    return past[-1]


def estimates_by_date(returns: pd.DataFrame, rf: pd.Series, groups: pd.Series,
                      dates: list[pd.Timestamp], window: int = est.DEFAULT_WINDOW,
                      mean_shrinkage: float = est.DEFAULT_MEAN_SHRINKAGE,
                      benchmarks: pd.DataFrame | None = None) -> dict:
    """Estimates for each rebalancing date d, on the window ending at the previous month-end
    (`information_date`), so e.end < d. Independent of the client, so they can be shared across
    profiles. With `benchmarks` (SPY, AGG) they also carry the Black-Litterman prior; apply a
    confidence with `with_view_confidence`."""
    return {d: est.estimate(est.estimation_window(returns, information_date(returns.index, d), window),
                            rf, groups, mean_shrinkage,
                            benchmarks=benchmarks)
            for d in dates}


def with_view_confidence(estimates: dict, confidence: float) -> dict:
    """Black-Litterman expected returns at every date (cheap: covariances are reused)."""
    return {d: est.with_view_confidence(e, confidence) for d, e in estimates.items()}


def with_insider_view(estimates: dict, signal: pd.DataFrame, confidence: float) -> dict:
    """Black-Litterman expected returns from the insider view at every rebalancing date, each using
    only the insider filings published by the end of its estimation window (e.end, the month before
    the trade; src/insiders.py)."""
    return {d: ins.with_insider_view(e, e.end, signal, confidence)[0] for d, e in estimates.items()}


def _simulate(returns: pd.DataFrame, universe: pd.DataFrame, rebals: list[pd.Timestamp], tc: pd.Series,
              target_at, name: str) -> BacktestResult:
    """Shared engine: at each rebalancing `target_at(t, w_drifted, invested)` returns
    (target weights or None, info dict, message); weights drift between rebalancings and each
    trade pays sum_i c_i |target_i - drifted_i| on the following month."""
    tickers = list(returns.columns)
    months = returns.index[returns.index >= rebals[0]]
    w = pd.Series(0.0, index=tickers)
    w_rows, ret_rows, reb_rows, events = {}, {}, [], []
    pending_cost, invested = 0.0, False
    for i, t in enumerate(months):
        if t in rebals:
            target, info, message = target_at(t, w, invested)
            if target is not None:
                target = target.reindex(tickers).fillna(0.0)
                pending_cost = float(tc @ (target - w).abs())
                reb_rows.append({"date": t, "status": "ok", "turnover": float((target - w).abs().sum()),
                                 "cost": pending_cost, "initial": not invested,
                                 "n_positions": int((target > 0.005).sum()), **info})
                w, invested = target, True
            else:
                events.append(f"{t:%Y-%m}: rebalancing skipped, weights kept ({message})")
                reb_rows.append({"date": t, "status": "skipped", "turnover": 0.0, "cost": 0.0, "initial": False})
        w_rows[t] = w.copy()
        if i + 1 < len(months):
            nxt = months[i + 1]
            r = returns.loc[nxt].reindex(tickers).fillna(0.0)
            gross = float(w @ r)
            ret_rows[nxt] = (1 + gross) * (1 - pending_cost) - 1
            pending_cost = 0.0
            w = w * (1 + r) / (1 + gross)
    out = pd.Series(ret_rows, name=name).sort_index()
    reb = pd.DataFrame(reb_rows).set_index("date")
    reb["realised_vol"] = [
        out.loc[out.index > t].head(REALISED_HORIZON).std() * np.sqrt(12)
        if (out.index > t).sum() >= REALISED_HORIZON else np.nan for t in reb.index]
    return BacktestResult(out, pd.DataFrame(w_rows).T, reb, events)


def _costs(universe: pd.DataFrame, returns: pd.DataFrame, tc: pd.Series | None) -> pd.Series:
    return (universe["tc_bps"] / 1e4 if tc is None else tc).reindex(returns.columns).fillna(0.0)


def run_backtest(returns: pd.DataFrame, rf: pd.Series, universe: pd.DataFrame,
                 constraints: list[opt.Constraint], freq: str = "Quarterly", tc: pd.Series | None = None,
                 start: pd.Timestamp = DEFAULT_START, window: int = est.DEFAULT_WINDOW,
                 estimates: dict | None = None, penalize_costs: bool = True) -> BacktestResult:
    """The strategy: re-optimised at each rebalancing. tc: cost per unit traded for each asset
    (default: universe tc_bps / 1e4)."""
    rebals = rebalance_dates(returns.index, start, freq)
    estimates = estimates or {}
    tc = _costs(universe, returns, tc)

    def target_at(t, w, invested):
        e = estimates.get(t) or est.estimate(
            est.estimation_window(returns, information_date(returns.index, t), window), rf,
            est.shrinkage_groups(universe))
        d = opt.ProblemData(e.mu, e.cov, universe, w_prev=w if invested else None,
                            tc=tc if penalize_costs else 0.0)
        res = opt.solve(d, constraints, diagnose=not invested)
        if res.ok:
            return res.weights, {"expected_return": res.expected_return, "ex_ante_vol": res.volatility,
                                 "n_eligible": len(e.mu)}, ""
        if not invested:
            n_bonds = int((universe["asset_class"].reindex(e.mu.index) == "Bond").sum())
            raise BacktestError(
                f"Constraints are infeasible at the first rebalancing ({t:%B %Y}). With data to "
                f"{e.end:%B %Y} only {len(e.mu)} assets, including {n_bonds} bond funds, had {window} "
                f"months of history, "
                f"so settings that work today may not work then. {res.message}")
        return None, {}, res.message

    return _simulate(returns, universe, rebals, tc, target_at, "Strategy")


def run_equal_weight(returns: pd.DataFrame, universe: pd.DataFrame, bond_share: float,
                     freq: str = "Quarterly", tc: pd.Series | None = None, start: pd.Timestamp = DEFAULT_START,
                     window: int = est.DEFAULT_WINDOW) -> BacktestResult:
    """Naive benchmark on the SAME universe: at each rebalancing, equal weights on the eligible
    stocks (same 60-month rule, so the same survivors) for 1 - bond_share, and equal weights on the
    eligible bond funds (cash equivalents excluded) for bond_share; same dates and same per-asset
    costs as the strategy. A naive reference: it also ignores the volatility budget and the
    bond-sleeve, sector and cash limits, so the gap to the strategy is not the effect of the
    optimisation alone."""
    rebals = rebalance_dates(returns.index, start, freq)
    tc = _costs(universe, returns, tc)

    def target_at(t, w, invested):
        eligible = est.estimation_window(returns, information_date(returns.index, t), window).columns
        u = universe.loc[eligible]
        stocks = u.index[u["vehicle"] == "Stock"]
        bonds = u.index[u["asset_class"] == "Bond"]
        target = pd.Series(0.0, index=returns.columns)
        target[stocks] = (1 - bond_share) / len(stocks)
        if bond_share > 0:
            target[bonds] = bond_share / len(bonds)
        return target, {"n_eligible": len(eligible)}, ""

    return _simulate(returns, universe, rebals, tc, target_at, "Equal weight")


def annual_cost(result: BacktestResult) -> float:
    """Average trading costs paid per year (fraction of the portfolio), initial investment included."""
    years = len(result.returns) / 12
    return float(result.rebalances["cost"].sum() / years) if years > 0 else np.nan


def annual_turnover(result: BacktestResult) -> float:
    """Average yearly turnover (buys + sells), excluding the initial investment."""
    reb = result.rebalances
    years = len(result.returns) / 12
    return float(reb.loc[~reb["initial"], "turnover"].sum() / years) if years > 0 else np.nan
