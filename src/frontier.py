"""Efficient frontier under the client's constraints.

Each point is the portfolio with the highest expected return for a given volatility, subject to
long-only, fully invested and every active client constraint except the volatility cap (which is
what moves along the curve). The curve runs from the minimum-variance portfolio to the
highest-return portfolio the constraints allow. The client's recommended portfolio is the point
at the profile's volatility cap.

Comparing it with the frontier WITHOUT client constraints (long-only and fully invested only)
shows what the client's rules cost: at the same risk, the gap between the two curves is the
expected return given up.

Two special points of the constrained curve: the minimum-variance portfolio (its left end) and
the maximum-Sharpe (tangency) portfolio. Returns are in excess of the T-bill, so the risk-free
asset sits at the origin and the capital market line runs from the origin through the tangency
portfolio, with slope equal to its Sharpe ratio.
"""
from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd

from src import optimizer as opt

CLASS_ORDER = ["Equity", "Bond", "Commodity", "Cash"]
GOLDEN = (np.sqrt(5) - 1) / 2
MIN_VOL = 1e-3  # below this the Sharpe ratio is not meaningful (cash)


class FrontierError(ValueError):
    """The constraints admit no portfolio, so there is no frontier."""


def _solve_extreme(data: opt.ProblemData, constraints: list[opt.Constraint], minimise_risk: bool) -> pd.Series:
    w = cp.Variable(len(data.tickers))
    parts = [p.constraint for _, p in opt._build_all(w, data, constraints)]
    objective = cp.Minimize(cp.sum_squares(data.sqrt_cov @ w)) if minimise_risk else cp.Maximize(data.mu.values @ w)
    status = opt._run(cp.Problem(objective, opt._base(w) + parts))
    if status not in opt.OK_STATUSES:
        raise FrontierError("The constraints cannot all be satisfied, so there is no efficient frontier.")
    weights = pd.Series(np.clip(w.value, 0, None), index=data.tickers)
    return weights / weights.sum()


def _point(data: opt.ProblemData, weights: pd.Series) -> dict:
    stats = opt.portfolio_stats(weights, data)
    by_class = weights.groupby(data.universe["asset_class"]).sum().reindex(CLASS_ORDER).fillna(0.0)
    return {"volatility": stats["volatility"], "expected_return": stats["expected_return"], **by_class.to_dict()}


def efficient_frontier(data: opt.ProblemData, constraints: list[opt.Constraint], n_points: int = 20,
                       max_vol: float | None = None) -> pd.DataFrame:
    """Frontier points (volatility, expected return, weight per asset class), from the
    minimum-variance portfolio to the maximum-return portfolio under `constraints` (the volatility
    cap, if present, is ignored), stopping at `max_vol` if given (display range). Trading costs are
    not part of the objective here."""
    others = [c for c in constraints if c.active and c.key != opt.VOL_KEY]
    data = opt.ProblemData(data.mu, data.cov, data.universe)  # no current holdings, no trading costs
    low = _point(data, _solve_extreme(data, others, minimise_risk=True))
    high = _point(data, _solve_extreme(data, others, minimise_risk=False))
    top = high["volatility"] if max_vol is None else min(high["volatility"], max_vol)
    points = [low]
    for v in np.linspace(low["volatility"], top, n_points)[1:]:
        res = opt.solve(data, others + [opt.max_volatility(float(v))], diagnose=False)
        if res.ok:
            points.append(_point(data, res.weights))
    if top == high["volatility"]:
        points.append(high)
    out = pd.DataFrame(points).sort_values("volatility")
    # the max-return portfolio can sit to the right of a flat top: keep only the rising part
    return out[out["expected_return"] >= out["expected_return"].cummax() - 1e-9].reset_index(drop=True)


def _sharpe(point: dict | None) -> float:
    if point is None or point["volatility"] < MIN_VOL:
        return -np.inf
    return point["expected_return"] / point["volatility"]


def max_sharpe_portfolio(data: opt.ProblemData, constraints: list[opt.Constraint], frontier: pd.DataFrame,
                         tol: float = 1e-4) -> dict:
    """Tangency portfolio: the highest Sharpe ratio (expected excess return / volatility) the
    client's constraints allow. Along the concave frontier f(sigma) the ratio f(sigma) / sigma has a
    single peak, so a golden-section search on the volatility cap, bracketed by the neighbours of
    the best point of `frontier` (computed with the same constraints), finds it."""
    others = [c for c in constraints if c.active and c.key != opt.VOL_KEY]
    data = opt.ProblemData(data.mu, data.cov, data.universe)
    pts = frontier[frontier["volatility"] >= MIN_VOL].reset_index(drop=True)
    if pts.empty:
        raise FrontierError("The frontier has no risky portfolio, so there is no tangency portfolio.")
    i = int((pts["expected_return"] / pts["volatility"]).to_numpy().argmax())
    best = pts.iloc[i].to_dict()

    def at(vol: float) -> dict | None:
        res = opt.solve(data, others + [opt.max_volatility(float(vol))], diagnose=False)
        return _point(data, res.weights) if res.ok else None

    a, b = pts["volatility"].iloc[max(i - 1, 0)], pts["volatility"].iloc[min(i + 1, len(pts) - 1)]
    c, d = b - GOLDEN * (b - a), a + GOLDEN * (b - a)
    fc, fd = at(c), at(d)
    while b - a > tol:
        if _sharpe(fc) >= _sharpe(fd):
            b, d, fd = d, c, fc
            c = b - GOLDEN * (b - a)
            fc = at(c)
        else:
            a, c, fc = c, d, fd
            d = a + GOLDEN * (b - a)
            fd = at(d)
    best = max((p for p in (best, fc, fd) if p is not None), key=_sharpe)
    return {**best, "sharpe": _sharpe(best)}
