"""cvxpy portfolio problem builder, constraint objects and infeasibility diagnosis.

Problem (long-only, fully invested, annualised excess returns):
    max  w'mu - c * ||w - w_prev||_1
    s.t. sqrt(w'Sigma w) <= sigma_max (volatility cap of the client's profile, mandatory)
         and the other active client constraints.
The transaction-cost term only applies when current holdings w_prev are given (backtest).
Shadow prices are reported per +1 percentage point of relaxation of the constraint bound, in
basis points of expected return per year.

Nothing here raises on an infeasible or failed solve: results carry a status and, when
infeasible, a diagnosis naming the conflicting constraints.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, NamedTuple

import cvxpy as cp
import numpy as np
import pandas as pd

SOLVERS = ["CLARABEL", "SCS"]
OK_STATUSES = {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
INFEASIBLE_STATUSES = {cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE}
ACTIVE_DUAL_TOL = 1e-6
TOL = 1e-6
VOL_KEY = "max_volatility"


@dataclass
class ProblemData:
    mu: pd.Series                     # annualised expected excess returns
    cov: pd.DataFrame                 # annualised covariance
    universe: pd.DataFrame            # tags indexed by ticker (src.data.load_universe)
    w_prev: pd.Series | None = None   # current holdings; enables turnover cost/constraint
    tc: float = 0.0                   # proportional transaction cost per unit traded

    def __post_init__(self):
        tickers = self.mu.index
        self.universe = self.universe.loc[tickers]
        self.cov = self.cov.loc[tickers, tickers]
        if self.w_prev is not None:
            self.w_prev = self.w_prev.reindex(tickers).fillna(0.0)
        vals, vecs = np.linalg.eigh(self.cov.values)
        self.sqrt_cov = vecs @ np.diag(np.sqrt(np.clip(vals, 0, None))) @ vecs.T

    @property
    def tickers(self) -> list[str]:
        return list(self.mu.index)

    def mask(self, condition: pd.Series) -> np.ndarray:
        return condition.loc[self.mu.index].astype(float).values


class Part(NamedTuple):
    label: str
    constraint: cp.Constraint
    elements: list[str] | None = None  # per-element labels for vector constraints


Builder = Callable[[cp.Variable, ProblemData, dict], list[Part]]


@dataclass
class Constraint:
    key: str
    name: str
    params: dict
    builder: Builder
    describe: Callable[[dict], str]
    active: bool = True

    @property
    def description(self) -> str:
        return self.describe(self.params)

    def build(self, w: cp.Variable, data: ProblemData) -> list[Part]:
        return self.builder(w, data, self.params)

    def with_params(self, **params) -> "Constraint":
        return replace(self, params={**self.params, **params})


# --- constraint factories -------------------------------------------------------------------
# Kept deliberately simple so each one can be explained to the client in one sentence.

def max_volatility(limit: float = 0.10) -> Constraint:
    """The risk budget of the profile. Mandatory: without it the problem would put everything
    in the single asset with the highest expected return."""
    return Constraint(
        VOL_KEY, "Maximum volatility", {"limit": limit},
        lambda w, d, p: [Part(f"Volatility <= {p['limit']:.0%}", cp.norm(d.sqrt_cov @ w, 2) <= p["limit"])],
        lambda p: f"Expected annual volatility at most {p['limit']:.0%}.")


def _cap_per_vehicle(vehicle: str, label: str):
    def build(w, d, p):
        idx = [i for i, t in enumerate(d.tickers) if d.universe.at[t, "vehicle"] == vehicle]
        return [Part(label, w[idx] <= p["limit"], [d.tickers[i] for i in idx])] if idx else []
    return build


def max_stock_weight(limit: float = 0.08) -> Constraint:
    """Cap on each single stock."""
    return Constraint(
        "max_stock_weight", "Maximum weight per stock", {"limit": limit},
        _cap_per_vehicle("Stock", "Max stock weight"),
        lambda p: f"No single company above {p['limit']:.0%} of the portfolio.")


def _cap_per_fund(bonds: bool, label: str):
    def build(w, d, p):
        idx = [i for i, t in enumerate(d.tickers)
               if d.universe.at[t, "vehicle"] == "ETF" and (d.universe.at[t, "asset_class"] == "Bond") == bonds]
        return [Part(label, w[idx] <= p["limit"], [d.tickers[i] for i in idx])] if idx else []
    return build


def max_etf_weight(limit: float = 0.20) -> Constraint:
    """Cap on each non-bond ETF (international equity funds, gold)."""
    return Constraint(
        "max_etf_weight", "Maximum weight per equity / gold ETF", {"limit": limit},
        _cap_per_fund(False, "Max ETF weight"),
        lambda p: f"No single equity or gold fund above {p['limit']:.0%} of the portfolio.")


def max_bond_etf_weight(limit: float = 0.15) -> Constraint:
    """Cap on each bond ETF, so the bond sleeve is spread over several segments."""
    return Constraint(
        "max_bond_etf_weight", "Maximum weight per bond ETF", {"limit": limit},
        _cap_per_fund(True, "Max bond ETF weight"),
        lambda p: f"No single bond fund above {p['limit']:.0%} of the portfolio.")


def _bond_mask(d, condition=None):
    bond = d.universe["asset_class"] == "Bond"
    return d.mask(bond if condition is None else bond & condition)


def max_bond_duration(limit: float = 6.0) -> Constraint:
    """Weighted average effective duration of the bond sleeve, linear form
    sum(w_b * D_b) <= limit * sum(w_b) (holds trivially without bonds)."""
    def build(w, d, p):
        dur = d.universe["duration"].fillna(0.0).loc[d.tickers].values
        return [Part(f"Bond duration <= {p['limit']:.1f}y", (_bond_mask(d) * dur) @ w <= p["limit"] * (_bond_mask(d) @ w))]

    return Constraint(
        "max_bond_duration", "Maximum bond duration", {"limit": limit}, build,
        lambda p: f"Average duration of the bond portfolio at most {p['limit']:.1f} years.")


def min_investment_grade(min_share: float = 0.80) -> Constraint:
    """Investment-grade funds at least `min_share` of the bond sleeve (linear form)."""
    def build(w, d, p):
        ig = _bond_mask(d, d.universe["credit_quality"] == "IG")
        return [Part(f"IG >= {p['min_share']:.0%} of bonds", ig @ w >= p["min_share"] * (_bond_mask(d) @ w))]

    return Constraint(
        "min_investment_grade", "Minimum investment grade", {"min_share": min_share}, build,
        lambda p: f"At least {p['min_share']:.0%} of the bond portfolio rated investment grade.")


def max_high_yield(limit: float = 0.10) -> Constraint:
    """High-yield funds (HYG, and EMB counted as high yield) at most `limit` of the whole portfolio."""
    def build(w, d, p):
        return [Part(f"High yield <= {p['limit']:.0%}", _bond_mask(d, d.universe["credit_quality"] == "HY") @ w <= p["limit"])]

    return Constraint(
        "max_high_yield", "Maximum high yield", {"limit": limit}, build,
        lambda p: f"High-yield and emerging-market bonds at most {p['limit']:.0%} of the portfolio.")


def max_sector_weight(limit: float = 0.40) -> Constraint:
    """Cap per equity sector; the international equity ETFs form one group ("International").
    Bond ETFs, gold and cash are not equities and are not subject to this cap."""
    def build(w, d, p):
        eq = d.universe["asset_class"] == "Equity"
        return [Part(f"{s} <= {p['limit']:.0%}", d.mask(eq & (d.universe["sector"] == s)) @ w <= p["limit"])
                for s in sorted(d.universe.loc[eq, "sector"].unique())]

    return Constraint(
        "max_sector_weight", "Maximum weight per sector", {"limit": limit}, build,
        lambda p: f"No equity sector (international funds count as one group) above {p['limit']:.0%} "
                  "of the portfolio.")


def em_equity_cap(ratio: float = 1 / 3) -> Constraint:
    """Emerging markets (EEM, PDD, MELI) at most `ratio` of the equity allocation, linear form
    EM <= ratio * equities (no division, so it stays convex and holds with zero equities)."""
    def build(w, d, p):
        eq = d.universe["asset_class"] == "Equity"
        return [Part(f"EM <= {p['ratio']:.0%} of equities",
                     d.mask(eq & d.universe["is_EM"]) @ w <= p["ratio"] * (d.mask(eq) @ w))]

    return Constraint(
        "em_equity_cap", "Emerging-markets limit", {"ratio": ratio}, build,
        lambda p: f"Emerging markets at most {p['ratio']:.0%} of the equity allocation.")


def exclusions(tickers: list[str] | None = None, sectors: list[str] | None = None) -> Constraint:
    def build(w, d, p):
        out = d.universe["sector"].isin(p["sectors"]) | d.universe.index.isin(p["tickers"])
        idx = [i for i, t in enumerate(d.tickers) if out[t]]
        return [Part("Excluded", w[idx] == 0, [d.tickers[i] for i in idx])] if idx else []

    def describe(p):
        items = [f"the {s} sector" for s in p["sectors"]] + p["tickers"]
        return "Never invest in: " + (", ".join(items) or "nothing excluded") + "."

    return Constraint("exclusions", "Exclusions (sectors or companies)",
                      {"tickers": list(tickers or []), "sectors": list(sectors or [])}, build, describe)


def min_bonds(min_share: float = 0.20) -> Constraint:
    """Minimum share in bond funds (the profile's floor; cash does not count)."""
    def build(w, d, p):
        if p["min_share"] <= 0:
            return []
        m = d.mask(d.universe["asset_class"] == "Bond")
        return [Part(f"Bonds >= {p['min_share']:.0%}", m @ w >= p["min_share"])]

    return Constraint(
        "min_bonds", "Minimum bonds", {"min_share": min_share}, build,
        lambda p: f"At least {p['min_share']:.0%} of the portfolio in bond funds.")


def max_cash(limit: float = 0.10) -> Constraint:
    def build(w, d, p):
        m = d.mask(d.universe["asset_class"] == "Cash")
        return [Part(f"Cash <= {p['limit']:.0%}", m @ w <= p["limit"])]

    return Constraint(
        "max_cash", "Maximum cash", {"limit": limit}, build,
        lambda p: f"At most {p['limit']:.0%} of the portfolio in cash.")


# --- solving --------------------------------------------------------------------------------

@dataclass
class InfeasibilityDiagnosis:
    conflicting: list[str]   # minimal set of constraint keys that cannot hold together
    single_fixes: list[str]  # constraints whose removal alone makes the problem solvable
    message: str


@dataclass
class OptimizationResult:
    status: str
    weights: pd.Series | None = None
    expected_return: float = np.nan
    volatility: float = np.nan
    sharpe: float = np.nan
    turnover: float = np.nan
    duals: pd.DataFrame = field(default_factory=pd.DataFrame)
    diagnosis: InfeasibilityDiagnosis | None = None
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.status in OK_STATUSES


def _run(problem: cp.Problem) -> str:
    """Solve with the first installed solver that returns a definitive status."""
    installed = set(cp.installed_solvers())
    status = "solver_error"
    for solver in (s for s in SOLVERS if s in installed):
        try:
            problem.solve(solver=solver)
        except (cp.SolverError, ArithmeticError, ValueError):
            continue
        status = problem.status
        if status in OK_STATUSES | INFEASIBLE_STATUSES:
            return status
    return status


def _base(w: cp.Variable) -> list[cp.Constraint]:
    return [cp.sum(w) == 1, w >= 0]


def _build_all(w, data, constraints) -> list[tuple[Constraint, Part]]:
    return [(c, part) for c in constraints if c.active for part in c.build(w, data)]


def _objective(w, data):
    obj = data.mu.values @ w
    if data.w_prev is not None and data.tc > 0:
        obj -= data.tc * cp.norm1(w - data.w_prev.values)
    return obj


def portfolio_stats(weights: pd.Series, data: ProblemData) -> dict:
    w = weights.reindex(data.tickers).fillna(0.0).values
    ret = float(data.mu.values @ w)
    vol = float(np.sqrt(max(w @ data.cov.values @ w, 0.0)))
    turnover = float(np.abs(w - data.w_prev.values).sum()) if data.w_prev is not None else np.nan
    return {"expected_return": ret, "volatility": vol,
            "sharpe": ret / vol if vol > TOL else np.nan, "turnover": turnover}


def _dual_table(pairs: list[tuple[Constraint, Part]]) -> pd.DataFrame:
    rows = []
    for c, part in pairs:
        dual = np.atleast_1d(np.asarray(part.constraint.dual_value, dtype=float))
        labels = part.elements or [None] * dual.size
        for value, element in zip(np.abs(dual), labels):
            rows.append({"key": c.key, "constraint": c.name,
                         "label": f"{part.label} [{element}]" if element else part.label,
                         "shadow_price": value, "bps_per_pp": value * 100,
                         "active": value > ACTIVE_DUAL_TOL})
    return pd.DataFrame(rows, columns=["key", "constraint", "label", "shadow_price", "bps_per_pp", "active"])


def is_feasible(data: ProblemData, constraints: list[Constraint]) -> bool:
    w = cp.Variable(len(data.tickers))
    parts = _build_all(w, data, constraints)
    return _run(cp.Problem(cp.Minimize(0), _base(w) + [p.constraint for _, p in parts])) in OK_STATUSES


def diagnose_infeasibility(data: ProblemData, constraints: list[Constraint]) -> InfeasibilityDiagnosis:
    """Deletion filter: shrink the active set to an irreducible conflicting subset, and list
    the constraints whose removal alone restores feasibility."""
    active = [c for c in constraints if c.active]
    names = {c.key: c.name for c in active}
    single = [c.key for c in active if is_feasible(data, [o for o in active if o is not c])]
    conflict = list(active)
    for c in list(conflict):
        trial = [o for o in conflict if o is not c]
        if not is_feasible(data, trial):
            conflict = trial
    keys = [c.key for c in conflict]
    msg = "These constraints cannot all be satisfied together: " + ", ".join(names[k] for k in keys) + "."
    if len(single) == 1:
        msg += f" Relaxing '{names[single[0]]}' alone would make the portfolio feasible."
    elif single:
        msg += " Relaxing any one of these would make the portfolio feasible: " + \
               ", ".join(names[k] for k in single) + "."
    else:
        msg += " Several constraints must be relaxed at the same time."
    return InfeasibilityDiagnosis(keys, single, msg)


def solve(data: ProblemData, constraints: list[Constraint], diagnose: bool = True) -> OptimizationResult:
    if not any(c.key == VOL_KEY and c.active for c in constraints):
        return OptimizationResult("invalid", message="A maximum volatility (the profile's risk budget) is required.")
    w = cp.Variable(len(data.tickers))
    pairs = _build_all(w, data, constraints)
    problem = cp.Problem(cp.Maximize(_objective(w, data)), _base(w) + [p.constraint for _, p in pairs])
    status = _run(problem)
    if status in OK_STATUSES:
        weights = pd.Series(np.clip(w.value, 0, None), index=data.tickers, name="weight")
        weights /= weights.sum()
        return OptimizationResult(status, weights, **portfolio_stats(weights, data),
                                  duals=_dual_table(pairs), message="Optimal portfolio found.")
    if status in INFEASIBLE_STATUSES:
        diag = diagnose_infeasibility(data, constraints) if diagnose else None
        return OptimizationResult(cp.INFEASIBLE, diagnosis=diag,
                                  message=diag.message if diag else "The constraints are incompatible.")
    return OptimizationResult(status, message=f"The solver failed (status: {status}). Try relaxing constraints.")
