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
    benchmark: pd.Series | None = None  # profile benchmark; enables the tracking-error constraint

    def __post_init__(self):
        tickers = self.mu.index
        self.universe = self.universe.loc[tickers]
        self.cov = self.cov.loc[tickers, tickers]
        if self.w_prev is not None:
            self.w_prev = self.w_prev.reindex(tickers).fillna(0.0)
        if self.benchmark is not None:
            self.benchmark = self.benchmark.reindex(tickers).fillna(0.0)
        vals, vecs = np.linalg.eigh(self.cov.values)
        self.sqrt_cov = vecs @ np.diag(np.sqrt(np.clip(vals, 0, None))) @ vecs.T

    @property
    def tickers(self) -> list[str]:
        return list(self.mu.index)

    def mask(self, condition: pd.Series) -> np.ndarray:
        return condition.loc[self.mu.index].astype(float).values

    @property
    def equity(self) -> np.ndarray:
        return self.mask(self.universe["asset_class"] == "Equity")


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

def max_weight(limit: float = 0.15) -> Constraint:
    return Constraint(
        "max_weight", "Maximum weight per asset", {"limit": limit},
        lambda w, d, p: [Part("Max weight", w <= p["limit"], d.tickers)],
        lambda p: f"No single investment above {p['limit']:.0%} of the portfolio.")


def asset_class_bands(bands: dict[str, tuple[float, float]]) -> Constraint:
    def build(w, d, p):
        parts = []
        for cls, (lo, hi) in p["bands"].items():
            m = d.mask(d.universe["asset_class"] == cls)
            if lo > 0:
                parts.append(Part(f"{cls} >= {lo:.0%}", m @ w >= lo))
            if hi < 1:
                parts.append(Part(f"{cls} <= {hi:.0%}", m @ w <= hi))
        return parts

    return Constraint(
        "asset_class_bands", "Asset-class bands", {"bands": bands}, build,
        lambda p: "; ".join(f"{c} between {lo:.0%} and {hi:.0%}" for c, (lo, hi) in p["bands"].items()) + ".")


def region_bands(bands: dict[str, tuple[float, float]]) -> Constraint:
    """Min/max share of the EQUITY allocation per region (linear form: lo*E <= R <= hi*E)."""
    def build(w, d, p):
        parts, eq = [], d.equity @ w
        for region, (lo, hi) in p["bands"].items():
            m = d.mask((d.universe["asset_class"] == "Equity") & (d.universe["region"] == region))
            if lo > 0:
                parts.append(Part(f"{region} >= {lo:.0%} of equities", m @ w >= lo * eq))
            if hi < 1:
                parts.append(Part(f"{region} <= {hi:.0%} of equities", m @ w <= hi * eq))
        return parts

    return Constraint(
        "region_bands", "Regional equity bands", {"bands": bands}, build,
        lambda p: "Share of the equity allocation: " + "; ".join(
            f"{r} {lo:.0%}-{hi:.0%}" for r, (lo, hi) in p["bands"].items()) + ".")


def em_equity_cap(ratio: float = 1 / 3) -> Constraint:
    def build(w, d, p):
        em = d.mask((d.universe["asset_class"] == "Equity") & d.universe["is_EM"])
        return [Part(f"EM <= {p['ratio']:.0%} of equities", em @ w <= p["ratio"] * (d.equity @ w))]

    return Constraint(
        "em_equity_cap", "Emerging-markets cap", {"ratio": ratio}, build,
        lambda p: f"Emerging-market shares at most {p['ratio']:.0%} of the equity allocation.")


def home_bias(min_share: float = 0.30) -> Constraint:
    def build(w, d, p):
        ch = d.mask((d.universe["asset_class"] == "Equity") & d.universe["is_swiss"])
        return [Part(f"Swiss >= {p['min_share']:.0%} of equities", ch @ w >= p["min_share"] * (d.equity @ w))]

    return Constraint(
        "home_bias", "Swiss home bias", {"min_share": min_share}, build,
        lambda p: f"At least {p['min_share']:.0%} of the equity allocation in Swiss companies.")


def max_non_chf(limit: float = 0.40) -> Constraint:
    return Constraint(
        "max_non_chf", "Foreign-currency limit", {"limit": limit},
        lambda w, d, p: [Part(f"Non-CHF <= {p['limit']:.0%}", d.mask(d.universe["non_chf"]) @ w <= p["limit"])],
        lambda p: f"At most {p['limit']:.0%} of the portfolio exposed to currencies other than CHF.")


def exclusions(tickers: list[str]) -> Constraint:
    def build(w, d, p):
        idx = [d.tickers.index(t) for t in p["tickers"] if t in d.tickers]
        return [Part("Excluded", w[idx] == 0, [d.tickers[i] for i in idx])] if idx else []

    return Constraint(
        "exclusions", "Exclusions (ESG / client wishes)", {"tickers": list(tickers)}, build,
        lambda p: "Never invest in: " + (", ".join(p["tickers"]) or "none") + ".")


def max_turnover(limit: float = 0.50) -> Constraint:
    def build(w, d, p):
        if d.w_prev is None:
            return []
        return [Part(f"Turnover <= {p['limit']:.0%}", cp.norm1(w - d.w_prev.values) <= p["limit"])]

    return Constraint(
        "max_turnover", "Turnover limit", {"limit": limit}, build,
        lambda p: f"Total trades (buys + sells) at most {p['limit']:.0%} of the portfolio per rebalancing.")


def max_volatility(limit: float = 0.10) -> Constraint:
    """The risk budget of the profile. Mandatory: without it the problem would put everything
    in the single asset with the highest expected return."""
    return Constraint(
        VOL_KEY, "Maximum volatility", {"limit": limit},
        lambda w, d, p: [Part(f"Volatility <= {p['limit']:.0%}", cp.norm(d.sqrt_cov @ w, 2) <= p["limit"])],
        lambda p: f"Expected annual volatility at most {p['limit']:.0%}.")


def max_tracking_error(limit: float = 0.03) -> Constraint:
    def build(w, d, p):
        if d.benchmark is None:
            return []
        return [Part(f"Tracking error <= {p['limit']:.1%}",
                     cp.norm(d.sqrt_cov @ (w - d.benchmark.values), 2) <= p["limit"])]

    return Constraint(
        "max_tracking_error", "Maximum tracking error", {"limit": limit}, build,
        lambda p: f"Expected deviation from the profile benchmark at most {p['limit']:.1%} per year.")


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
    tracking_error: float = np.nan
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
    te = np.nan
    if data.benchmark is not None:
        a = w - data.benchmark.values
        te = float(np.sqrt(max(a @ data.cov.values @ a, 0.0)))
    return {"expected_return": ret, "volatility": vol,
            "sharpe": ret / vol if vol > TOL else np.nan, "turnover": turnover, "tracking_error": te}


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
