"""Suitability questionnaire -> risk profile -> lambda, CHF benchmark and default constraints.

Inspired by the LSFin/FIDLEG suitability test (art. 12 LSFin: the client's financial situation,
investment objectives, and knowledge & experience). Seven questions, each answer scored 1-4:
- Risk CAPACITY (can the client bear losses?): horizon, income stability, share of wealth.
- Risk WILLINGNESS (does the client want to?): objective, tolerable loss, reaction to a drawdown.
- KNOWLEDGE & experience.
Mapping, kept deliberately simple so it can be explained to the client:
1. capacity and willingness are each averaged; the average maps to a level with the cut-offs
   <1.75 Conservative, <2.50 Balanced, <3.25 Growth, otherwise Dynamic;
2. the profile is the LOWER of the two levels (prudence: a client who wants risk but cannot
   afford it, or the reverse, gets the lower profile);
3. knowledge caps the profile: "none" -> at most Balanced, "basic" -> at most Growth;
4. knock-out: a horizon below 3 years always gives Conservative.

Each profile has a CHF benchmark built from the universe (strategic allocation with a Swiss
home bias), and default constraints set around it. Every benchmark satisfies its own default
constraints (tested), so the defaults are always feasible.

Risk aversion: lambda = REFERENCE_SHARPE / sigma(benchmark). Rationale: with equilibrium returns
and a risk-free asset, the unconstrained optimum for risk aversion lambda is
(delta / lambda) * w_ref + rest in cash, whose volatility is (delta / lambda) * sigma_ref =
Sharpe / lambda. Choosing lambda = Sharpe / sigma_benchmark therefore targets the risk level of
the profile benchmark. lambda depends on the estimation window (it is recomputed with Sigma).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import optimizer as opt
from src.estimators import REFERENCE_SHARPE, Estimates

PROFILES = ["Conservative", "Balanced", "Growth", "Dynamic"]
LEVEL_CUTOFFS = [1.75, 2.50, 3.25]
KNOWLEDGE_CAP = {1: "Balanced", 2: "Growth"}
SHORT_HORIZON_ANSWER = 0  # option index of "less than 3 years"


@dataclass(frozen=True)
class Question:
    key: str
    text: str
    dimension: str              # capacity | willingness | knowledge
    options: tuple[str, ...]    # answer i scores i + 1


QUESTIONS = [
    Question("horizon", "When will you need most of this money?", "capacity",
             ("In less than 3 years", "In 3 to 5 years", "In 5 to 10 years", "In more than 10 years")),
    Question("income", "How stable are your income and savings capacity?", "capacity",
             ("Unstable, I may need to withdraw from this portfolio", "Rather stable, little saving",
              "Stable, regular savings", "Very stable, large savings capacity")),
    Question("wealth_share", "What share of your total wealth does this portfolio represent?", "capacity",
             ("More than 75%", "50% to 75%", "25% to 50%", "Less than 25%")),
    Question("objective", "What is your main investment objective?", "willingness",
             ("Preserve capital", "Regular income with limited fluctuations",
              "Balanced growth of capital", "Maximise long-term growth")),
    Question("max_loss", "What is the largest loss over one year you could accept?", "willingness",
             ("5%", "10%", "20%", "More than 30%")),
    Question("drawdown", "Your portfolio falls by 20% in a few months. What do you do?", "willingness",
             ("Sell everything", "Sell part of it", "Hold and wait", "Invest more")),
    Question("knowledge", "Which statement best describes your investment knowledge and experience?",
             "knowledge",
             ("None", "Basic: savings accounts, some funds",
              "Good: equities and bonds for several years", "Professional or extensive experience")),
]


@dataclass
class Assessment:
    profile: str
    scores: dict[str, int]          # per question, 1-4
    dimensions: dict[str, float]    # averages per dimension
    levels: dict[str, str]          # profile implied by capacity / willingness
    reasons: list[str]              # plain-language explanation for the client


def _level(avg: float) -> int:
    return int(np.searchsorted(LEVEL_CUTOFFS, avg, side="right"))


def assess(answers: dict[str, int]) -> Assessment:
    """answers: question key -> chosen option index (0-based)."""
    scores = {}
    for q in QUESTIONS:
        if q.key not in answers:
            raise ValueError(f"Missing answer to '{q.text}'")
        a = answers[q.key]
        if not 0 <= a < len(q.options):
            raise ValueError(f"Invalid answer {a} to '{q.text}'")
        scores[q.key] = a + 1
    dims = {d: float(np.mean([scores[q.key] for q in QUESTIONS if q.dimension == d]))
            for d in ("capacity", "willingness", "knowledge")}
    cap, will = _level(dims["capacity"]), _level(dims["willingness"])
    level = min(cap, will)
    reasons = [f"Risk capacity score {dims['capacity']:.2f}/4 -> {PROFILES[cap]}.",
               f"Risk willingness score {dims['willingness']:.2f}/4 -> {PROFILES[will]}.",
               f"The lower of the two is retained: {PROFILES[level]}."]
    cap_name = KNOWLEDGE_CAP.get(scores["knowledge"])
    if cap_name and level > PROFILES.index(cap_name):
        level = PROFILES.index(cap_name)
        reasons.append(f"Limited investment knowledge: profile capped at {cap_name}.")
    if answers["horizon"] == SHORT_HORIZON_ANSWER and level > 0:
        level = 0
        reasons.append("Investment horizon below 3 years: Conservative profile required.")
    return Assessment(PROFILES[level], scores, dims,
                      {"capacity": PROFILES[cap], "willingness": PROFILES[will]}, reasons)


@dataclass(frozen=True)
class ProfileSpec:
    name: str
    description: str
    benchmark: dict[str, float]
    asset_bands: dict[str, tuple[float, float]]
    max_weight: float
    home_bias: float        # min Swiss share of equities
    max_non_chf: float
    max_volatility: float
    max_tracking_error: float


# Share of the EQUITY allocation per region, common to all profiles: keeps Europe and Japan in
# every portfolio and avoids a single foreign market dominating. EM is capped separately (1/3).
REGION_BANDS = {"Switzerland": (0.0, 0.60), "US": (0.15, 0.50), "Europe": (0.08, 0.30),
                "Japan": (0.03, 0.15), "Asia-Pacific": (0.0, 0.10)}
EM_EQUITY_CAP = 1 / 3

SPECS = {
    "Conservative": ProfileSpec(
        "Conservative", "Capital preservation first; mostly Swiss franc bonds, 25% equities.",
        {"CSSMI.SW": .10, "SPY": .075, "EZU": .03, "EWU": .01, "EWJ": .015, "EEM": .02,
         "CSBGC3.SW": .20, "CSBGC7.SW": .20, "CSBGC0.SW": .15, "LQD": .04, "BWX": .03, "IEF": .03,
         "GLD": .03, "SRECHA.SW": .04, "CASH_CHF": .03},
        {"Equity": (0.15, 0.35), "Bond": (0.50, 0.80), "Alternative": (0.0, 0.10),
         "Real Estate": (0.0, 0.10), "Cash": (0.0, 0.15)},
        max_weight=0.20, home_bias=0.30, max_non_chf=0.30, max_volatility=0.06, max_tracking_error=0.02),
    "Balanced": ProfileSpec(
        "Balanced", "Balance between growth and stability; 50% equities, 40% bonds.",
        {"CSSMI.SW": .15, "CSSMIM.SW": .05, "SPY": .14, "IWM": .01, "EZU": .05, "EWU": .02, "EWJ": .03,
         "EPP": .01, "EEM": .04, "CSBGC3.SW": .10, "CSBGC7.SW": .14, "CSBGC0.SW": .10, "LQD": .03,
         "EMB": .02, "IEF": .01, "GLD": .03, "SRECHA.SW": .04, "CASH_CHF": .03},
        {"Equity": (0.35, 0.60), "Bond": (0.30, 0.50), "Alternative": (0.0, 0.15),
         "Real Estate": (0.0, 0.10), "Cash": (0.0, 0.10)},
        max_weight=0.15, home_bias=0.30, max_non_chf=0.40, max_volatility=0.09, max_tracking_error=0.03),
    "Growth": ProfileSpec(
        "Growth", "Long-term capital growth; 70% equities, accepts significant fluctuations.",
        {"CSSMI.SW": .15, "CSSMIM.SW": .07, "SPY": .15, "QQQ": .04, "IWM": .03, "EZU": .08, "EWU": .03,
         "EWJ": .05, "EPP": .02, "EEM": .06, "AAXJ": .02, "CSBGC7.SW": .08, "CSBGC0.SW": .06, "LQD": .02,
         "HYG": .02, "EMB": .02, "GLD": .04, "SRECHA.SW": .03, "VNQ": .02, "CASH_CHF": .01},
        {"Equity": (0.55, 0.80), "Bond": (0.10, 0.35), "Alternative": (0.0, 0.15),
         "Real Estate": (0.0, 0.10), "Cash": (0.0, 0.10)},
        max_weight=0.15, home_bias=0.25, max_non_chf=0.65, max_volatility=0.12, max_tracking_error=0.04),
    "Dynamic": ProfileSpec(
        "Dynamic", "Maximum long-term growth; 90% equities, accepts large drawdowns.",
        {"CSSMI.SW": .15, "CSSMIM.SW": .10, "SPY": .15, "QQQ": .08, "IWM": .05, "EZU": .10, "EWU": .04,
         "EWJ": .07, "EPP": .03, "EEM": .09, "AAXJ": .04, "CSBGC7.SW": .03, "HYG": .02, "GLD": .03,
         "VNQ": .02},
        {"Equity": (0.75, 1.00), "Bond": (0.0, 0.20), "Alternative": (0.0, 0.15),
         "Real Estate": (0.0, 0.10), "Cash": (0.0, 0.10)},
        max_weight=0.15, home_bias=0.20, max_non_chf=0.75, max_volatility=0.15, max_tracking_error=0.05),
}


def benchmark_weights(profile: str, tickers: list[str]) -> pd.Series:
    """Profile benchmark over `tickers`, renormalised if some assets are not selected."""
    w = pd.Series(SPECS[profile].benchmark, dtype=float).reindex(tickers).fillna(0.0)
    if w.sum() <= 0:
        raise ValueError(f"No asset of the {profile} benchmark is in the selection.")
    return (w / w.sum()).rename("benchmark")


def risk_aversion(profile: str, cov: pd.DataFrame) -> float:
    """lambda = REFERENCE_SHARPE / benchmark volatility (see module docstring)."""
    b = benchmark_weights(profile, list(cov.index))
    return REFERENCE_SHARPE / float(np.sqrt(b @ cov @ b))


def default_constraints(profile: str) -> list[opt.Constraint]:
    s = SPECS[profile]
    return [
        opt.max_weight(s.max_weight),
        opt.asset_class_bands(dict(s.asset_bands)),
        opt.region_bands(dict(REGION_BANDS)),
        opt.em_equity_cap(EM_EQUITY_CAP),
        opt.home_bias(s.home_bias),
        opt.max_non_chf(s.max_non_chf),
        opt.exclusions([]),
        opt.max_turnover(1.0),          # only binds when current holdings are given
        opt.max_volatility(s.max_volatility),
        opt.max_tracking_error(s.max_tracking_error),
    ]


def problem_data(profile: str, estimates: Estimates, universe: pd.DataFrame,
                 w_prev: pd.Series | None = None, tc: float = 0.001) -> opt.ProblemData:
    tickers = list(estimates.mu.index)
    return opt.ProblemData(estimates.mu, estimates.cov, universe, risk_aversion(profile, estimates.cov),
                           w_prev=w_prev, tc=tc, benchmark=benchmark_weights(profile, tickers))
