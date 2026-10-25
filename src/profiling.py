"""Suitability questionnaire -> risk profile -> volatility cap, benchmark and default constraints.

A standard suitability assessment covering the client's financial situation, investment
objectives, and knowledge & experience. Seven questions, each answer scored 1-4:
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

Risk budget: each profile has a maximum expected volatility; the optimiser maximises expected
return under that cap (src/optimizer.py), and a benchmark mixing SPY and US aggregate bonds
(src/benchmarks.py). Each profile also sets a floor on bond funds (cash does not count) equal
to its benchmark's bond share (70 / 40 / 20 / 0%), active by default. The other default
constraints are the same for every profile: at most 10% cash, 8% per stock, 15% per bond ETF,
20% per other ETF, 40% per equity sector (international ETFs as one group), emerging markets at most 1/3 of equities, no
exclusions. The bond sleeve also has profile limits: average duration at most 5 / 6 / 7 / 8 years,
investment grade at least 90 / 80 / 70 / 60% of the bonds, high yield (HYG, EMB) at most
5 / 10 / 15 / 20% of the portfolio.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import optimizer as opt
from src.estimators import Estimates

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
    max_volatility: float      # risk budget (annualised, ex ante), calibrated on the benchmark
    benchmark_equity: float    # SPY share of the profile benchmark, the rest in AGG
    max_bond_duration: float   # years, average of the bond sleeve (AGG is about 6)
    min_investment_grade: float  # share of the bond sleeve
    max_high_yield: float      # share of the whole portfolio (HYG + EMB)

    @property
    def min_bonds(self) -> float:
        """Bond floor of the profile: the benchmark's bond share."""
        return round(1 - self.benchmark_equity, 2)


# Volatility cap = historical volatility of the profile benchmark (2003-10 to 2026-09) rounded up
# to the next percent: 5.91 / 9.28 / 11.85 / 14.52% -> 6 / 10 / 12 / 15%, see src/benchmarks.py.
SPECS = {
    "Conservative": ProfileSpec("Conservative", "Capital preservation first; low fluctuations.", 0.06, 0.30,
                                5.0, 0.90, 0.05),
    "Balanced": ProfileSpec("Balanced", "Balance between growth and stability.", 0.10, 0.60, 6.0, 0.80, 0.10),
    "Growth": ProfileSpec("Growth", "Long-term capital growth; accepts significant fluctuations.", 0.12, 0.80,
                          7.0, 0.70, 0.15),
    "Dynamic": ProfileSpec("Dynamic", "Maximum long-term growth; accepts large drawdowns.", 0.15, 1.00,
                           8.0, 0.60, 0.20),
}
MAX_CASH = 0.10
MAX_STOCK_WEIGHT = 0.08
MAX_ETF_WEIGHT = 0.20       # equity and gold funds
MAX_BOND_ETF_WEIGHT = 0.15
MAX_SECTOR_WEIGHT = 0.40
EM_EQUITY_CAP = 1 / 3


def default_constraints(profile: str) -> list[opt.Constraint]:
    s = SPECS[profile]
    return [opt.max_volatility(s.max_volatility), opt.min_bonds(s.min_bonds), opt.max_cash(MAX_CASH),
            opt.max_stock_weight(MAX_STOCK_WEIGHT), opt.max_etf_weight(MAX_ETF_WEIGHT),
            opt.max_bond_etf_weight(MAX_BOND_ETF_WEIGHT), opt.max_bond_duration(s.max_bond_duration),
            opt.min_investment_grade(s.min_investment_grade), opt.max_high_yield(s.max_high_yield),
            opt.max_sector_weight(MAX_SECTOR_WEIGHT), opt.em_equity_cap(EM_EQUITY_CAP), opt.exclusions()]


def problem_data(profile: str, estimates: Estimates, universe: pd.DataFrame,
                 w_prev: pd.Series | None = None, tc: float = 0.001) -> opt.ProblemData:
    return opt.ProblemData(estimates.mu, estimates.cov, universe, w_prev=w_prev, tc=tc)
