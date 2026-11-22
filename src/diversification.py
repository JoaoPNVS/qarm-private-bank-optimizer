"""Where a portfolio's risk comes from, and how diversified it is (ex ante).

Risk contributions (risk budgeting, Euler allocation principle). With sigma(w) = sqrt(w' Sigma w):
    marginal risk           MR_i  = d sigma / d w_i = (Sigma w)_i / sigma
    absolute contribution   RC_i  = w_i * MR_i
    relative contribution   RC*_i = RC_i / sigma
Volatility is homogeneous of degree one, so by Euler's theorem the contributions add up exactly:
sum_i RC_i = sigma and sum_i RC*_i = 1. A contribution is negative when the asset moves against
the rest of the portfolio (it lowers the total risk).

Diversification measures, on the weights (over every eligible asset, zeros included):
    Herfindahl index       H = sum_i w_i^2             (1 = a single position)
    effective number       1 / H                       (number of equal positions with the same H)
    Shannon entropy        E = -sum_i w_i ln w_i       -> effective number exp(E)
    Gini coefficient       0 = equal weights in every eligible asset, close to 1 = everything in one
and on the risk:
    diversification ratio  DR = sum_i w_i sigma_i / sigma(w)   (1 = no diversification benefit;
                           the most diversified portfolio is the one that maximises it)
    top-10 risk share      share of the volatility coming from the ten largest contributors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TOL = 1e-12
TOP_N = 10


def _aligned(weights: pd.Series, cov: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    w = weights.astype(float)
    return w.values, cov.loc[w.index, w.index].values


def risk_contributions(weights: pd.Series, cov: pd.DataFrame) -> pd.DataFrame:
    """Per asset: weight, marginal risk, absolute and relative risk contribution. A riskless
    portfolio (e.g. all cash) has no risk to split: every contribution is then zero."""
    w, c = _aligned(weights, cov)
    sigma_w = c @ w
    vol = float(np.sqrt(max(w @ sigma_w, 0.0)))
    if vol <= TOL:
        zeros = np.zeros_like(w)
        return pd.DataFrame({"weight": w, "marginal": zeros, "absolute": zeros, "relative": zeros},
                            index=weights.index)
    marginal = sigma_w / vol
    absolute = w * marginal
    return pd.DataFrame({"weight": w, "marginal": marginal, "absolute": absolute, "relative": absolute / vol},
                        index=weights.index)


def by_group(contributions: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """Weights and risk contributions summed by group (e.g. asset class). Euler contributions are
    additive, so a group's contribution is the sum of its members'."""
    return contributions[["weight", "absolute", "relative"]].groupby(groups.reindex(contributions.index)).sum()


def gini(weights: pd.Series) -> float:
    """Gini coefficient of the weights (Lorenz curve over every asset given, zeros included)."""
    x = np.sort(np.clip(weights.astype(float).values, 0.0, None))
    n, total = len(x), x.sum()
    if n < 2 or total <= TOL:
        return 0.0
    return float(2 * np.sum(np.arange(1, n + 1) * x) / (n * total) - (n + 1) / n)


def diversification_measures(weights: pd.Series, cov: pd.DataFrame) -> dict:
    """Concentration of the weights and diversification of the risk (see the module docstring)."""
    w = weights.clip(lower=0).astype(float)
    w = w / w.sum()
    hhi = float((w ** 2).sum())
    held = w[w > TOL]
    entropy = float(-(held * np.log(held)).sum())
    wv, c = _aligned(w, cov)
    vol = float(np.sqrt(max(wv @ c @ wv, 0.0)))
    stand_alone = np.sqrt(np.clip(np.diag(c), 0.0, None))
    rel = risk_contributions(w, cov)["relative"]
    return {"herfindahl": hhi,
            "effective_n": 1 / hhi,
            "entropy_n": float(np.exp(entropy)),
            "gini": gini(w),
            "diversification_ratio": float(wv @ stand_alone / vol) if vol > TOL else np.nan,
            "top10_risk_share": float(rel.nlargest(TOP_N).sum()) if vol > TOL else np.nan}
