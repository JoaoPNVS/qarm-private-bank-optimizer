"""Bond sleeve analytics: segments, maturity, credit quality, average duration and rating.

Fund characteristics are the CURRENT issuer fact-sheet values stored in data/universe.csv
(see scripts/build_universe.py); averages are weighted by the portfolio weights of the bond funds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Notch scale used to average ratings; a letter bucket without +/- sits on its middle notch.
RATING_NOTCH = {"AAA": 1, "AA+": 2, "AA": 3, "AA-": 4, "A+": 5, "A": 6, "A-": 7, "BBB+": 8, "BBB": 9, "BBB-": 10,
                "BB+": 11, "BB": 12, "BB-": 13, "B+": 14, "B": 15, "B-": 16, "CCC+": 17, "CCC": 18, "CCC-": 19,
                "CC": 20, "C": 21, "D": 22}
NOTCH_RATING = {v: k for k, v in RATING_NOTCH.items()}
MATURITY_ORDER = ["Ultra-short", "Short", "Intermediate", "Long"]
DUST = 1e-4            # solver residuals below 0.01% of the portfolio are ignored
MIN_SLEEVE = 1e-3      # a bond sleeve below 0.1% of the portfolio is reported as "no bonds"


def rating_from_score(score: float) -> str:
    return NOTCH_RATING[int(np.clip(round(score), 1, 22))] if np.isfinite(score) else "n/a"


def sleeve_summary(weights: pd.Series, universe: pd.DataFrame) -> dict:
    """Shares of the bond sleeve by issuer type, maturity bucket, credit quality and segment, plus
    weighted average duration and rating. No (or only residual) bonds -> zero share, NaN averages."""
    w = weights.reindex(universe.index).fillna(0.0)
    bonds = universe["asset_class"] == "Bond"
    wb = w[bonds & (w > DUST)]
    u = universe.loc[wb.index]
    total = float(wb.sum())
    if total < MIN_SLEEVE:
        return {"share": 0.0, "duration": np.nan, "rating_score": np.nan, "avg_rating": "n/a",
                "ig_share": np.nan, "hy_share_of_portfolio": 0.0, "by_issuer": pd.Series(dtype=float),
                "by_maturity": pd.Series(dtype=float), "by_quality": pd.Series(dtype=float),
                "funds": pd.DataFrame()}
    rel = wb / total
    score = float((rel * u["rating_score"]).sum())
    funds = pd.DataFrame({"Segment": u["bond_type"], "Issuer": u["issuer_type"], "Maturity": u["maturity_bucket"],
                          "Duration": u["duration"], "Rating": u["avg_rating"], "Quality": u["credit_quality"],
                          "Weight": wb, "Share of bonds": rel}).sort_values("Weight", ascending=False)
    return {"share": total,
            "duration": float((rel * u["duration"]).sum()),
            "rating_score": score, "avg_rating": rating_from_score(score),
            "ig_share": float(rel[u["credit_quality"] == "IG"].sum()),
            "hy_share_of_portfolio": float(wb[u["credit_quality"] == "HY"].sum()),
            "by_issuer": rel.groupby(u["issuer_type"]).sum().sort_values(ascending=False),
            "by_maturity": rel.groupby(u["maturity_bucket"]).sum().reindex(MATURITY_ORDER).fillna(0.0),
            "by_quality": rel.groupby(u["credit_quality"]).sum().reindex(["IG", "HY"]).fillna(0.0),
            "funds": funds}
