"""Independent numerical checks of each constraint on a solution (shared by test modules)."""
import numpy as np

TOL = 1e-5


def satisfied(c, w, d):
    uni, p = d.universe, c.params
    w = w.reindex(uni.index).fillna(0.0)
    eq = uni["asset_class"] == "Equity"
    if c.key == "max_volatility":
        return np.sqrt(w @ d.cov @ w) <= p["limit"] + TOL
    if c.key == "max_stock_weight":
        return w[eq].max() <= p["limit"] + TOL
    if c.key == "max_sector_weight":
        return w[eq].groupby(uni.loc[eq, "sector"]).sum().max() <= p["limit"] + TOL
    if c.key == "exclusions":
        out = uni["sector"].isin(p["sectors"]) | uni.index.isin(p["tickers"])
        return w[out].abs().sum() <= TOL
    if c.key == "min_defensive":
        return w[uni["asset_class"].isin(["Bond", "Cash"])].sum() >= p["min_share"] - TOL
    raise KeyError(c.key)
