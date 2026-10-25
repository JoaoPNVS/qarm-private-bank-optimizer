"""Independent numerical checks of each constraint on a solution (shared by test modules)."""
import numpy as np

TOL = 1e-5


def share(w, uni, cond):
    return float(w[cond.loc[w.index]].sum())


def satisfied(c, w, d):
    uni, p = d.universe, c.params
    is_eq = uni["asset_class"] == "Equity"
    eq = share(w, uni, is_eq)
    if c.key == "max_weight":
        return w.max() <= p["limit"] + TOL
    if c.key == "asset_class_bands":
        return all(lo - TOL <= share(w, uni, uni["asset_class"] == k) <= hi + TOL for k, (lo, hi) in p["bands"].items())
    if c.key == "region_bands":
        return all(lo * eq - TOL <= share(w, uni, is_eq & (uni["region"] == r)) <= hi * eq + TOL
                   for r, (lo, hi) in p["bands"].items())
    if c.key == "em_equity_cap":
        return share(w, uni, is_eq & uni["is_EM"]) <= p["ratio"] * eq + TOL
    if c.key == "home_bias":
        return share(w, uni, is_eq & uni["is_swiss"]) >= p["min_share"] * eq - TOL
    if c.key == "max_non_chf":
        return share(w, uni, uni["non_chf"]) <= p["limit"] + TOL
    if c.key == "exclusions":
        return w.reindex(p["tickers"]).fillna(0).abs().max() <= TOL if p["tickers"] else True
    if c.key == "max_turnover":
        return d.w_prev is None or (w - d.w_prev).abs().sum() <= p["limit"] + TOL
    if c.key == "max_volatility":
        return np.sqrt(w @ d.cov @ w) <= p["limit"] + TOL
    if c.key == "max_tracking_error":
        a = w - d.benchmark
        return np.sqrt(a @ d.cov @ a) <= p["limit"] + TOL
    raise KeyError(c.key)
