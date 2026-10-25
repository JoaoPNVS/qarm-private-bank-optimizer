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
        return w[uni["vehicle"] == "Stock"].max() <= p["limit"] + TOL
    bond = uni["asset_class"] == "Bond"
    if c.key == "max_etf_weight":
        return w[(uni["vehicle"] == "ETF") & ~bond].max() <= p["limit"] + TOL
    if c.key == "max_bond_etf_weight":
        return w[(uni["vehicle"] == "ETF") & bond].max() <= p["limit"] + TOL
    if c.key == "max_bond_duration":
        return (w[bond] * uni.loc[bond, "duration"]).sum() <= p["limit"] * w[bond].sum() + TOL
    if c.key == "min_investment_grade":
        return w[bond & (uni["credit_quality"] == "IG")].sum() >= p["min_share"] * w[bond].sum() - TOL
    if c.key == "max_high_yield":
        return w[bond & (uni["credit_quality"] == "HY")].sum() <= p["limit"] + TOL
    if c.key == "em_equity_cap":
        return w[eq & uni["is_EM"]].sum() <= p["ratio"] * w[eq].sum() + TOL
    if c.key == "max_sector_weight":
        return w[eq].groupby(uni.loc[eq, "sector"]).sum().max() <= p["limit"] + TOL
    if c.key == "exclusions":
        out = uni["sector"].isin(p["sectors"]) | uni.index.isin(p["tickers"])
        return w[out].abs().sum() <= TOL
    if c.key == "min_bonds":
        return w[uni["asset_class"] == "Bond"].sum() >= p["min_share"] - TOL
    if c.key == "max_cash":
        return w[uni["asset_class"] == "Cash"].sum() <= p["limit"] + TOL
    raise KeyError(c.key)
