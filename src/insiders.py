"""Insider-trading house view for Black-Litterman (SEC Form 4 data, scripts/download_insiders.py).

Why insiders. Officers and directors know their companies best. When they buy shares on the open
market with their own money, they signal that they expect prices to rise; sales say much less
(insiders sell to diversify, to pay taxes or under automatic 10b5-1 plans). Insider buying across
ALL companies has predicted the stock market (Seyhun 1988, 1992; Lakonishok & Lee 2001).

The signal: BREADTH of insider buying. Among the US listed companies whose officers or directors
traded their own shares on the open market over the last SHARE_MONTHS months (Form 4, codes P and
S), the share of companies with purchases. Counting companies, not dollars, stops a few huge sales
from dominating. Breadth peaked at the market lows (Dec 2008 - Mar 2009: 60% against about 33%
normally; Oct 2011; May 2020). It has drifted down since 2006 as automatic sale plans spread, so it
is compared with its own last Z_WINDOW months (z-score) rather than with a fixed level.

The view (one relative view, P = +1 SPY - 1 AGG):
    E[SPY] - E[AGG] = equilibrium spread + KAPPA * z,   z capped at +-Z_CAP,
with uncertainty Omega = P tau Sigma P' (1 - c) / c (He & Litterman), c = confidence in the insider
view (sidebar). For this single view, c is its weight against the equilibrium. Because every asset
is correlated with SPY and AGG, the view moves all expected returns: when insiders buy, assets that
behave like equities become more attractive than assets that behave like bonds.

Point in time. A Form 4 is public on its filing date, at most two business days after the trade; the
breadth of month m uses the filings of month m, complete at its month-end. A rebalancing at t uses
breadth up to t only (the z-score too). No view when the latest data is older than STALE_MONTHS.

Evidence, 2009-2026 (rolling z): after the top fifth of months by breadth, SPY beat the T-bill by
about 20% over the next year, against about 12% after the other months (the lower four groups are
not in order); the 12-month predictive regression has a Newey-West t-stat of about 2.2 (1.4 from
2012). The effect is modest and noisy, and insiders tend to buy early (spring 2008). Backtest
2012-2026 at c = 50% vs no view: Sharpe +0.01 to +0.03 in every profile, smaller drawdowns,
realised volatility at the promised budget.

Stock-level purchases are NOT used as views: in our 150 large companies, stocks bought by their
insiders did no better than the same month's universe or sector average over the next 3-12 months
(`purchase_event_study`; t-stats around zero), in line with the literature (the effect is in small
firms). They are shown in the app as information only.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from src import estimators as est

SHARE_MONTHS = 3              # breadth measured over the last 3 months of filings
Z_WINDOW = 60                 # breadth compared with its own last 5 years
Z_MIN_MONTHS = 36             # history needed before the z-score is defined
Z_CAP = 2.0                   # cap on |z| (one extreme reading cannot dominate)
KAPPA = 0.02                  # equity-bond spread added per standard deviation of breadth (annualised)
STALE_MONTHS = 6              # no view when the latest insider data is older than this
DEFAULT_CONFIDENCE = 0.50     # sidebar default
MAX_CONFIDENCE = 0.90
RECENT_MONTHS = 6             # purchases listed in the app
MIN_PURCHASE_USD = 25_000     # smaller purchases are token amounts
MARKET_VIEW = "US equities (SPY) vs US bonds (AGG)"


@dataclass
class InsiderView:
    date: pd.Timestamp
    data_month: pd.Timestamp | None   # month of the latest breadth used (None: no view)
    buy_share: float | None           # breadth at data_month
    z: float | None                   # breadth vs its own last 5 years, capped
    P: pd.DataFrame                   # view portfolio (0 or 1 row), columns = assets of the prior
    Q: pd.Series                      # view value (annualised excess return spread)
    table: pd.DataFrame               # the view for display (equilibrium, view, posterior)


def market_signal(market: pd.DataFrame) -> pd.DataFrame:
    """Monthly breadth of insider buying and its z-score against its own last Z_WINDOW months. Point in
    time: the values of month m use months up to m only."""
    m = market.asfreq("ME", fill_value=0)
    buys = m["buy_companies"].rolling(SHARE_MONTHS).sum()
    sells = m["sell_companies"].rolling(SHARE_MONTHS).sum()
    share = buys / (buys + sells)
    hist = share.rolling(Z_WINDOW, min_periods=Z_MIN_MONTHS)
    return pd.DataFrame({"buy_share": share, "z": (share - hist.mean()) / hist.std()})


def signal_at(signal: pd.DataFrame, date: pd.Timestamp) -> tuple[pd.Timestamp | None, float | None, float | None]:
    """(data month, breadth, capped z) known at `date`; Nones when missing or stale."""
    s = signal.loc[:date].dropna()
    if s.empty or s.index[-1] < date - pd.DateOffset(months=STALE_MONTHS):
        return None, None, None
    row = s.iloc[-1]
    return s.index[-1], float(row["buy_share"]), float(np.clip(row["z"], -Z_CAP, Z_CAP))


def build_view(prior: pd.Series, date: pd.Timestamp, signal: pd.DataFrame) -> InsiderView:
    """The insider view at `date` on the assets of `prior` (investable assets + SPY and AGG)."""
    month, share, z = signal_at(signal, date)
    cols = ["view", "equilibrium", "view_value", "evidence"]
    if z is None or not {"SPY", "AGG"} <= set(prior.index):
        return InsiderView(date, None, None, None, pd.DataFrame(columns=prior.index), pd.Series(dtype=float),
                           pd.DataFrame(columns=cols))
    eq = float(prior["SPY"] - prior["AGG"])
    q = eq + KAPPA * z
    P = pd.DataFrame([pd.Series({"SPY": 1.0, "AGG": -1.0})], index=[MARKET_VIEW]).reindex(
        columns=prior.index, fill_value=0.0).fillna(0.0)
    table = pd.DataFrame([{"view": MARKET_VIEW, "equilibrium": eq, "view_value": q,
                           "evidence": f"insider buying breadth {share:.0%} in the 3 months to {month:%b %Y}, "
                                       f"{z:+.1f} standard deviations vs its last 5 years"}], columns=cols)
    return InsiderView(date, month, share, z, P, pd.Series({MARKET_VIEW: q}), table)


def with_insider_view(e: est.Estimates, date: pd.Timestamp, signal: pd.DataFrame,
                      confidence: float = DEFAULT_CONFIDENCE) -> tuple[est.Estimates, InsiderView]:
    """The window's estimates with Black-Litterman expected returns from the insider view at `date`
    (assets and benchmark series alike). Covariances are unchanged; without a view the prior is used."""
    if e.prior_mu is None or e.joint_cov is None:
        raise est.EstimationError("The insider view needs estimates computed with the SPY and AGG benchmark series.")
    if not 0 <= confidence <= MAX_CONFIDENCE:
        raise est.EstimationError(f"Confidence in the insider view must be between 0 and {MAX_CONFIDENCE:.0%}.")
    v = build_view(e.prior_mu, date, signal)
    post = est.black_litterman_posterior(e.prior_mu, e.joint_cov, v.P, v.Q, confidence)
    v.table["posterior"] = [float(post["SPY"] - post["AGG"])] * len(v.table)
    bench = [t for t in post.index if t not in e.mu.index]
    return replace(e, mu=post.reindex(e.mu.index), benchmark_mu=post.reindex(bench), method="insider",
                   view_confidence=float(confidence)), v


# --- information and evidence (app) -----------------------------------------------------------

def forward_return(r: pd.Series | pd.DataFrame, horizon: int):
    """Compounded return over the NEXT `horizon` months, aligned on the current month (NaN at the end)."""
    return np.expm1(np.log1p(r).rolling(horizon).sum().shift(-horizon))


def recent_purchases(trades: pd.DataFrame, date: pd.Timestamp, months: int = RECENT_MONTHS,
                     min_value: float = MIN_PURCHASE_USD) -> pd.DataFrame:
    """Open-market purchases by officers or directors of our stocks filed in (date - months, date]."""
    start = date - pd.DateOffset(months=months)
    b = trades[(trades["direction"] == "buy") & (trades["filing_date"] > start) & (trades["filing_date"] <= date)
               & (trades["value_usd"] >= min_value)]
    return b.sort_values("filing_date", ascending=False).reset_index(drop=True)


def market_evidence(signal: pd.DataFrame, spy_excess: pd.Series, horizon: int = 12) -> pd.DataFrame:
    """Average SPY excess return over the next `horizon` months by quintile of the breadth z-score
    (whole history: an in-sample description, not a backtest)."""
    fwd = forward_return(spy_excess, horizon)
    d = pd.concat([signal["z"].rename("z"), fwd.rename("fwd")], axis=1).dropna()
    if len(d) < 25:
        return pd.DataFrame(columns=["z_range", "next_return", "months"])
    labels = ["1 - least buying", "2", "3", "4", "5 - most buying"]
    d["quintile"] = pd.qcut(d["z"], 5, labels=labels)
    g = d.groupby("quintile", observed=True)
    return pd.DataFrame({"z_range": g["z"].agg(lambda s: f"{s.min():+.1f} to {s.max():+.1f}"),
                         "next_return": g["fwd"].mean(), "months": g.size()})


def purchase_event_study(trades: pd.DataFrame, returns: pd.DataFrame, universe: pd.DataFrame,
                         horizons: tuple[int, ...] = (3, 6, 12), min_value: float = MIN_PURCHASE_USD) -> pd.DataFrame:
    """Return of our stocks after a month with an insider purchase, relative to the SAME months'
    equal-weight average of our stocks and of the stock's sector (mean, t-stat, number of events)."""
    stocks = universe.index[universe["vehicle"] == "Stock"].intersection(returns.columns)
    r = returns[stocks]
    rel = {"universe": r.sub(r.mean(axis=1), axis=0),
           "sector": r - r.T.groupby(universe.loc[stocks, "sector"]).transform("mean").T}
    b = trades[(trades["direction"] == "buy") & (trades["value_usd"] >= min_value) & trades["ticker"].isin(stocks)]
    events = (b.assign(month=b["filing_date"] + pd.offsets.MonthEnd(0))[["ticker", "month"]].drop_duplicates())
    rows = []
    for h in horizons:
        row = {"horizon": f"{h} months"}
        for name, x in rel.items():
            stacked = forward_return(x, h).stack()
            vals = stacked.reindex(pd.MultiIndex.from_frame(events[["month", "ticker"]])).dropna()
            row[f"vs_{name}"] = vals.mean() if len(vals) else np.nan
            row[f"t_{name}"] = vals.mean() / (vals.std() / np.sqrt(len(vals))) if len(vals) > 2 else np.nan
            row["events"] = len(vals)
        rows.append(row)
    return pd.DataFrame(rows).set_index("horizon")
