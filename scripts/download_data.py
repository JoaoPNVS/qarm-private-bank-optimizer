"""Download all market data ONCE and save it to data/*.parquet (committed to the repo).

The Streamlit app never calls the network: it only reads the files written here.
Everything is in USD; there is no currency conversion.

Sources (all yfinance)
- Prices: daily adjusted close (dividends reinvested), sampled at month-end, from 1998.
- Risk-free: 3-month T-bill yield ^IRX (% p.a.); monthly rf_t = IRX_{t-1} / 1200, i.e. the
  rate known at the start of the month. CASH_USD earns rf.
- Benchmarks (not investable): SPY (S&P 500) and AGG (US aggregate bonds).
- Liquidity and trading costs: average daily dollar volume (unadjusted close x volume) over the
  12 months to the last month-end, written to universe.csv (adv_usd_m, liquidity_tier, tc_bps).
  Run scripts/build_universe.py first: it rewrites universe.csv without these columns.

Usage:  python scripts/download_data.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
START = "1998-01-01"
BENCHMARK_TICKERS = ["SPY", "AGG"]
RF_TICKER = "^IRX"
CASH_TICKER = "CASH_USD"
MAX_ABS_MONTHLY_RETURN = 2.00   # single stocks can gain 60-170% in a month; above +200% = bad tick
WARN_ABS_MONTHLY_RETURN = 1.00  # printed for a manual check
MAX_ABS_DAILY_MOVE = 1.00       # a single-day move above 100% is printed for a manual check
# Moves above MAX_ABS_MONTHLY_RETURN checked by hand against daily prices and volumes (real).
REVIEWED_MOVES = {
    ("REGN", "2000-02"): "biotech bubble: $13 -> $56 over the month on 10-20x normal volume",
}
# Trading cost per unit traded (one way, bps), by liquidity tier. Stocks are tiered by average
# daily dollar volume (12 months to 2026-09: median $1.2bn, 43 stocks >= $2bn, 22 < $500m).
# ETFs are not tiered by volume (their liquidity comes from the underlying via creation and
# redemption: VGK trades $0.26bn a day with a 1-2 bp spread): high-yield and EM bond funds 8 bps,
# every other fund (bonds, BIL/FLOT, equity funds, gold) 3 bps. Cash is free.
STOCK_TIERS = [(2_000, "Very liquid stock", 2), (500, "Liquid stock", 5), (0, "Less liquid stock", 10)]
HY_EM_BOND_FUNDS = {"HYG", "EMB"}


def last_complete_month_end() -> pd.Timestamp:
    """Previous month-end: the current month is never complete at download time."""
    return pd.Timestamp.today().normalize() - pd.offsets.MonthEnd(1)


def download_daily(tickers: list[str]) -> pd.DataFrame:
    raw = yf.download(tickers, start=START, interval="1d", auto_adjust=True,
                      progress=False, threads=True)["Close"]
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    missing = [t for t in tickers if t not in raw.columns or raw[t].dropna().empty]
    if missing:
        sys.exit(f"ERROR: no data returned for {missing}")
    return raw[tickers]


def liquidity_and_costs(universe: pd.DataFrame, end: pd.Timestamp) -> pd.DataFrame:
    """adv_usd_m (average daily dollar volume, $m, 12 months to `end`), liquidity_tier, tc_bps."""
    tickers = universe.loc[universe["ticker"] != CASH_TICKER, "ticker"].tolist()
    raw = yf.download(tickers, start=end - pd.DateOffset(months=13), end=end + pd.Timedelta(days=1),
                      auto_adjust=False, progress=False, threads=True)
    dollar = (raw["Close"] * raw["Volume"]).loc[end - pd.DateOffset(months=12):end]
    adv = (dollar.mean() / 1e6).reindex(tickers)
    if adv.isna().any() or (adv <= 0).any():
        sys.exit(f"ERROR: no volume data for {adv.index[adv.isna() | (adv <= 0)].tolist()}")
    out = universe[["ticker", "vehicle"]].set_index("ticker")
    out["adv_usd_m"] = adv.round(1)
    for t, row in out.iterrows():
        if row["vehicle"] == "Cash":
            tier, bps = "Cash", 0
        elif row["vehicle"] == "ETF":
            tier, bps = ("HY / EM bond fund", 8) if t in HY_EM_BOND_FUNDS else ("Fund", 3)
        else:
            tier, bps = next((name, c) for floor, name, c in STOCK_TIERS if adv[t] >= floor)
        out.loc[t, ["liquidity_tier", "tc_bps"]] = [tier, bps]
    return out[["adv_usd_m", "liquidity_tier", "tc_bps"]]


def check_daily_jumps(daily: pd.DataFrame) -> None:
    """Warn about daily moves that usually reveal bad ticks (e.g. a misplaced decimal)."""
    jumps = daily.pct_change(fill_method=None).abs()
    bad = jumps[jumps > MAX_ABS_DAILY_MOVE].stack()
    for (date, t), r in bad.items():
        print(f"WARNING: {t} daily move {r:.0%} on {date.date()} — check data_start in universe.csv")


def main() -> None:
    universe = pd.read_csv(DATA_DIR / "universe.csv")
    investable = universe.loc[universe["ticker"] != CASH_TICKER, "ticker"].tolist()
    tickers = investable + BENCHMARK_TICKERS
    end = last_complete_month_end()

    print(f"Downloading {len(tickers)} assets + {RF_TICKER} from yfinance ...")
    daily = download_daily(tickers + [RF_TICKER])

    # Discard known-corrupted early history (universe.csv "data_start").
    for t, start in universe.dropna(subset=["data_start"]).set_index("ticker")["data_start"].items():
        daily.loc[daily.index < pd.Timestamp(start), t] = np.nan
    check_daily_jumps(daily[tickers])

    # last() takes the last valid quote of each month (NaN if the asset did not trade).
    monthly = daily.resample("ME").last().loc[:end]
    prices = monthly[tickers]
    irx = monthly[RF_TICKER].rename("irx_pct")
    if irx.isna().any():
        sys.exit(f"ERROR: missing {RF_TICKER} months: {irx.index[irx.isna()].strftime('%Y-%m').tolist()}")

    returns = prices.pct_change(fill_method=None)
    rf = (irx.shift(1) / 1200).rename("rf_monthly")
    returns = pd.concat([returns, rf.rename(CASH_TICKER)], axis=1).iloc[1:].dropna(how="all")

    big = returns.abs().max()
    for t, r in big[big > WARN_ABS_MONTHLY_RETURN].items():
        print(f"NOTE: {t} largest monthly move {r:+.0%} in {returns[t].abs().idxmax():%Y-%m}")
    extreme = returns.abs().stack()
    extreme = [(t, d.strftime("%Y-%m")) for (d, t), r in extreme[extreme > MAX_ABS_MONTHLY_RETURN].items()]
    unreviewed = [m for m in extreme if m not in REVIEWED_MOVES]
    if unreviewed:
        sys.exit(f"ERROR: implausible monthly returns (review them, then add to REVIEWED_MOVES): {unreviewed}")

    DATA_DIR.mkdir(exist_ok=True)
    prices.to_parquet(DATA_DIR / "prices.parquet")
    returns[investable + [CASH_TICKER]].to_parquet(DATA_DIR / "returns.parquet")
    returns[BENCHMARK_TICKERS].to_parquet(DATA_DIR / "benchmarks.parquet")
    pd.concat([irx, rf], axis=1).iloc[1:].to_parquet(DATA_DIR / "rf.parquet")
    meta = {
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "currency": "USD",
        "last_month_end": str(end.date()),
        "first_return": str(returns.index[0].date()),
        "n_assets": len(investable) + 1,
        "sources": {"prices": "yfinance daily auto_adjust -> month-end", "benchmarks": BENCHMARK_TICKERS,
                    "rf": f"{RF_TICKER} (3-month T-bill, % p.a.), lagged 1 month, /1200"},
    }
    print("Computing liquidity tiers and trading costs ...")
    costs = liquidity_and_costs(universe, end)
    universe = universe.drop(columns=[c for c in costs.columns if c in universe.columns]).set_index("ticker")
    universe.join(costs).reset_index().to_csv(DATA_DIR / "universe.csv", index=False)
    meta["liquidity"] = f"average daily dollar volume, 12 months to {end.date()}"
    (DATA_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved returns {returns.shape} from {meta['first_return']} to {meta['last_month_end']}")


if __name__ == "__main__":
    main()
