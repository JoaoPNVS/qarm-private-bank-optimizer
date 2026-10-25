"""Download all market data ONCE and save it to data/*.parquet (committed to the repo).

The Streamlit app never calls the network: it only reads the files written here.

Sources
- Prices: yfinance daily adjusted close (dividends reinvested), sampled at month-end.
- FX: USDCHF=X (CHF per USD) converts every USD-listed asset to CHF; EURCHF=X kept for reporting.
- Risk-free: SARON from the SNB data portal (cube "zimoma", free, no key). yfinance has no
  reliable CHF short rate, hence this documented exception to the "yfinance only" rule.

Usage:  python scripts/download_data.py
"""
from __future__ import annotations

import io
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
START = "1995-01-01"
FX_TICKERS = ["USDCHF=X", "EURCHF=X"]
CASH_TICKER = "CASH_CHF"
SNB_URL = "https://data.snb.ch/api/cube/zimoma/data/csv/en"
MAX_ABS_MONTHLY_RETURN = 0.60  # above this a monthly move is treated as a data error
MAX_ABS_DAILY_MOVE = 0.25  # warning threshold for a single-day move


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


def download_saron() -> pd.Series:
    """Monthly SARON (% p.a.) from the SNB data portal."""
    with urllib.request.urlopen(SNB_URL, timeout=60) as resp:
        text = resp.read().decode("utf-8-sig")
    df = pd.read_csv(io.StringIO(text), sep=";", skiprows=2)
    s = df[df["D0"] == "SARON"].dropna(subset=["Value"])
    out = pd.Series(s["Value"].astype(float).values,
                    index=pd.to_datetime(s["Date"]) + pd.offsets.MonthEnd(0), name="SARON")
    if out.empty:
        sys.exit("ERROR: SARON series not found in SNB cube zimoma")
    return out.sort_index()


def check_daily_jumps(daily: pd.DataFrame) -> None:
    """Warn about daily moves that usually reveal bad ticks (e.g. a misplaced decimal)."""
    jumps = daily.pct_change(fill_method=None).abs()
    bad = jumps[jumps > MAX_ABS_DAILY_MOVE].stack()
    for (date, t), r in bad.items():
        print(f"WARNING: {t} daily move {r:.0%} on {date.date()} — check data_start in universe.csv")


def main() -> None:
    universe = pd.read_csv(DATA_DIR / "universe.csv")
    tickers = universe.loc[universe["ticker"] != CASH_TICKER, "ticker"].tolist()
    end = last_complete_month_end()

    print(f"Downloading {len(tickers)} assets + FX from yfinance ...")
    daily = download_daily(tickers + FX_TICKERS)

    # Discard known-corrupted early history (universe.csv "data_start").
    for t, start in universe.dropna(subset=["data_start"]).set_index("ticker")["data_start"].items():
        daily.loc[daily.index < pd.Timestamp(start), t] = np.nan
    check_daily_jumps(daily)

    # last() takes the last valid quote of each month (NaN if the asset did not trade).
    monthly = daily.resample("ME").last().loc[:end]

    prices_local = monthly[tickers]
    fx = monthly[FX_TICKERS]

    listed = universe.set_index("ticker")["listed_currency"]
    prices_chf = prices_local.copy()
    usd = [t for t in tickers if listed[t] == "USD"]
    prices_chf[usd] = prices_local[usd].mul(fx["USDCHF=X"], axis=0)
    unknown = set(listed[tickers]) - {"CHF", "USD"}
    if unknown:
        sys.exit(f"ERROR: no FX conversion implemented for listed currencies {unknown}")

    returns_chf = prices_chf.pct_change(fill_method=None)

    print("Downloading SARON from the SNB data portal ...")
    saron = download_saron().loc[:end]
    # Return of month t is earned at the rate known at the end of month t-1.
    rf = (saron.shift(1) / 1200).rename("rf").reindex(returns_chf.index)
    returns_chf[CASH_TICKER] = rf
    returns_chf = returns_chf.iloc[1:].dropna(how="all")

    too_big = returns_chf.abs().max()
    too_big = too_big[too_big > MAX_ABS_MONTHLY_RETURN]
    if not too_big.empty:
        sys.exit(f"ERROR: implausible monthly CHF returns:\n{too_big}")

    DATA_DIR.mkdir(exist_ok=True)
    prices_local.to_parquet(DATA_DIR / "prices_local.parquet")
    fx.to_parquet(DATA_DIR / "fx.parquet")
    returns_chf.to_parquet(DATA_DIR / "returns_chf.parquet")
    pd.DataFrame({"saron_pct": saron, "rf_monthly": rf.reindex(saron.index)}).to_parquet(
        DATA_DIR / "rf_chf.parquet")
    meta = {
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "last_month_end": str(end.date()),
        "first_return": str(returns_chf.index[0].date()),
        "n_assets": int(returns_chf.shape[1]),
        "sources": {"prices": "yfinance daily auto_adjust -> month-end",
                    "fx": FX_TICKERS, "rf": f"SNB SARON ({SNB_URL}), lagged 1 month, /1200"},
    }
    (DATA_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved returns {returns_chf.shape} from {meta['first_return']} to {meta['last_month_end']}")


if __name__ == "__main__":
    main()
