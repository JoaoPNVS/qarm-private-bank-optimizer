"""Load parquet data, CHF conversion and universe tags.

All files are produced once by scripts/build_universe.py and scripts/download_data.py; nothing
here touches the network. Every loader raises DataError with a message that the app can show
to the user as-is.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CASH_TICKER = "CASH_CHF"
BENCHMARK_TICKER = "SPY"
UNIVERSE_COLUMNS = ["ticker", "name", "asset_class", "sector", "isin_country", "index",
                    "listed_currency", "data_start"]
ASSET_CLASSES = {"Equity", "Bond", "Cash"}


class DataError(RuntimeError):
    """Missing or inconsistent data file."""


def _read_parquet(name: str, data_dir: Path) -> pd.DataFrame:
    path = data_dir / name
    if not path.exists():
        raise DataError(f"Data file '{name}' not found. Run `python scripts/download_data.py`.")
    try:
        return pd.read_parquet(path)
    except Exception as exc:  # corrupted file, wrong engine, ...
        raise DataError(f"Could not read '{name}': {exc}") from exc


def load_universe(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Investable universe (stocks, Swiss government bond ETFs, CHF cash), indexed by ticker."""
    path = data_dir / "universe.csv"
    if not path.exists():
        raise DataError("data/universe.csv not found. Run `python scripts/build_universe.py`.")
    uni = pd.read_csv(path, parse_dates=["data_start"], keep_default_na=False, na_values={"data_start": [""]})
    missing = set(UNIVERSE_COLUMNS) - set(uni.columns)
    if missing:
        raise DataError(f"universe.csv is missing columns {sorted(missing)}")
    if uni["ticker"].duplicated().any():
        raise DataError(f"Duplicated tickers: {uni.loc[uni['ticker'].duplicated(), 'ticker'].tolist()}")
    bad_class = set(uni["asset_class"]) - ASSET_CLASSES
    if bad_class:
        raise DataError(f"Unknown asset classes {sorted(bad_class)}")
    return uni.set_index("ticker")


def load_returns(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Monthly total returns in CHF (month-end index, one column per ticker, NaN before inception)."""
    ret = _read_parquet("returns_chf.parquet", data_dir)
    unknown = set(ret.columns) - set(load_universe(data_dir).index)
    if unknown:
        raise DataError(f"Returns contain tickers absent from universe.csv: {sorted(unknown)}")
    return ret.sort_index()


def load_benchmark_returns(data_dir: Path = DATA_DIR) -> pd.Series:
    """Monthly total return of SPY converted to CHF (benchmark only, not investable)."""
    return _read_parquet("benchmarks_chf.parquet", data_dir)[BENCHMARK_TICKER].sort_index().rename("SPY (CHF)")


def load_rf(data_dir: Path = DATA_DIR) -> pd.Series:
    """Monthly CHF risk-free rate (SARON lagged one month, decimal per month)."""
    rf = _read_parquet("rf_chf.parquet", data_dir)["rf_monthly"].dropna()
    if rf.empty:
        raise DataError("rf_chf.parquet contains no risk-free observations.")
    return rf.rename("rf")


def constant_rf(index: pd.DatetimeIndex, annual_rate: float) -> pd.Series:
    """Fallback risk-free series: a user-set constant annual rate (decimal)."""
    return pd.Series(annual_rate / 12, index=index, name="rf")


def load_fx(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return _read_parquet("fx.parquet", data_dir).sort_index()
