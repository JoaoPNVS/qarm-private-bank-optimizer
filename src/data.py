"""Load parquet data, CHF conversion and universe tags.

All files are produced once by scripts/download_data.py; nothing here touches the network.
Every loader raises DataError with a message that the app can show to the user as-is.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CASH_TICKER = "CASH_CHF"
UNIVERSE_COLUMNS = ["ticker", "name", "asset_class", "region", "currency", "listed_currency",
                    "is_EM", "is_swiss", "sector", "data_start"]
ASSET_CLASSES = {"Equity", "Bond", "Alternative", "Real Estate", "Cash"}


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
    """Universe with tags, indexed by ticker. `non_chf` flags any foreign-currency exposure
    (EM-FX and Multi included), which is what the FX constraint counts."""
    path = data_dir / "universe.csv"
    if not path.exists():
        raise DataError("data/universe.csv not found.")
    uni = pd.read_csv(path, parse_dates=["data_start"])
    missing = set(UNIVERSE_COLUMNS) - set(uni.columns)
    if missing:
        raise DataError(f"universe.csv is missing columns {sorted(missing)}")
    if uni["ticker"].duplicated().any():
        raise DataError(f"Duplicated tickers: {uni.loc[uni['ticker'].duplicated(), 'ticker'].tolist()}")
    bad_class = set(uni["asset_class"]) - ASSET_CLASSES
    if bad_class:
        raise DataError(f"Unknown asset classes {sorted(bad_class)}")
    for col in ["is_EM", "is_swiss"]:
        if uni[col].dtype != bool:
            raise DataError(f"Column '{col}' must contain only True/False")
    uni["non_chf"] = uni["currency"] != "CHF"
    return uni.set_index("ticker")


def load_returns(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Monthly total returns in CHF (month-end index, one column per ticker, NaN before inception)."""
    ret = _read_parquet("returns_chf.parquet", data_dir)
    unknown = set(ret.columns) - set(load_universe(data_dir).index)
    if unknown:
        raise DataError(f"Returns contain tickers absent from universe.csv: {sorted(unknown)}")
    return ret.sort_index()


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


def common_start(returns: pd.DataFrame, tickers: list[str]) -> pd.Timestamp:
    """First month from which every selected asset has an uninterrupted return history."""
    gaps = returns[tickers].isna().any(axis=1)
    if gaps.all():
        raise DataError("The selected assets never have returns available at the same time.")
    if not gaps.any():
        return returns.index[0]
    return returns.index[returns.index > gaps[gaps].index.max()][0]
