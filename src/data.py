"""Load parquet data and universe tags (everything in USD, no currency conversion).

All files are produced once by scripts/build_universe.py and scripts/download_data.py; nothing
here touches the network. Every loader raises DataError with a message that the app can show
to the user as-is.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CASH_TICKER = "CASH_USD"
BENCHMARK_TICKERS = ["SPY", "AGG"]
UNIVERSE_COLUMNS = ["ticker", "name", "asset_class", "sector", "vehicle", "is_EM", "bond_type", "issuer_type",
                    "maturity_bucket", "duration", "wam", "credit_quality", "avg_rating", "rating_score",
                    "char_source", "char_as_of", "isin_country", "index", "listed_currency", "data_start"]
NUMERIC_COLUMNS = ["duration", "wam", "rating_score"]  # bond funds only, NaN elsewhere
VEHICLES = {"Stock", "ETF", "Cash"}
ASSET_CLASSES = {"Equity", "Bond", "Commodity", "Cash"}


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
    """Investable universe, indexed by ticker."""
    path = data_dir / "universe.csv"
    if not path.exists():
        raise DataError("data/universe.csv not found. Run `python scripts/build_universe.py`.")
    uni = pd.read_csv(path, parse_dates=["data_start"], keep_default_na=False,
                      na_values={c: [""] for c in ["data_start"] + NUMERIC_COLUMNS})
    missing = set(UNIVERSE_COLUMNS) - set(uni.columns)
    if missing:
        raise DataError(f"universe.csv is missing columns {sorted(missing)}")
    if uni["ticker"].duplicated().any():
        raise DataError(f"Duplicated tickers: {uni.loc[uni['ticker'].duplicated(), 'ticker'].tolist()}")
    bad_class = set(uni["asset_class"]) - ASSET_CLASSES
    if bad_class:
        raise DataError(f"Unknown asset classes {sorted(bad_class)}")
    if uni["is_EM"].dtype != bool:
        raise DataError("Column 'is_EM' must contain only True/False.")
    if set(uni["vehicle"]) - VEHICLES:
        raise DataError(f"Unknown vehicles {sorted(set(uni['vehicle']) - VEHICLES)}")
    bonds = uni["asset_class"] == "Bond"
    if uni.loc[bonds, NUMERIC_COLUMNS].isna().any().any() or not uni.loc[bonds, "credit_quality"].isin(["IG", "HY"]).all():
        raise DataError("Every bond fund needs duration, maturity, rating and IG/HY tags in universe.csv.")
    if (uni["listed_currency"] != "USD").any():
        raise DataError("Every asset must be USD-listed (no currency conversion in this project).")
    return uni.set_index("ticker")


def load_returns(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Monthly total returns in USD (month-end index, one column per ticker, NaN before inception)."""
    ret = _read_parquet("returns.parquet", data_dir)
    unknown = set(ret.columns) - set(load_universe(data_dir).index)
    if unknown:
        raise DataError(f"Returns contain tickers absent from universe.csv: {sorted(unknown)}")
    return ret.sort_index()


def load_benchmark_returns(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Monthly total returns of the benchmark series SPY and AGG (not investable)."""
    return _read_parquet("benchmarks.parquet", data_dir)[BENCHMARK_TICKERS].sort_index()


def load_rf(data_dir: Path = DATA_DIR) -> pd.Series:
    """Monthly risk-free rate: 3-month T-bill (^IRX) lagged one month, decimal per month."""
    rf = _read_parquet("rf.parquet", data_dir)["rf_monthly"].dropna()
    if rf.empty:
        raise DataError("rf.parquet contains no risk-free observations.")
    return rf.rename("rf")
