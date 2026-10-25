"""Build data/universe.csv ONCE (the result is committed; the app never runs this).

Universe = all Nasdaq-100 companies + the largest other S&P 500 companies by market cap, up to
N_STOCKS US-listed stocks (one share class per company), 3 Swiss government bond ETFs and
synthetic CHF cash (SARON).

Sources (snapshot at build time):
- constituents: Wikipedia "List of NASDAQ-100 companies" and "List of S&P 500 companies";
- sectors: GICS sector from the S&P 500 list; for Nasdaq-100 members outside the S&P 500 the
  Yahoo sector is mapped to its GICS name;
- market caps: yfinance (only used to rank S&P 500 companies).
Using today's constituents to study the past creates SURVIVORSHIP BIAS (see CLAUDE.md).

Usage:  python scripts/build_universe.py
"""
from __future__ import annotations

import io
import sys
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
N_STOCKS = 150
NDX_URL = "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies"
SPX_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
DROP_SHARE_CLASSES = {"GOOG", "FOX", "NWS"}  # keep GOOGL, FOXA, NWSA
YAHOO_TO_GICS = {"Technology": "Information Technology", "Healthcare": "Health Care",
                 "Consumer Cyclical": "Consumer Discretionary", "Consumer Defensive": "Consumer Staples",
                 "Financial Services": "Financials", "Basic Materials": "Materials"}
NON_EQUITY = [
    ("CSBGC3.SW", "iShares Swiss Domestic Govt Bond 0-3y", "Bond", "Swiss Government Bonds", "CH", "CHF"),
    ("CSBGC7.SW", "iShares Swiss Domestic Govt Bond 3-7y", "Bond", "Swiss Government Bonds", "CH", "CHF"),
    ("CSBGC0.SW", "iShares Swiss Domestic Govt Bond 7-15y", "Bond", "Swiss Government Bonds", "CH", "CHF"),
    ("CASH_CHF", "Cash CHF (SARON)", "Cash", "Cash", "CH", "CHF"),
]


def wiki_table(url: str, column: str) -> pd.DataFrame:
    req = urllib.request.Request(url, headers={"User-Agent": "QARM-student-project/1.0"})
    html = urllib.request.urlopen(req, timeout=60).read().decode("utf-8")
    for table in pd.read_html(io.StringIO(html)):
        if column in table.columns and len(table) > 90:
            return table
    sys.exit(f"ERROR: no constituent table with column '{column}' at {url}")


def yahoo(symbol: str) -> str:
    return str(symbol).strip().replace(".", "-")


def market_cap(ticker: str) -> float:
    try:
        return float(yf.Ticker(ticker).fast_info["market_cap"] or 0)
    except Exception:  # missing quote: rank it last
        return 0.0


def main() -> None:
    ndx = wiki_table(NDX_URL, "Ticker")
    spx = wiki_table(SPX_URL, "Symbol")
    spx = pd.DataFrame({"ticker": spx["Symbol"].map(yahoo), "name": spx["Security"],
                        "sector": spx["GICS Sector"]}).set_index("ticker")
    ndx = pd.DataFrame({"ticker": ndx["Ticker"].map(yahoo), "name": ndx["Company"],
                        "isin_country": ndx["ISIN"].str[:2]}).set_index("ticker")
    ndx = ndx.drop(index=DROP_SHARE_CLASSES, errors="ignore")
    spx = spx.drop(index=DROP_SHARE_CLASSES, errors="ignore")

    ndx["sector"] = spx["sector"].reindex(ndx.index)
    for t in ndx.index[ndx["sector"].isna()]:
        raw = yf.Ticker(t).info.get("sector")
        if not raw:
            sys.exit(f"ERROR: no sector found for {t}")
        ndx.loc[t, "sector"] = YAHOO_TO_GICS.get(raw, raw)
    ndx["index"] = ["Nasdaq-100 & S&P 500" if t in spx.index else "Nasdaq-100" for t in ndx.index]

    others = spx.drop(index=ndx.index, errors="ignore").copy()
    print(f"Ranking {len(others)} S&P 500 companies by market cap ...")
    others["market_cap"] = [market_cap(t) for t in others.index]
    n_extra = N_STOCKS - len(ndx)
    extra = others.sort_values("market_cap", ascending=False).head(n_extra)
    extra = extra.assign(isin_country="US", index="S&P 500").drop(columns="market_cap")

    stocks = pd.concat([ndx, extra])
    if stocks.index.duplicated().any() or len(stocks) != N_STOCKS:
        sys.exit("ERROR: duplicated tickers or wrong universe size")
    stocks = stocks.assign(asset_class="Equity", listed_currency="USD").sort_values(["sector", "name"])
    rest = pd.DataFrame(NON_EQUITY, columns=["ticker", "name", "asset_class", "sector", "isin_country",
                                             "listed_currency"]).set_index("ticker").assign(index="")
    uni = pd.concat([stocks, rest]).reset_index()
    uni["data_start"] = ""
    cols = ["ticker", "name", "asset_class", "sector", "isin_country", "index", "listed_currency", "data_start"]
    uni[cols].to_csv(DATA_DIR / "universe.csv", index=False)
    (DATA_DIR / "universe_as_of.txt").write_text(f"Constituents snapshot: {date.today().isoformat()}\n")
    print(f"Saved {len(uni)} assets ({len(stocks)} stocks) to data/universe.csv")
    print(stocks["sector"].value_counts().to_string())


if __name__ == "__main__":
    main()
