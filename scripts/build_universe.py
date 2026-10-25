"""Build data/universe.csv ONCE (the result is committed; the app never runs this).

Universe = all Nasdaq-100 companies + the largest other S&P 500 companies by market cap, up to
N_STOCKS US-listed stocks (one share class per company), plus fixed ETFs (international
equities, USD bonds by type, gold) and synthetic USD cash (3-month T-bill). Everything is USD.

Tags: `vehicle` (Stock / ETF / Cash: 8% cap per stock, 20% cap per ETF), `is_EM` (EEM and the
emerging-market companies PDD and MELI count toward the EM <= 1/3 of equities rule; other
foreign companies listed in the US are developed-market), `bond_type` for bond ETFs.
International equity ETFs share the sector "International" (capped like any equity sector);
bond and gold ETFs have their own sectors and are not subject to the sector cap.

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
EM_COMPANIES = {"PDD", "MELI"}  # China, Latin America
# ticker, name, asset_class, sector, vehicle, is_EM, bond_type
FIXED_ASSETS = [
    ("VGK", "Vanguard FTSE Europe ETF", "Equity", "International", "ETF", False, ""),
    ("EWJ", "iShares MSCI Japan ETF", "Equity", "International", "ETF", False, ""),
    ("EEM", "iShares MSCI Emerging Markets ETF", "Equity", "International", "ETF", True, ""),
    # USD bond ETFs as market segments (bond_type). IGSB / IGIB replace VCSH / VCIT: same segments,
    # history from 2007-01 instead of 2009-11. VCLT, BNDX and FLOT have no older equivalent.
    ("BIL", "SPDR Bloomberg 1-3 Month T-Bill ETF", "Bond", "Bonds", "ETF", False, "Treasury"),
    ("SHY", "iShares 1-3 Year Treasury Bond ETF", "Bond", "Bonds", "ETF", False, "Treasury"),
    ("IEI", "iShares 3-7 Year Treasury Bond ETF", "Bond", "Bonds", "ETF", False, "Treasury"),
    ("IEF", "iShares 7-10 Year Treasury Bond ETF", "Bond", "Bonds", "ETF", False, "Treasury"),
    ("TLT", "iShares 20+ Year Treasury Bond ETF", "Bond", "Bonds", "ETF", False, "Treasury"),
    ("IGSB", "iShares 1-5 Year Investment Grade Corporate Bond ETF", "Bond", "Bonds", "ETF", False, "Corporate IG"),
    ("IGIB", "iShares 5-10 Year Investment Grade Corporate Bond ETF", "Bond", "Bonds", "ETF", False, "Corporate IG"),
    ("VCLT", "Vanguard Long-Term Corporate Bond ETF", "Bond", "Bonds", "ETF", False, "Corporate IG"),
    ("HYG", "iShares iBoxx $ High Yield Corporate Bond ETF", "Bond", "Bonds", "ETF", False, "High Yield"),
    ("MUB", "iShares National Muni Bond ETF", "Bond", "Bonds", "ETF", False, "Municipal"),
    ("MBB", "iShares MBS ETF", "Bond", "Bonds", "ETF", False, "Mortgage-backed"),
    ("EMB", "iShares J.P. Morgan USD Emerging Markets Bond ETF", "Bond", "Bonds", "ETF", False, "EM USD"),
    ("BNDX", "Vanguard Total International Bond ETF (USD hedged)", "Bond", "Bonds", "ETF", False, "International hedged"),
    ("TIP", "iShares TIPS Bond ETF", "Bond", "Bonds", "ETF", False, "TIPS"),
    ("FLOT", "iShares Floating Rate Bond ETF", "Bond", "Bonds", "ETF", False, "Floating rate"),
    ("GLD", "SPDR Gold Shares", "Commodity", "Gold", "ETF", False, ""),
    ("CASH_USD", "Cash USD (3-month T-bill)", "Cash", "Cash", "Cash", False, ""),
]
COLUMNS = ["ticker", "name", "asset_class", "sector", "vehicle", "is_EM", "bond_type", "isin_country",
           "index", "listed_currency", "data_start"]


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
    stocks = stocks.assign(asset_class="Equity", vehicle="Stock", bond_type="",
                           is_EM=stocks.index.isin(EM_COMPANIES)).sort_values(["sector", "name"])
    fixed = pd.DataFrame(FIXED_ASSETS, columns=COLUMNS[:7]).set_index("ticker").assign(isin_country="US", index="")
    uni = pd.concat([stocks, fixed]).reset_index().assign(listed_currency="USD", data_start="")
    uni[COLUMNS].to_csv(DATA_DIR / "universe.csv", index=False)
    (DATA_DIR / "universe_as_of.txt").write_text(f"Constituents snapshot: {date.today().isoformat()}\n")
    print(f"Saved {len(uni)} assets ({len(stocks)} stocks) to data/universe.csv")
    print(stocks["sector"].value_counts().to_string())


if __name__ == "__main__":
    main()
