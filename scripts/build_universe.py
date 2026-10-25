"""Build data/universe.csv ONCE (the result is committed; the app never runs this).

Universe = all Nasdaq-100 companies + the largest other S&P 500 companies by market cap, up to
N_STOCKS US-listed stocks (one share class per company), plus fixed ETFs (international
equities, USD bonds by type, gold) and synthetic USD cash (3-month T-bill). Everything is USD.

Bond tags (issuer_type, maturity_bucket, duration, wam, credit_quality, avg_rating, rating_score,
char_source, char_as_of) come from BOND_CHARACTERISTICS: current issuer fact-sheet values applied
to the whole history (simplification).

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.bonds import NOTCH_RATING, RATING_NOTCH  # noqa: E402  (one rating scale for the project)

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
ISH, VG = "https://www.ishares.com/us/products/", "https://www.vanguardsouthamerica.com/en/product/etf/fixed-income/"
# Bond fund characteristics copied from the issuers' product pages. CURRENT values, applied to the
# whole history (simplification: durations and ratings of the funds drift over time).
# ticker: (issuer_type, effective duration (y), weighted avg maturity (y), {rating: % of fund}, source, as of)
BOND_CHARACTERISTICS = {
    "BIL": ("Treasury", 0.10, 0.10, {"AAA": 0.06, "AA+": 99.94},
            "https://www.ssga.com/us/en/intermediary/etfs/spdr-bloomberg-1-3-month-t-bill-etf-bil", "2026-10-22"),
    "SHY": ("Treasury", 1.79, 1.89, {"AA": 99.70}, ISH + "239452", "2026-10-22"),
    "IEI": ("Treasury", 4.21, 4.71, {"AA": 99.69}, ISH + "239455", "2026-10-22"),
    "IEF": ("Treasury", 6.84, 8.41, {"AA": 99.99}, ISH + "239456", "2026-10-22"),
    "TLT": ("Treasury", 14.96, 26.03, {"AA": 100.0}, ISH + "239454", "2026-10-22"),
    "IGSB": ("Corporate", 2.64, 3.09, {"AAA": 0.21, "AA": 9.64, "A": 45.73, "BBB": 44.36, "BB": 0.05},
             ISH + "239451", "2026-10-22"),
    "IGIB": ("Corporate", 5.87, 7.56, {"AAA": 0.34, "AA": 7.49, "A": 43.30, "BBB": 48.33}, ISH + "239463", "2026-10-22"),
    "VCLT": ("Corporate", 11.6, 22.2, {"AAA": 1.52, "AA": 12.58, "A": 44.25, "BBB": 41.72, "BB": 0.03},
             VG + "3147/vanguard-long-term-corporate-bond-etf", "2026-09-30"),  # AA incl. 0.36% US government
    "HYG": ("Corporate", 3.17, 4.16, {"BBB": 0.51, "BB": 59.17, "B": 31.76, "CCC": 7.08, "CC": 0.31},
            ISH + "239565", "2026-10-22"),
    "MUB": ("Municipal", 7.03, 8.50, {"AAA": 20.64, "AA": 58.04, "A": 14.71, "BBB": 0.88}, ISH + "239766", "2026-10-22"),
    "MBB": ("Agency mortgage", 5.66, 7.56, {"AA": 98.72}, ISH + "239465", "2026-10-22"),
    "EMB": ("EM sovereign", 6.32, 10.48, {"A": 17.53, "BBB": 33.66, "BB": 25.35, "B": 19.07, "CCC": 3.30, "CC": 0.25,
                                          "D": 0.33}, ISH + "239572", "2026-10-22"),
    "BNDX": ("International", 6.4, 8.2, {"AAA": 25.71, "AA": 17.17, "A": 36.89, "BBB": 18.55},
             VG + "3711/vanguard-total-international-bond-etf", "2026-09-30"),
    "TIP": ("Treasury", 6.21, 7.04, {"AA": 99.71}, ISH + "239467", "2026-10-22"),
    "FLOT": ("Corporate", 0.02, 1.85, {"AAA": 22.07, "AA": 26.01, "A": 45.19, "BBB": 6.04}, ISH + "239534", "2026-10-22"),
}
HIGH_YIELD = {"HYG", "EMB"}  # EMB counts as high yield for the client's constraint (average BB+)
def maturity_bucket(wam: float) -> str:
    return "Ultra-short" if wam < 1 else "Short" if wam < 5 else "Intermediate" if wam < 10 else "Long"


def bond_tags(ticker: str) -> dict:
    issuer, duration, wam, ratings, source, as_of = BOND_CHARACTERISTICS[ticker]
    score = sum(RATING_NOTCH[r] * w for r, w in ratings.items()) / sum(ratings.values())
    return {"issuer_type": issuer, "maturity_bucket": maturity_bucket(wam), "duration": duration, "wam": wam,
            "credit_quality": "HY" if ticker in HIGH_YIELD else "IG", "rating_score": round(score, 2),
            "avg_rating": NOTCH_RATING[round(score)], "char_source": source, "char_as_of": as_of}


COLUMNS = ["ticker", "name", "asset_class", "sector", "vehicle", "is_EM", "bond_type", "issuer_type",
           "maturity_bucket", "duration", "wam", "credit_quality", "avg_rating", "rating_score", "char_source",
           "char_as_of", "isin_country",
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
    tags = pd.DataFrame({t: bond_tags(t) for t in fixed.index[fixed["asset_class"] == "Bond"]}).T
    if set(tags.index) != set(BOND_CHARACTERISTICS):
        sys.exit("ERROR: every bond ETF needs an entry in BOND_CHARACTERISTICS (and vice versa)")
    fixed = fixed.join(tags)
    uni = pd.concat([stocks, fixed]).reset_index().assign(listed_currency="USD", data_start="")
    uni[COLUMNS].to_csv(DATA_DIR / "universe.csv", index=False)
    (DATA_DIR / "universe_as_of.txt").write_text(f"Constituents snapshot: {date.today().isoformat()}\n")
    print(f"Saved {len(uni)} assets ({len(stocks)} stocks) to data/universe.csv")
    print(stocks["sector"].value_counts().to_string())


if __name__ == "__main__":
    main()
