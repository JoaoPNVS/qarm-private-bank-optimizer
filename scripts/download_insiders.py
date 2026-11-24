"""Download the SEC insider-trading data ONCE and save small derived files to data/ (committed).

Source: SEC "Insider Transactions Data Sets", the XML part of every Form 3, 4 and 5, one zip per
quarter from 2006Q1: https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets
The Streamlit app never calls the network: it only reads the files written here.

The SEC refuses automated requests that do not identify their sender. Set a contact for the
session before running (it is sent to sec.gov only and never written anywhere):
    SEC_USER_AGENT="Your Name your.email@example.com" python scripts/download_insiders.py

Raw zips (about 10 MB per quarter, 0.8 GB in all) are cached OUTSIDE the repo (default
~/.cache/qarm-insiders, option --cache). Each quarter is reduced to a small extract of its
purchase and sale filings; the zip is then deleted (--keep-zips keeps it). A quarter already
extracted is not downloaded again.

What is kept (the signals built on it are defined in src/insiders.py):
- Form 4 only: due within two business days of a trade. Amendments (4/A) and annual Forms 5 are
  left out. Every filing is dated by its FILING date, the day the trade became public.
- Open-market purchases (transaction code P) and sales (S). Grants (A), option exercises (M), tax
  withholding (F), gifts (G), ... are not decisions to buy or sell and are left out.
- Officers and directors. A filer who is only a 10% holder (often a fund) is left out.
- Listed issuers only (a trading symbol other than NONE / NA).

Outputs:
- data/insider_market.parquet   monthly, all US issuers: number of filings and of companies with
                                insider purchases and with insider sales, dollar amounts.
- data/insider_trades.parquet   every purchase or sale filing by an officer or director of one of
                                our stocks: filing date, ticker, insider, role, shares, value.
- data/insider_metadata.json    source, quarters covered, download date, issuer CIKs per ticker.

Usage:  SEC_USER_AGENT="Name email" python scripts/download_insiders.py [--cache DIR] [--keep-zips]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SOURCE_PAGE = "https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets"
# the SEC moved the newest files (2026Q2 on) to a second folder; both are tried
ZIP_URLS = ["https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{q}_form345.zip",
            "https://www.sec.gov/files/datastandardsinnovation/data/insider-transactions-data-sets/{q}_form345.zip"]
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FIRST_QUARTER = "2006Q1"   # first quarterly data set published by the SEC
PAUSE_S = 0.25             # SEC fair-access limit: at most 10 requests per second
NOT_LISTED = {"", "NONE", "NA", "N/A", "NULL"}

# Companies whose SEC registrant (CIK) changed with a reorganisation: their insiders filed under the
# predecessor's CIK before. Found by searching the filings for the trading symbol and the company
# name, checked by hand (a symbol alone can mislead: BKR was Michael Baker Corp until 2010).
# Plain renamings under the same CIK (Facebook -> Meta, Priceline -> Booking, ...) need nothing.
FORMER_CIKS: dict[str, dict[int, str]] = {
    "GOOGL": {1288776: "Google Inc., before the Alphabet holding (Oct 2015)"},
    "DIS": {1001039: "The Walt Disney Company before the 21st Century Fox deal (Mar 2019)"},
    "BLK": {1364742: "BlackRock Inc. before the 2024 holding reorganisation",
            1060021: "BlackRock Inc. before the 2006 Merrill Lynch asset-management deal"},
    "XOM": {34088: "Exxon Mobil Corp (the new 2026 registrant had no Form 4 yet)"},
    "ETN": {31277: "Eaton Corp before Eaton Corp plc (Dec 2012)"},
    "AVGO": {1441634: "Avago Technologies (2009-2016)", 1649338: "Broadcom Ltd (2016-2018)"},
    "LIN": {884905: "Praxair (LIN prices before the 2018 Linde merger are Praxair's)"},
    "BKR": {808362: "Baker Hughes Inc (BHI) before the 2017 GE Oil & Gas deal"},
    "ACN": {1134538: "Accenture Ltd (Bermuda) before the 2009 move to Ireland"},
    "MRVL": {1058057: "Marvell Technology Group Ltd (Bermuda) before the 2021 move to Delaware"},
    "MRK": {64978: "Merck & Co before the 2009 Schering-Plough merger"},
}
# A current CIK that belonged to ANOTHER company before a date: {ticker: (CIK, first valid date)}.
# Merck's 2009 merger kept Schering-Plough's registrant, renamed Merck & Co.
CIK_VALID_FROM = {"MRK": (310158, "2009-11-04")}

SUB_COLS = ["ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE", "ISSUERCIK", "ISSUERNAME",
            "ISSUERTRADINGSYMBOL", "AFF10B5ONE"]
OWN_COLS = ["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNERNAME", "RPTOWNER_RELATIONSHIP", "RPTOWNER_TITLE"]
TRANS_COLS = ["ACCESSION_NUMBER", "TRANS_DATE", "TRANS_CODE", "TRANS_SHARES", "TRANS_PRICEPERSHARE"]


# --- download --------------------------------------------------------------------------------

def user_agent() -> str:
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if "@" not in ua:
        sys.exit('ERROR: set SEC_USER_AGENT="Your Name your.email@example.com" (the SEC requires a contact).')
    return ua


def fetch(url: str, ua: str, retries: int = 4) -> bytes | None:
    """GET with the SEC contact header; None on 404 (data set not published)."""
    for attempt in range(retries):
        req = urllib.request.Request(url, headers={"User-Agent": ua, "Accept-Encoding": "identity"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = resp.read()
            time.sleep(PAUSE_S)
            return body
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            if exc.code == 403:
                sys.exit("ERROR: the SEC refused the request (403). Check SEC_USER_AGENT (name and e-mail).")
            wait = 2 ** attempt
        except (urllib.error.URLError, TimeoutError):
            wait = 2 ** attempt
        print(f"  retry in {wait}s: {url}")
        time.sleep(wait)
    sys.exit(f"ERROR: could not download {url}")


def quarters(first: str, last: pd.Period) -> list[str]:
    return [f"{p.year}q{p.quarter}" for p in pd.period_range(pd.Period(first, "Q"), last, freq="Q")]


# --- extraction ------------------------------------------------------------------------------

def _read(z: zipfile.ZipFile, name: str, cols: list[str]) -> pd.DataFrame:
    member = next((n for n in z.namelist() if n.upper().endswith(name)), None)
    if member is None:
        raise ValueError(f"{name} missing in {z.filename}")
    with z.open(member) as fh:
        return pd.read_csv(fh, sep="\t", dtype=str, usecols=lambda c: c in cols, quoting=csv.QUOTE_NONE,
                           on_bad_lines="skip", encoding="utf-8", encoding_errors="replace",
                           keep_default_na=False, na_values=[""])


def _date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, format="%d-%b-%Y", errors="coerce")


def extract_quarter(zip_path: Path) -> pd.DataFrame:
    """One row per Form 4 filing that reports an open-market purchase or sale (any filer)."""
    with zipfile.ZipFile(zip_path) as z:
        sub = _read(z, "SUBMISSION.TSV", SUB_COLS)
        own = _read(z, "REPORTINGOWNER.TSV", OWN_COLS)
        tr = _read(z, "NONDERIV_TRANS.TSV", TRANS_COLS)
    sub = sub[sub["DOCUMENT_TYPE"] == "4"]
    tr = tr[tr["TRANS_CODE"].isin(["P", "S"]) & tr["ACCESSION_NUMBER"].isin(sub["ACCESSION_NUMBER"])].copy()
    tr["shares"] = pd.to_numeric(tr["TRANS_SHARES"], errors="coerce")
    tr["value"] = tr["shares"] * pd.to_numeric(tr["TRANS_PRICEPERSHARE"], errors="coerce")
    tr["trade_date"] = _date(tr["TRANS_DATE"])
    agg = tr.groupby(["ACCESSION_NUMBER", "TRANS_CODE"]).agg(
        lines=("shares", "size"), shares=("shares", "sum"), value=("value", "sum")).unstack("TRANS_CODE")
    agg.columns = [f"{field}_{code}" for field, code in agg.columns]
    for col in [f"{f}_{c}" for f in ("lines", "shares", "value") for c in ("P", "S")]:
        agg[col] = agg[col].fillna(0.0) if col in agg else 0.0
    agg["trade_date"] = tr.groupby("ACCESSION_NUMBER")["trade_date"].min()

    rel = own["RPTOWNER_RELATIONSHIP"].fillna("")
    own = own.assign(director=rel.str.contains("Director"), officer=rel.str.contains("Officer"),
                     ten_pct=rel.str.contains("TenPercentOwner"))
    owners = own.groupby("ACCESSION_NUMBER").agg(
        director=("director", "any"), officer=("officer", "any"), ten_pct=("ten_pct", "any"),
        owner_cik=("RPTOWNERCIK", "first"), owner_name=("RPTOWNERNAME", "first"),
        owner_title=("RPTOWNER_TITLE", "first"), n_owners=("RPTOWNERCIK", "size"))

    out = (sub.set_index("ACCESSION_NUMBER").join(agg, how="inner").join(owners, how="left")
           .reset_index().rename(columns={"ACCESSION_NUMBER": "accession", "ISSUERNAME": "issuer_name",
                                          "ISSUERTRADINGSYMBOL": "symbol"}))
    out["filing_date"] = _date(out["FILING_DATE"])
    out["issuer_cik"] = pd.to_numeric(out["ISSUERCIK"], errors="coerce").astype("Int64")
    plan = out["AFF10B5ONE"].str.lower() if "AFF10B5ONE" in out else pd.Series(pd.NA, index=out.index)
    out["plan_10b5_1"] = plan.map({"1": True, "true": True, "0": False, "false": False}).astype("boolean")
    for col in ("director", "officer", "ten_pct"):
        out[col] = out[col].fillna(False).astype(bool)
    keep = ["accession", "filing_date", "trade_date", "issuer_cik", "issuer_name", "symbol", "owner_cik",
            "owner_name", "owner_title", "director", "officer", "ten_pct", "n_owners", "plan_10b5_1",
            "lines_P", "lines_S", "shares_P", "shares_S", "value_P", "value_S"]
    return out[keep].dropna(subset=["filing_date", "issuer_cik"])


def load_extracts(cache: Path, ua: str, keep_zips: bool) -> tuple[pd.DataFrame, list[str]]:
    (cache / "extracts").mkdir(parents=True, exist_ok=True)
    (cache / "zips").mkdir(parents=True, exist_ok=True)
    wanted = quarters(FIRST_QUARTER, pd.Timestamp.today().to_period("Q") - 1)
    frames, done = [], []
    for i, q in enumerate(wanted):
        ext = cache / "extracts" / f"{q}.parquet"
        if not ext.exists():
            zpath = cache / "zips" / f"{q}.zip"
            if not zpath.exists():
                body = next((b for url in ZIP_URLS if (b := fetch(url.format(q=q), ua)) is not None), None)
                if body is None:
                    if i >= len(wanted) - 2:   # the latest quarter can be published with a delay
                        print(f"{q}: not published yet, skipped")
                        continue
                    sys.exit(f"ERROR: data set {q} not found at the SEC")
                zpath.write_bytes(body)
            extract_quarter(zpath).to_parquet(ext, index=False)
            if not keep_zips:
                zpath.unlink()
        frames.append(pd.read_parquet(ext))
        done.append(q)
        print(f"{q}: {len(frames[-1]):,} purchase/sale filings")
    return pd.concat(frames, ignore_index=True), done


# --- outputs ---------------------------------------------------------------------------------

def insider_filings(raw: pd.DataFrame) -> pd.DataFrame:
    """Officer/director filings of listed issuers, with a net direction (buy or sell)."""
    f = raw[(raw["director"] | raw["officer"]) & ~raw["symbol"].fillna("").str.upper().isin(NOT_LISTED)].copy()
    only_buy = (f["lines_P"] > 0) & (f["lines_S"] == 0)
    only_sell = (f["lines_S"] > 0) & (f["lines_P"] == 0)
    f["direction"] = np.where(only_buy, "buy", np.where(only_sell, "sell",
                              np.where(f["value_P"] >= f["value_S"], "buy", "sell")))
    f["month"] = f["filing_date"] + pd.offsets.MonthEnd(0)
    return f


def market_monthly(f: pd.DataFrame) -> pd.DataFrame:
    buys, sells = f[f["direction"] == "buy"], f[f["direction"] == "sell"]
    out = pd.DataFrame({
        "buy_filings": buys.groupby("month").size(),
        "sell_filings": sells.groupby("month").size(),
        "buy_companies": buys.groupby("month")["issuer_cik"].nunique(),
        "sell_companies": sells.groupby("month")["issuer_cik"].nunique(),
        "buy_value_musd": buys.groupby("month")["value_P"].sum() / 1e6,
        "sell_value_musd": sells.groupby("month")["value_S"].sum() / 1e6,
    }).fillna(0.0)
    out.index.name = "month"
    return out.astype({c: int for c in ("buy_filings", "sell_filings", "buy_companies", "sell_companies")})


def cik_map(tickers: list[str], ua: str) -> tuple[dict[str, list[int]], list[str]]:
    """Current SEC registrant of each ticker (company_tickers.json), plus FORMER_CIKS."""
    body = fetch(TICKERS_URL, ua)
    if body is None:
        sys.exit("ERROR: company_tickers.json not found at the SEC")
    sec = {row["ticker"].upper(): int(row["cik_str"]) for row in json.loads(body).values()}
    out, missing = {}, []
    for t in tickers:
        cik = next((sec[k] for k in (t, t.replace("-", "."), t.replace(".", "-")) if k in sec), None)
        if cik is None:
            missing.append(t)
            continue
        out[t] = [cik] + list(FORMER_CIKS.get(t, {}))
    return out, missing


def role(row) -> str:
    title = row["owner_title"] if isinstance(row["owner_title"], str) and row["owner_title"].strip() else ""
    if row["officer"]:
        base = f"Officer ({title})" if title else "Officer"
        return f"Director & {base}" if row["director"] else base
    return "Director"


def universe_trades(f: pd.DataFrame, ciks: dict[str, list[int]]) -> pd.DataFrame:
    to_ticker = {c: t for t, cs in ciks.items() for c in cs}
    u = f[f["issuer_cik"].isin(to_ticker)].copy()
    u["ticker"] = u["issuer_cik"].map(to_ticker)
    for ticker, (cik, start) in CIK_VALID_FROM.items():
        u = u[~((u["ticker"] == ticker) & (u["issuer_cik"] == cik) & (u["filing_date"] < pd.Timestamp(start)))]
    buy = u["direction"] == "buy"
    u["shares"] = np.where(buy, u["shares_P"], u["shares_S"])
    u["value_usd"] = np.where(buy, u["value_P"], u["value_S"])
    u["role"] = u.apply(role, axis=1)
    cols = ["filing_date", "trade_date", "ticker", "issuer_name", "owner_name", "role", "direction",
            "shares", "value_usd", "plan_10b5_1", "accession"]
    return u[cols].rename(columns={"owner_name": "insider"}).sort_values(["filing_date", "ticker"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "qarm-insiders")
    parser.add_argument("--keep-zips", action="store_true")
    args = parser.parse_args()
    ua = user_agent()

    universe = pd.read_csv(DATA_DIR / "universe.csv", keep_default_na=False)
    stocks = universe.loc[universe["vehicle"] == "Stock", "ticker"].tolist()
    raw, done = load_extracts(args.cache, ua, args.keep_zips)
    filings = insider_filings(raw)
    market = market_monthly(filings)
    ciks, missing = cik_map(stocks, ua)
    if missing:
        print(f"WARNING: no SEC registrant found for {missing}")
    trades = universe_trades(filings, ciks)

    market.to_parquet(DATA_DIR / "insider_market.parquet")
    trades.to_parquet(DATA_DIR / "insider_trades.parquet", index=False)
    meta = {
        "source": SOURCE_PAGE,
        "quarters": [done[0], done[-1]],
        "first_filing": f"{filings['filing_date'].min():%Y-%m-%d}",
        "last_filing": f"{filings['filing_date'].max():%Y-%m-%d}",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "filters": "Form 4, open-market purchases (P) and sales (S), officers and directors, listed issuers",
        "insider_filings": int(len(filings)),
        "universe_filings": int(len(trades)),
        "issuer_ciks": ciks,
        "former_ciks": {t: {str(c): name for c, name in d.items()} for t, d in FORMER_CIKS.items()},
        "no_registrant": missing,
        # foreign private issuers (ASML, Shopify, ...) are exempt from Form 4: no insider data
        "no_filings": sorted(set(ciks) - set(trades["ticker"])),
    }
    (DATA_DIR / "insider_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"\n{len(filings):,} insider filings {meta['first_filing']} -> {meta['last_filing']}; "
          f"{len(trades):,} in our {len(ciks)} stocks ({(trades['direction'] == 'buy').sum():,} purchases).")


if __name__ == "__main__":
    main()
