# Private Client Portfolio Optimizer — QARM II project

## What this project is
A Streamlit web app for the HEC Lausanne course *Quantitative Asset and Risk Management II*.
We act as an **international private bank**. Our clients are international private clients
whose reference currency is the **USD**. The app profiles the client, builds the portfolio with
the highest expected return within the client's risk budget and constraints, backtests the
strategy and simulates the client's wealth.

Grading criteria (from the professor) — every feature should serve one of these:
1. Content: complexity of client & data, strategy covers the client's needs, uses course tools
   (quant optimization, covariance estimation, risk measures).
2. Presentation: results must be clear to a "client" in a 15-min pitch; we must be able to answer
   live questions (so the app must make "what if" changes easy).
3. Web app: design, interactivity, completeness, no technical issues (must never crash).

## Currency
- Everything is in **USD**: prices, returns, risk-free rate, benchmarks, client wealth.
- **No FX conversion anywhere** in the code or the data pipeline. All assets are USD-listed.

## The client
- Suitability questionnaire (`src/profiling.py`): 7 questions, each answer scored 1-4, on three
  dimensions — risk CAPACITY (horizon, income stability, share of wealth), risk WILLINGNESS
  (objective, tolerable loss, reaction to a drawdown) and KNOWLEDGE & experience. No reference
  to any national regulation.
- Mapping (explainable to the client): capacity and willingness averages map to a level
  (<1.75 Conservative, <2.50 Balanced, <3.25 Growth, otherwise Dynamic); the profile is the
  LOWER of the two; knowledge caps it ("none" -> at most Balanced, "basic" -> at most Growth);
  a horizon below 3 years always gives Conservative.
- Each profile sets a **volatility cap** (risk budget) and a **floor on bond funds** (cash does
  not count):

  | Profile      | Vol cap | Bond floor         | Benchmark (SPY / USD bonds) |
  |--------------|--------:|-------------------:|-----------------------------|
  | Conservative |      6% |                70% | 30 / 70                     |
  | Balanced     |     10% |                40% | 60 / 40                     |
  | Growth       |     12% |                20% | 80 / 20                     |
  | Dynamic      |     15% |                 0% | 100 / 0                     |

  Calibration rule for the vol caps: historical volatility of the profile benchmark rounded UP
  to the next percent, so the optimised portfolio takes the same risk as its benchmark.
  Measured on the USD benchmarks (2003-10 to 2026-09): vols 5.91 / 9.28 / 11.85 / 14.52%, hence
  the caps 6 / 10 / 12 / 15% (team decision; the rule is tested in `tests/test_benchmarks.py`).
  The bond floor equals the benchmark's bond share. It applies to bond funds only: with cash
  counted, the 2021-2026 window (bonds lost money against T-bills) filled every floor with cash.

## Client constraints (`src/optimizer.py`; simple, explainable, all editable in the app)
- Maximum volatility = the profile's risk budget (mandatory).
- Minimum share in bond funds = the profile's floor (active by default, can be changed).
- Maximum cash & equivalents: 10% (CASH_USD + BIL + FLOT; active by default, can be changed).
- Max weight per single stock: 8%; per bond ETF: 15%; per other ETF (equity funds, gold): 20%.
- Bond sleeve limits (defaults by profile Conservative / Balanced / Growth / Dynamic):
  average effective duration of the bonds at most 5 / 6 / 7 / 8 years (linear form
  sum(w_b D_b) <= D sum(w_b)); investment grade at least 90 / 80 / 70 / 60% of the bonds;
  high yield (HYG + EMB) at most 5 / 10 / 15 / 20% of the portfolio.
- Max weight per equity sector: 40%.
- Emerging markets ≤ 1/3 of the equity allocation (linear form: EM ≤ (1/3)·equities).
- Exclusions of sectors or individual companies.
- Infeasibility diagnosis: the optimiser never crashes; a deletion filter names the minimal set
  of conflicting constraints and the ones whose removal alone restores feasibility.

## Data (yfinance only; Wikipedia for the constituent snapshot)
- **Universe** (`data/universe.csv`, one file, each asset tagged with asset_class, sector,
  is_EM, bond_type):
  - the 150 current US stocks (all Nasdaq-100 members + the largest other S&P 500 members,
    one share class per company, GICS sector), built once by `scripts/build_universe.py`
    (snapshot date in `data/universe_as_of.txt`);
  - international equity ETFs: Europe, Japan, emerging markets (proposed VGK, EWJ, EEM);
  - 15 USD bond ETFs used as market segments (`bond_type`; individual bonds have no free
    prices): Treasury by maturity BIL (1-3 months), SHY (1-3y), IEI (3-7y),
    IEF (7-10y), TLT (20y+); corporate investment grade by maturity IGSB (1-5y), IGIB (5-10y),
    VCLT (10y+); high yield HYG; municipal MUB; agency mortgage-backed MBB; emerging-market
    USD sovereigns EMB; international USD-hedged BNDX; TIPS (TIP); floating rate FLOT.
    IGSB/IGIB replace the requested VCSH/VCIT (same segments, history from 2007-02 instead of
    2009-12). VCLT, BNDX and FLOT have no older equivalent and become eligible only in 2014-11,
    2018-06 and 2016-06. **BIL and FLOT (duration ~0) are cash equivalents:** asset_class Cash,
    sector "Cash equivalents"; they count in the 10% cash cap, not in the bond floor, and are
    shown under "Cash & equivalents". The bond sleeve is the other 13 funds. The floor stays
    feasible for every profile at every rebalancing from 2012 (tested);
  - bond tags (columns of universe.csv, from `BOND_CHARACTERISTICS` in `scripts/build_universe.py`):
    `issuer_type` (Treasury, Corporate, Municipal, Agency mortgage, EM sovereign, International),
    `maturity_bucket` from the weighted average maturity (<1y Ultra-short, 1-5y Short, 5-10y
    Intermediate, >=10y Long), `duration` (effective duration, years), `wam`, `credit_quality`
    (IG / HY; HYG and EMB are HY, EMB averages BB+), `avg_rating` / `rating_score` (weighted
    average of the fund's rating breakdown on a notch scale AAA=1 ... D=22). **Source and date in
    `char_source` / `char_as_of`:** iShares and State Street (BIL) product pages as of 2026-10-22,
    Vanguard product pages (VCLT, BNDX) as of 2026-09-30. Yahoo's fund data was rejected (its
    durations are wrong, e.g. TLT 3.6y instead of ~15y). **Simplification: these CURRENT values
    are applied to the whole backtest history**, although fund durations and ratings drift;
  - gold (GLD);
  - CASH_USD, synthetic, earning the 3-month T-bill.
- **Risk-free**: 3-month T-bill yield `^IRX` (yfinance). Monthly rf_t = IRX_{t-1} / 1200 (rate
  known at the start of the month). CASH_USD earns rf: zero excess return, zero variance.
- **Benchmarks** (`src/benchmarks.py`, not investable): SPY, and one benchmark per profile =
  SPY / US aggregate bonds (proposed AGG) rebalanced monthly, shares as in the table above.
- Prices: yfinance daily adjusted close (dividends reinvested) -> month-end. Only complete
  months are kept.
- **Eligibility**: an asset enters the optimisation only once it has a complete 60-month
  history (recent IPOs/spin-offs and young ETFs wait).
- Data checks in `scripts/download_data.py`: abort on a monthly move above +200% (bad tick),
  print moves above 100% for manual review; `data_start` in universe.csv can cut corrupted
  early history.
- **Survivorship bias (important limitation).** The stock universe is TODAY's index membership
  applied to the past. Companies that went bankrupt, were acquired, shrank or left the indices
  (Lehman, Yahoo, Twitter, ...) are missing, and today's members are by construction the ones
  that grew. Any backtest on this universe overstates what a real investor could have earned,
  and the optimiser benefits most because it overweights past winners. SPY is free of this
  bias, so outperformance versus SPY must be read as an UPPER bound. A clean study would need
  point-in-time constituents (e.g. CRSP), which are not freely available.
- **Data is downloaded once and saved to `data/*.parquet`, which are committed to the repo.**
  The Streamlit app reads parquet only — it must never call the network at runtime.

## Methodology
- **Objective: maximise expected return under the profile's volatility cap.** Long-only, fully
  invested: max w'mu s.t. sqrt(w'Sigma w) <= sigma_max(profile) and the client constraints
  (+ L1 transaction-cost penalty when current holdings are given, e.g. in the backtest).
- Expected returns: historical mean over the 60-month window, shrunk toward the grand mean of
  the asset's shrinkage group: mu_i = (1 - a) m_i + a m_group, **a = 0.8**. Groups = asset
  classes, except **bonds, grouped by maturity bucket** (Short / Intermediate / Long): with 15
  segments, the whole bond class mean (-3.3% on 2021-2026, driven by long bonds in 2022) pulled
  BIL to -2.7% although its real excess return is ~0. Cash equivalents shrink toward the cash
  group (CASH_USD, BIL, FLOT). Raw 5-year means are mostly noise and a max-return optimiser
  piles into past winners; shrinking within groups keeps bonds from inheriting equity returns.
  The AGG benchmark is shrunk toward "Bond Intermediate". See `src/estimators.py`.
- Covariance: Ledoit-Wolf on standardised returns over the 60-month window (shrinks
  correlations toward zero and keeps each asset's sample volatility exactly; the identity
  target on raw returns would inflate low-risk bond volatilities).
- Solver: cvxpy with CLARABEL, fallback SCS.
- **Backtest** (`src/backtest.py`): common start at the 2012-01-31 rebalancing (first date with
  five eligible bond funds: 70% bonds at 15% per fund), rolling 60-month window, rebalancing
  every 3 months from the start (or
  semi-annual / annual), transaction costs 10 bps per unit traded. Compared with SPY and the
  profile benchmark: annualised return, volatility, Sharpe, max drawdown. It must compare the
  **ex-ante volatility promised** at each rebalancing with the **realised volatility**.
- **Simulation** (`src/risk.py`): Monte Carlo block bootstrap (12-month blocks) of the client's
  wealth over 5-20 years: median, 5% and 95% paths, versus the benchmark.

## App (`app.py`, five tabs)
1. **Client Profile** — questionnaire and resulting profile, with the reasons.
2. **Constraints** — each constraint with its toggle/slider, preset to the profile defaults;
   clear message naming the conflicting constraints when infeasible.
3. **Portfolio** — ex-ante return / volatility / Sharpe vs benchmark and SPY, allocation by asset
   class and equity sector, holdings; **Bond portfolio** section: share of bonds, average
   duration and rating, investment-grade and high-yield shares, breakdown by issuer type,
   maturity bucket and credit quality, table of bond funds (`src/bonds.py`).
4. **Backtest** — cumulative performance vs SPY and profile benchmark, metrics table,
   ex-ante vs realised volatility.
5. **Simulation** — wealth fan chart and percentiles vs benchmark.

## Code organisation
```
app.py                     # Streamlit UI only — no heavy logic
src/data.py                # loading parquet, universe tags
src/estimators.py          # shrunk means and Ledoit-Wolf covariance, eligibility window
src/optimizer.py           # cvxpy problem builder, constraint objects, infeasibility diagnosis
src/benchmarks.py          # SPY and SPY/bond profile benchmarks, ex-ante comparison
src/bonds.py               # bond sleeve analytics (segments, duration, rating), rating scale
src/backtest.py            # rolling backtest engine
src/risk.py                # performance metrics, Monte Carlo block bootstrap
src/profiling.py           # questionnaire -> profile -> vol cap, bond floor, default constraints
src/report.py              # PDF report (charts drawn with matplotlib — no kaleido)
scripts/build_universe.py  # constituents snapshot -> data/universe.csv
scripts/download_data.py   # prices, risk-free rate, benchmarks -> data/*.parquet
tests/                     # pytest for data, estimators, constraints, profiles, backtest
reference/                 # friend's app — read-only inspiration, NEVER copy its ERC logic
```

## Rules for Claude Code
- UI text, code, comments: English. Explanations to the team in chat: French.
- `reference/friend_app.py` is a previous group's project (ERC pension fund). You may reuse
  GENERIC patterns (Streamlit layout, CSS cards, metric formatting, PDF export, backtest loop
  structure), but rewrite them in our own structure and style. Do not reuse its strategy, texts,
  team section, colors or branding.
- Every function that can fail (solver, data) must fail gracefully with a clear message in the app.
- Cache expensive computations with `st.cache_data`.
- Tests are split with the pytest marker `slow` (app scenarios with Streamlit AppTest, several
  minutes; the whole of `tests/test_smoke.py`). During development run only the fast tests:
  `pytest -m "not slow"` (~45 s). Before the final commit of a step run the FULL suite: `pytest`
  (fast + slow) and a `streamlit run app.py` headless smoke test, then say done.
- New AppTest scenarios go in `tests/test_smoke.py` (marked slow); everything else stays fast.
- One commit per finished and tested step, with a clear message.
- Keep `requirements.txt` pinned and deployable on Streamlit Community Cloud.
- Target **Python 3.12** (local `.venv` and Streamlit Cloud "Advanced settings"). The pinned set
  must stay free of deprecation warnings: `pytest.ini` turns DeprecationWarning/FutureWarning
  into errors. Known constraints: pandas 2.x (yfinance warns under pandas 3), matplotlib 3.10.x
  (pandas 2.3 plotting warns under 3.11), numpy 2.3.x (pandas 2.3 resample warns under 2.5).
