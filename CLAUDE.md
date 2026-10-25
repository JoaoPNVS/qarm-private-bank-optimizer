# Swiss Private Client Optimizer — QARM II project

## What this project is
A Streamlit web app for the HEC Lausanne course *Quantitative Asset and Risk Management II*.
We act as a Swiss private bank / wealth manager. Our client is a **demanding Swiss private
client (CHF-based) with many investment constraints**. The app builds an optimal constrained
portfolio (maximum expected return under the profile's volatility cap), backtests it
and simulates the client's wealth.

Grading criteria (from the professor) — every feature should serve one of these:
1. Content: complexity of client & data, strategy covers the client's needs, uses course tools
   (quant optimization, covariance estimation, risk measures).
2. Presentation: results must be clear to a "client" in a 15-min pitch; we must be able to answer
   live questions (so the app must make "what if" changes easy).
3. Web app: design, interactivity, completeness, no technical issues (must never crash).

## The client
- Swiss resident, reference currency **CHF**. All returns are converted to CHF.
- Risk profile from an **LSFIN/FIDLEG-style suitability questionnaire** (horizon, loss tolerance,
  experience, income stability, objective) → profile Conservative / Balanced / Growth / Dynamic →
  maps to a volatility cap (risk budget) and to default constraint values.
  Implemented in `src/profiling.py` (7 questions: capacity / willingness / knowledge; profile =
  lower of capacity and willingness, knowledge cap, <3y horizon knock-out; each profile has a volatility cap and a CHF benchmark).
- Client constraints (simple, explainable, all editable in the app; `src/optimizer.py`):
  - Maximum volatility = the profile's risk budget (mandatory)
  - Max weight per stock (default 8%; bond ETFs and cash are not capped)
  - Max weight per equity sector (default 40%)
  - Exclusions of sectors or individual companies
  - Optional minimum share in bonds + cash (off by default, suggested level per profile)

## Data (no paid data — yfinance, plus Wikipedia/SNB for reference lists and the risk-free rate)
- **Universe** (`data/universe.csv`, built once by `scripts/build_universe.py`, snapshot date in
  `data/universe_as_of.txt`): 150 US-listed stocks = all Nasdaq-100 members + the largest other
  S&P 500 members by market cap (one share class per company: GOOGL, FOXA, NWSA kept), each
  with its GICS sector; 3 Swiss government bond ETFs (iShares CSBGC3/CSBGC7/CSBGC0, 0-3y,
  3-7y, 7-15y); synthetic CASH_CHF earning SARON. A few Nasdaq-100 members are foreign
  companies listed in the US (ASML, ARM, PDD, CCEP, FER, TRI, NBIS...); `isin_country` is the
  ISIN prefix, not the headquarters.
- **Benchmarks** (`src/benchmarks.py`): SPY converted to CHF (`data/benchmarks_chf.parquet`, not
  investable), and one benchmark per profile = SPY (CHF) / Swiss govt bonds 3-7y (CSBGC7.SW)
  rebalanced monthly: Conservative 30/70, Balanced 60/40, Growth 80/20, Dynamic 100/0 (mixed
  benchmarks start 2008-02). Ex-ante comparisons estimate SPY jointly with the assets (same
  Ledoit-Wolf fit, mean shrunk toward the equity grand mean); SPY is never investable.
- **Volatility caps** calibrated on the benchmarks: historical benchmark volatility 2008-02 to
  2026-09 rounded up to the next percent -> Conservative 6%, Balanced 10%, Growth 13%,
  Dynamic 16% (benchmark vols 5.1 / 9.5 / 12.6 / 15.7%). The portfolio thus takes the same
  risk as the benchmark it is compared with.
- Prices: yfinance daily adjusted close (dividends reinvested) -> month-end, from 2003.
  Everything is converted to CHF with USDCHF=X (USD assets have CHF returns from 2003-10).
  Only complete months are kept.
- Eligibility: an asset enters the optimisation only once it has a complete 60-month history
  (recent IPOs/spin-offs such as ARM, CRWV, GEV, SPCX wait; CHF bond ETFs have returns from
  2008-02, the 0-3y from 2009-08, so they become eligible in 2013-01 and 2014-07).
- Data checks in `download_data.py`: abort on a monthly CHF move above +200% (bad tick), print
  moves above 100% for manual review (all reviewed real, e.g. PLTR 2020-11, MRNA 2026-08 on
  40x normal volume). `data_start` in universe.csv can cut corrupted early history.
- Risk-free: **SARON from the SNB data portal** (cube `zimoma`, free, no API key), downloaded by
  the same script into `data/rf_chf.parquet`. **Documented exception to the "yfinance only" rule:**
  yfinance has no reliable CHF short rate. Monthly rf_t = SARON_{t-1} / 1200 (rate known at the
  start of the month). The app offers a user-settable constant annual rate as fallback
  (`src.data.constant_rf`).
- **Survivorship bias (important limitation).** The universe is TODAY's index membership applied
  to the past. Companies that went bankrupt, were acquired, shrank or left the indices since
  2003 (Lehman, Yahoo, Twitter, ...) are missing, and today's members are by construction the
  ones that grew. Any backtest on this universe therefore overstates what a real investor could
  have earned, and the stock-picking optimiser benefits most because it overweights past winners.
  SPY (CHF) is free of this bias, so outperformance versus SPY must be read as an UPPER bound.
  A clean study would need point-in-time constituents (e.g. CRSP), which are not freely available.
- **Data is downloaded once and saved to `data/*.parquet`, which are committed to the repo.**
  The Streamlit app reads parquet only — it must never call yfinance at runtime.

## Methodology
- **Objective: maximise expected return under the profile's volatility cap.** Long-only, fully
  invested: max w'mu s.t. sqrt(w'Sigma w) <= sigma_max(profile) and the client constraints
  (+ L1 transaction-cost penalty when current holdings are given, e.g. in the backtest).
  The volatility cap is mandatory (without it the problem puts everything in one asset).
- Expected returns: historical mean over the estimation window (default 60 months), shrunk
  toward the grand mean of the asset's own class: mu_i = (1 - a) m_i + a m_class, a = 0.5 by
  default (raw 5-year means are mostly noise; shrinking within each class keeps bonds from
  inheriting equity returns). See `src/estimators.py`.
- Covariance: Ledoit-Wolf applied to standardised returns (shrinks correlations toward zero,
  keeps each asset's sample volatility exactly; the identity target on raw returns would inflate
  low-risk CHF bonds); CASH_CHF is risk-free (zero variance, zero excess return).
- Eligibility: an asset enters the optimisation only once it has a complete 60-month history.
- Removed in the October 2026 redesign: equilibrium returns, Black-Litterman house views,
  risk-aversion lambda, cost of constraints, efficient frontier / minimum variance.
- Infeasibility: the optimiser never crashes; a deletion filter names the minimal set of
  conflicting constraints and the ones whose removal alone restores feasibility. Shadow prices
  (dual values) of binding constraints are kept, in bps of expected return per +1pp.
- Solver: CLARABEL with fallback to SCS.
- Backtest: rolling estimation window (default 60 months, user-adjustable 36–84 months), rebalancing
  quarterly/semi/annual, transaction costs. Compare with SPY in CHF and the profile benchmark.
- Risk: annualized vol, Sharpe, max drawdown; Monte Carlo (block bootstrap) for wealth projection.

## Code organisation
```
app.py                 # Streamlit UI only — no heavy logic
src/data.py            # loading parquet, CHF conversion, universe tags
src/estimators.py      # mu and covariance estimators
src/optimizer.py       # cvxpy problem builder, constraint objects, infeasibility diagnosis
src/benchmarks.py      # SPY (CHF) and SPY/CHF-bond profile benchmarks, ex-ante comparison
src/backtest.py        # rolling backtest engine
src/risk.py            # metrics, VaR/CVaR, risk contributions, stress tests, Monte Carlo
src/profiling.py       # questionnaire -> profile -> volatility cap + default constraints
src/report.py          # PDF report (charts drawn with matplotlib — no kaleido)
scripts/build_universe.py  # constituents snapshot -> data/universe.csv
scripts/download_data.py
tests/                 # pytest for optimizer constraints and backtest sanity
reference/             # friend's app — read-only inspiration, NEVER copy its ERC logic
```

## Rules for Claude Code
- UI text, code, comments: English. Explanations to the team in chat: French.
- `reference/friend_app.py` is a previous group's project (ERC pension fund). You may reuse
  GENERIC patterns (Streamlit layout, CSS cards, metric formatting, PDF export, backtest loop
  structure), but rewrite them in our own structure and style. Do not reuse its strategy, texts,
  team section, colors or branding.
- Every function that can fail (solver, data) must fail gracefully with a clear message in the app.
- Cache expensive computations with `st.cache_data`.
- After each feature: run `pytest` and `streamlit run app.py` headless smoke test before saying done.
- Keep `requirements.txt` pinned and deployable on Streamlit Community Cloud.
- Target **Python 3.12** (local `.venv` and Streamlit Cloud "Advanced settings"). The pinned set
  must stay free of deprecation warnings: `pytest.ini` turns DeprecationWarning/FutureWarning
  into errors. Known constraints: pandas 2.x (yfinance warns under pandas 3), matplotlib 3.10.x
  (pandas 2.3 plotting warns under 3.11), numpy 2.3.x (pandas 2.3 resample warns under 2.5).
