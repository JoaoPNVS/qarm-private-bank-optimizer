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
- Example constraints (all must be editable in the app):
  - Emerging-market equities ≤ 1/3 of total equity exposure (ratio constraint, linear form)
  - Swiss equities ≥ X% of equity exposure (home bias)
  - Max weight per asset (e.g. 15%), min weight if held (optional)
  - Asset-class bands (equity, bonds, alternatives, cash) depending on profile
  - Max FX exposure not in CHF (unhedged foreign currency) ≤ Y%
  - ESG / sector exclusions (toggle assets out)
  - Turnover cap per rebalance and transaction costs (L1 penalty)
  - Optional: max portfolio volatility, max tracking error vs benchmark

## Data (no paid data — everything from yfinance)
- Investable universe = liquid ETFs + a few Swiss blue chips, chosen so constraints are
  meaningful (each asset tagged with: asset_class, region, currency, is_EM, is_swiss, sector/ESG flag).
  Keep the universe definition in ONE file: `data/universe.csv`.
- FX: USDCHF=X, EURCHF=X (convert to CHF). Monthly frequency, month-end.
- Risk-free: **SARON from the SNB data portal** (cube `zimoma`, free, no API key), downloaded by
  the same script into `data/rf_chf.parquet`. **Documented exception to the "yfinance only" rule:**
  yfinance has no reliable CHF short rate. Monthly rf_t = SARON_{t-1} / 1200 (rate known at the
  start of the month). The app offers a user-settable constant annual rate as fallback
  (`src.data.constant_rf`). The `CASH_CHF` asset is synthetic and earns this rf.
- Currency tags: `currency` = economic FX exposure (used by the FX constraint), `listed_currency`
  = quote currency (used for CHF conversion; only CHF and USD exist, so USDCHF=X suffices).
  EEM, AAXJ and EMB are tagged `EM-FX` and international funds (BWX, RWX) `Multi`; every
  currency other than CHF counts as non-CHF exposure (`non_chf` column added by `load_universe`).
- Data cleaning: `data_start` in universe.csv discards corrupted early Yahoo history
  (SRECHA.SW before 2011-02, ABBN.SW before 2003). The download script aborts on any monthly
  CHF return above 60% and warns on daily moves above 25%. Roche is `RO.SW` (ROG.SW is not on
  Yahoo). Only complete months are kept. Common sample of all 35 assets starts 2011-03.
- Benchmark: a profile-dependent CHF benchmark (e.g. 60/40 built from universe ETFs).
- **Data is downloaded once by `scripts/download_data.py` and saved to `data/*.parquet`,
  which are committed to the repo.** The Streamlit app reads parquet only — it must never call
  yfinance at runtime (rate limits would break the live demo).

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
src/backtest.py        # rolling backtest engine
src/risk.py            # metrics, VaR/CVaR, risk contributions, stress tests, Monte Carlo
src/profiling.py       # questionnaire -> profile -> volatility cap + default constraints
src/report.py          # PDF report (charts drawn with matplotlib — no kaleido)
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
