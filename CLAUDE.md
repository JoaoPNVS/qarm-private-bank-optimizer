# Swiss Private Client Optimizer — QARM II project

## What this project is
A Streamlit web app for the HEC Lausanne course *Quantitative Asset and Risk Management II*.
We act as a Swiss private bank / wealth manager. Our client is a **demanding Swiss private
client (CHF-based) with many investment constraints**. The app builds an optimal constrained
portfolio, backtests it, and explains to the client what each constraint costs him.

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
  maps to a risk-aversion parameter λ and to default constraint values.
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
- Covariance: Ledoit-Wolf shrinkage applied to standardised returns (shrinks correlations, keeps
  each asset's sample volatility); CASH_CHF is risk-free (zero variance, zero excess return).
- Expected returns: DEFAULT = equilibrium-implied returns mu = delta·Sigma·w_ref, with the
  50/40/10 reference portfolio `REFERENCE_WEIGHTS` in `src/estimators.py` and delta set by a 0.35
  reference Sharpe. Option: sample mean shrunk toward the grand mean. Raw means concentrate the
  optimiser on past winners (see the module docstring).
- House views (Black-Litterman on the equilibrium prior): absolute or relative views on tickers or
  tag-based baskets (reference-weighted), each with a confidence c; Omega_k = (1/c - 1)·tau·p'Σp
  (c = 50% = He-Litterman), tau = 0.05 (cancels out of the posterior mean). Default demo set
  `HOUSE_VIEWS` (4 views), switchable off. No active view => exactly the equilibrium returns.
- Extra constraints: regional bands on the share of EQUITIES per region (so no profile drops
  Europe or Japan), optional max tracking error vs the profile benchmark. Constraint costs are
  shown in bps and in CHF per year on the client's wealth.
- Optimizer: cvxpy, long-only, fully invested:
  max  wᵀμ − (λ/2) wᵀΣw − c·‖w − w_prev‖₁   subject to all active client constraints.
  Solver: CLARABEL with fallback to SCS/ECOS. If infeasible, the app tells the user WHICH
  constraints conflict (re-solve dropping one at a time) instead of crashing.
- Constraint analysis (our key differentiator):
  - Dual values (shadow prices) of each constraint from cvxpy.
  - "Cost of constraint": Sharpe / expected utility with vs without each constraint.
  - Efficient frontier: unconstrained vs constrained, client portfolio marked on it,
    weights along the frontier (stacked area).
- Backtest: rolling estimation window (default 60 months, user-adjustable 36–84 months), rebalancing quarterly/semi/annual,
  transaction costs, turnover cap. Compare with: profile benchmark, unconstrained MV, equal weight.
- Risk: annualized vol, Sharpe, max drawdown, historical VaR/CVaR 95%, risk contributions per
  asset and per asset class, historical stress tests (2008, 2011, 2015 SNB floor removal,
  2020 Covid, 2022 rates), Monte Carlo (block bootstrap) for wealth projection.

## Code organisation
```
app.py                 # Streamlit UI only — no heavy logic
src/data.py            # loading parquet, CHF conversion, universe tags
src/estimators.py      # mu and covariance estimators
src/optimizer.py       # cvxpy problem builder, constraint objects, infeasibility diagnosis
src/backtest.py        # rolling backtest engine
src/risk.py            # metrics, VaR/CVaR, risk contributions, stress tests, Monte Carlo
src/profiling.py       # questionnaire -> profile -> lambda + default constraints
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
