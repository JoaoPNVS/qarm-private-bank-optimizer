# Private Client Portfolio Optimizer — QARM II project

## What this project is
A Streamlit web app for the HEC Lausanne course *Quantitative Asset and Risk Management II*.
We act as an **international private bank**. Our clients are international private clients
whose reference currency is the **USD**. The app profiles the client, builds the portfolio with
the highest expected return within the client's volatility budget and constraints, backtests the
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
- Each profile sets a **volatility cap** (volatility budget) and a **floor on bond funds** (cash does
  not count). Wording: say "volatility budget", not "risk budget": in the course, risk budgeting
  means splitting risk contributions across assets (shown in the Portfolio tab), a different idea.

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
- Maximum volatility = the profile's volatility budget (mandatory).
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
- Shadow prices (Holdings, "Binding constraints"): gain in expected return, bps per year, per unit
  of relaxation in the bound's own unit: +1pp of the portfolio; +1pp of the bonds for investment
  grade and +1pp of the equities for EM (dual x bond or equity share); +1 year for duration (dual x
  bond share). n/a for an exclusion or a 0% / 100% bound (dual not unique). Tested against
  finite-difference re-solves.

## Data (yfinance; Wikipedia for the constituent snapshot; SEC for insider trades)
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
- **Insider trades (SEC Form 4)**, `scripts/download_insiders.py`: the SEC "Insider Transactions
  Data Sets" (XML part of every Form 3/4/5, one zip per quarter, 2006Q1-2026Q3; the newest files
  moved to a second SEC folder, both are tried). Kept: Form 4 only (not 4/A or 5), open-market
  purchases (code P) and sales (S) by officers and directors (pure 10% holders left out), listed
  issuers, dated by FILING date. About 1.1 million filings; 74,000 in our stocks (2,527 purchases).
  Outputs (committed, small): `data/insider_market.parquet` (monthly, all US issuers: filings and
  companies with purchases / sales, USD amounts), `data/insider_trades.parquet` (every purchase or
  sale filing in one of our stocks), `data/insider_metadata.json`. Issuers are matched to our tickers
  by SEC CIK (`company_tickers.json`), plus `FORMER_CIKS` for reorganisations (Google Inc. before
  Alphabet, Avago/Broadcom Ltd, Praxair for LIN, Baker Hughes Inc for BKR, ...) and
  `CIK_VALID_FROM` (Merck's CIK was Schering-Plough's until Nov 2009). Foreign private issuers
  (ASML, CCEP, FER, SHOP, TRI) are exempt from Form 4. **The SEC requires a contact:** run with
  `SEC_USER_AGENT="Name email"` (environment variable, never written to the repo). Raw zips
  (~0.8 GB) are cached OUTSIDE the repo (`~/.cache/qarm-insiders`), reduced to small extracts.
- **Data is downloaded once and saved to `data/*.parquet`, which are committed to the repo.**
  The Streamlit app reads parquet only — it must never call the network at runtime.

## Methodology
- **Objective: maximise expected return under the profile's volatility cap.** Long-only, fully
  invested: max w'mu s.t. sqrt(w'Sigma w) <= sigma_max(profile) and the client constraints
  (+ L1 transaction-cost penalty when current holdings are given, e.g. in the backtest).
- **Expected returns: Black-Litterman** with ONE HOUSE VIEW, insider trading (team decision); the client
  sets only its confidence (Client settings). Without the insider files: equilibrium returns, no view.
  Prior = equilibrium returns by reverse optimisation, pi = delta Cov(asset, market) with
  market = 60% SPY / 40% AGG and delta = 2.5. General posterior for any views
  (`black_litterman_posterior` in `src/estimators.py`): mu = pi + tau Sigma P'(P tau Sigma P' + Omega)^-1
  (Q - P pi), Omega = diag(P tau Sigma P') (1 - c) / c, tau = 0.05 (He & Litterman).
- **House view: insider trading** (`src/insiders.py`). Breadth of insider buying = buying
  company-months / (buying + selling company-months) over the last 3 months: each month, the number of
  US listed companies whose officers/directors bought on the open market and the number whose insiders
  sold, each added up over 3 months (a company active in several months, or on both sides, counts each
  time; counting companies, not dollars). z-score against its own last 60 months
  (it drifts down since 2006 as 10b5-1 sale plans spread), capped at +-2. ONE relative view:
  E[SPY] - E[AGG] = equilibrium spread + 2% x z. Confidence c: slider 0-90%, default 50% (for a
  single view, c is exactly its weight against the equilibrium). Point in time: filings dated up to
  the end of the estimation window (in the backtest the month before the trade); no view if the data is older than 6 months. Evidence: breadth peaked at the
  market lows (Dec 2008 - Mar 2009, Oct 2011, May 2020); shown on SPY MINUS AGG (what the view
  predicts): after the top fifth of months by breadth, SPY beat AGG by ~18% over the next year vs ~11%
  after the other months (SPY vs T-bill: ~20% vs ~12%); 12-month predictive regression Newey-West
  t ~2.2 (1.3 from 2012). In-sample, overlapping the backtest: presented as a hypothesis, not proof.
  **Stock-level purchases are NOT used as views:** in our 150 large companies, stocks bought by
  their insiders did no better than the same month's universe or sector average over 3-12 months
  (`purchase_event_study`, t-stats around 0); shown in the app as information only. Backtest
  2012-05 to 2026-09 (one-month lag, see Backtest) at c = 50% vs no view (c = 0): Sharpe +0.01 to
  +0.03 in every profile, drawdowns about the same, realised volatility at the budget (Balanced
  1.05 vs 1.04, 10.3% realised, max DD -20.3% vs -20.3%). The breadth uses ALL listed companies, so it is free of our universe's
  survivorship bias.
- **Naive comparison (Backtest tab only, not offered to clients): past performance**
  (`black_litterman_mu`, `with_view_confidence`). Views = every
  asset's 60-month historical mean (P = identity). c = 10% (fixed in the
  comparison; 0% = pure equilibrium). The views are correlated and reinforce each other: at 10% a
  typical asset moves about a third of the way to its historical mean, SPY most of the way.
  Backtest 2012-05 to 2026-09, Balanced: Sharpe 1.13 (old method 0.88, equal weight 1.14), realised
  volatility 11.2% for a 10% budget (old 12.6%); c = 0: 10.6% (10.1% on average over the 12 months
  after each rebalancing). Omega
  proportional to the full tau Sigma (exact linear blend) was tested and rejected (worse). The
  portfolio and the benchmarks (SPY, AGG) get their expected returns from the same posterior.
  Its higher backtest Sharpe comes with more risk than promised in every profile (e.g. 7.1% for a
  6% budget), deeper drawdowns and about twice the turnover, and partly with hindsight: it favours past
  winners in a list of today's winners. Hence the insider view is the only house view.
- Previous method, kept for comparison and tests: historical mean over the 60-month window, shrunk toward the grand mean of
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
- **Transaction costs, asset-specific** (no slider): `scripts/download_data.py` computes each
  asset's average daily dollar volume (unadjusted close x volume, 12 months to the last
  month-end) and writes `adv_usd_m`, `liquidity_tier`, `tc_bps` to universe.csv. Tiers (cost per
  unit traded, one way): stocks >= $2bn/day 2 bps (43 stocks), $0.5-2bn 5 bps (85), < $0.5bn
  10 bps (22) — the 12 months to 2026-09 have a median of $1.2bn; high-yield / EM bond funds
  (HYG, EMB) 8 bps; every other fund (bond funds, BIL, FLOT, equity funds, gold) 3 bps, not
  tiered by volume because ETF liquidity comes from the underlying (VGK: $0.26bn a day, 1-2 bp
  spread); cash 0. The optimiser penalises sum_i c_i |w_i - w_prev_i| and the backtest charges
  the same costs. Run `build_universe.py` before `download_data.py` (the build rewrites the CSV).
  Today's recommendation starts from cash (no w_prev): its expected return is BEFORE the one-off cost
  of buying it (shown in the Overview caption). Limitation: the tiers use 2025-26 liquidity applied to
  the past; with every stock at 10 bps the backtest Sharpe falls by 0.04-0.06 (Balanced 1.01).
- **Backtest** (`src/backtest.py`): **timing = one-month implementation lag**: the decision traded at
  the close of month t uses data up to t-1 only (estimation window, eligibility, insider filings;
  `information_date`), so nothing assumes the month-end close is known before trading at it
  (deciding and trading at the same close added +0.01 to +0.03 to every profile's Sharpe). Common
  start at the 2012-04-30 rebalancing (first quarter-end whose data, to 2012-03, has five eligible
  bond funds: 70% bonds at 15% per fund), returns from 2012-05, rolling 60-month window, rebalancing
  every 3 months from the start (or
  semi-annual / annual), asset-specific transaction costs (see below). Compared with SPY and the
  profile benchmark: annualised return, volatility, Sharpe, max drawdown. It must compare the
  **ex-ante volatility promised** at each rebalancing with the **realised volatility**.
- **Equal-weight benchmark, same universe** (`run_equal_weight`), one per profile: at each
  rebalancing equal weights on the eligible stocks (same 60-month rule, hence the same
  survivors) for 1 - floor, and equal weights on the eligible bond funds (cash equivalents
  excluded) for the profile's bond floor; same dates, drift and per-asset costs as the strategy.
  It has the same survivorship bias, so the comparison is free of it; but the gap is NOT the effect
  of the optimisation alone: the equal-weight portfolio also ignores the volatility budget and the
  bond-sleeve limits (its bonds: ~6.4y duration, ~15% high yield, ~85% IG with today's fund data,
  outside the Conservative / Balanced limits). Backtest Sharpe vs equal weight / profile benchmark:
  Conservative 0.91 vs 0.86 / 0.64, Balanced 1.05 vs 1.14 / 0.84, Growth 1.07 vs 1.22 / 0.89, Dynamic
  1.13 vs 1.26 / 0.92 (above equal weight only for Conservative, above the benchmark in every profile). Shown in every backtest table and chart, with a plain-language "How to read these
  results" box for the client.
- **Simulation** (`src/risk.py`; labelled "Our strategy, 2012–2026 replayed", with a box above the
  amounts saying they replay a strong period and are not the ex-ante expected return): Monte Carlo block bootstrap (12-month blocks, 5,000 paths,
  fixed seed) of the client's wealth over 5-20 years, starting from the client wealth. History =
  the strategy's OUT-OF-SAMPLE backtest returns (2012-2026: each quarter's portfolio chosen with
  the previous 60 months only, trading costs deducted) and the profile benchmark on the SAME
  months; both are resampled with the same blocks. (Replaying today's weights on the months used
  to choose them was in-sample: Balanced median $52m from $5m in 10 years, 0% chance of loss.)
  Shown: median, 5% and 95% outcomes; the chance of a one-year loss above the limit the client
  stated in the questionnaire (5 / 10 / 20 / 30%); the worst fall from a peak during the horizon,
  typical (median) and bad case (5%); one-year value at risk and expected shortfall (CVaR) at 95%
  on the simulated one-year returns (expected shortfall is coherent, VaR is not); for the portfolio
  and the benchmark. The simulation inherits the survivorship bias (stated in the app).
- **Risk contributions and diversification** (`src/diversification.py`, ex ante, Portfolio tab):
  Euler decomposition of the volatility, MR_i = (Sigma w)_i / sigma, RC_i = w_i MR_i, RC*_i =
  RC_i / sigma (they add up exactly to sigma; tested against the course's 3-asset risk-budgeting
  example), by asset class and for the 10 largest contributors, compared with the benchmark
  (SPY carries most of a 60/40's risk). Diversification measures: Herfindahl index and its
  inverse (effective number of positions), exp(Shannon entropy), Gini coefficient of the weights
  over the eligible assets, diversification ratio sum(w_i sigma_i) / sigma (vs benchmark), share
  of risk from the top 10. Reporting only: the strategy does not target risk contributions (no
  ERC; that was the previous group's project).
- **Special frontier points** (`src/frontier.py`): minimum-variance portfolio (left end of the
  constrained curve) and maximum-Sharpe (tangency) portfolio under the client's constraints, found
  by golden-section search on the volatility cap (the Sharpe ratio along a concave frontier has one
  peak); capital market line from the T-bill (origin, returns are in excess) through it, labelled "theoretical allocation line" in the app (its points mix with cash or leverage and can break the client's rules).

## App (`app.py`, light "mist" UI: six tabs in a floating bottom navigation, no sidebar)
0. **Home** (opens first, no page header) — a deep-blue ribbon "Welcome to the Private Client Portfolio
   Optimizer" (what we do, the insider house view, four highlights), "Latest available view (data to <month>)"
   (insider signal and breadth, investment universe, data date: only what is true before the client answers;
   the client's profile and portfolio figures come after the questionnaire) and the five steps of the app, one card per tab.
1. **Client Profile** — "Client settings" box (wealth, profile override, confidence in the insider
   house view; these inputs drive every tab), the questionnaire as an accordion, and the resulting
   profile with the reasons.
2. **Portfolio** — "Adjust constraints" expander (each constraint with its toggle/slider, preset to
   the profile defaults; clear message naming the conflicting constraints when infeasible) and four
   sub-tabs:
   - *Overview*: ex-ante return / volatility / Sharpe vs benchmark and SPY; **efficient frontier**
     (`src/frontier.py`) with and without the client's constraints, the benchmark and SPY marked,
     the minimum-variance and maximum-Sharpe portfolios and the capital market line, the allocation
     along the curve and the return the constraints cost at the client's risk;
   - *Allocation*: by asset class and equity sector; **Where the risk comes from**: money vs risk
     share by asset class, the 10 largest risk contributors, diversification measures
     (`src/diversification.py`);
   - *Bonds*: share of bonds, average duration and rating, investment-grade and high-yield shares,
     breakdown by issuer type, maturity bucket and credit quality, table of bond funds (`src/bonds.py`);
   - *Holdings*: holdings chart, full allocation in USD, binding constraints.
3. **Insider Signal** — what the insider view says today (breadth, z-score, equity premium vs
   equilibrium, equities in the portfolio vs no view), breadth over time against SPY, the evidence
   (next-12-month SPY return by breadth quintile, honest limits), the Black-Litterman formula and
   view table, recent purchases in our companies and the stock-level event study (why we do not bet
   on single stocks), data source and coverage.
4. **Backtest** — cumulative performance vs SPY and profile benchmark, ex-ante vs realised
   volatility, drawdowns, allocation over time, then the metrics and trading tables; a "Compare with
   naive expected returns" toggle backtests the insider view (client's confidence), the naive
   past-performance view (10%) and no view side by side.
5. **Simulation** — wealth fan chart, median / 5% / 95% outcomes, one-year loss beyond the client's
   stated limit, worst fall, one-year VaR and expected shortfall, vs benchmark.
- **About Us** — a dialog opened by the top-right button: short project presentation and one card
  per team member. Team data lives in ONE editable file, `data/team.json` (name, role, description,
  photo path, LinkedIn; optional `repository` link). Photos go in `assets/team/`; a missing photo
  shows a round avatar with the initials (`src/team.py`). No e-mail addresses. Own texts and style,
  nothing from reference/.

## Code organisation
```
app.py                     # Streamlit UI only — no heavy logic
src/data.py                # loading parquet, universe tags
src/estimators.py          # Black-Litterman (and shrunk) means, Ledoit-Wolf covariance, eligibility window
src/insiders.py            # insider-buying breadth, the insider house view, evidence tables
src/optimizer.py           # cvxpy problem builder, constraint objects, infeasibility diagnosis
src/frontier.py            # efficient frontier under the client's constraints, min-variance, tangency
src/diversification.py     # risk contributions (Euler), diversification measures
src/benchmarks.py          # SPY and SPY/bond profile benchmarks, ex-ante comparison
src/bonds.py               # bond sleeve analytics (segments, duration, rating), rating scale
src/backtest.py            # rolling backtest engine
src/risk.py                # performance metrics, Monte Carlo block bootstrap, VaR / expected shortfall
src/profiling.py           # questionnaire -> profile -> vol cap, bond floor, default constraints
src/team.py                # About Us: team.json loader, photo or initials avatar
scripts/build_universe.py  # constituents snapshot -> data/universe.csv
scripts/download_data.py   # prices, risk-free rate, benchmarks -> data/*.parquet
scripts/download_insiders.py  # SEC Form 4 data sets -> data/insider_*.parquet (needs SEC_USER_AGENT)
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
  into errors. Known constraints: pandas 2.x (yfinance warns under pandas 3), numpy 2.3.x
  (pandas 2.3 resample warns under 2.5). matplotlib and fpdf2 were removed with the unused PDF
  report module.
- Colours: one `THEMES` table in `app.py` (night, slate, mist, ivory) holds the page colours (CSS
  variables) and the chart palette; the active one is `mist` (light blue-grey, navy accent), chosen for
  readability on a projector. `APP_THEME=<name>` overrides it for a preview. `.streamlit/config.toml`
  must hold the same base and colours (Streamlit's own widgets read them). One accent colour only;
  the bottom navigation has six tabs: compact buttons below 1200 px, and below 820 px they shrink
  so all six fit.
