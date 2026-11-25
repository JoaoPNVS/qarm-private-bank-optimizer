# Private Client Portfolio Optimizer

A Streamlit web app built for the course *Quantitative Asset and Risk Management II* at
HEC Lausanne. We play the role of an **international private bank** advising clients whose
reference currency is the **US dollar**: the app profiles the client, builds the portfolio with
the highest expected return within the client's volatility budget and constraints, backtests the
strategy and simulates the client's future wealth. Its house view follows **what company insiders
do with their own money**: the breadth of insider buying across all US listed companies (SEC
Form 4 filings since 2006) tilts the portfolio between equities and bonds.

## The app

| Tab | What it shows |
|---|---|
| Home | what the app does, today's recommended portfolio and insider signal at a glance, and the five steps |
| Client Profile | client settings (wealth, profile override, house view and its confidence) and the 7-question suitability questionnaire → Conservative / Balanced / Growth / Dynamic, with the reasons |
| Portfolio | every client constraint with a toggle and a slider (in the "Adjust constraints" panel), preset to the profile; conflicting constraints are named when the portfolio is impossible. Sub-tabs: *Overview* (expected return / volatility / Sharpe vs the profile benchmark and SPY, efficient frontier with and without the client's constraints and the return they cost, minimum-variance and maximum-Sharpe portfolios, capital market line), *Allocation* (by asset class and sector, where the risk comes from: risk contributions and diversification measures), *Bonds* (segments, duration, rating), *Holdings* (weights in USD, trading cost per asset, binding constraints) |
| Insider Signal | today's breadth of insider buying and the view it implies, its history against SPY, the evidence and its limits, the Black-Litterman view, recent insider purchases in our companies and why we do not bet on single stocks |
| Backtest | rolling backtest since 2012 vs an equal-weight portfolio of the same universe, the profile benchmark and SPY; promised vs realised volatility; allocation over time; side-by-side comparison of the house views |
| Simulation | Monte Carlo (block bootstrap) of the client's wealth over 5-20 years vs the benchmark: range of outcomes, loss beyond the client's limit, worst fall, one-year VaR and expected shortfall |

The "About Us" button (top right) presents the project and the team.

## Methodology

- **Profiles.** Risk capacity, risk willingness and knowledge are scored separately; the profile
  is the lower of capacity and willingness, capped by knowledge (horizon < 3 years → Conservative).
  Each profile has a volatility cap (6 / 10 / 12 / 15%) equal to the historical volatility of its
  benchmark rounded up, and a minimum share in bond funds (70 / 40 / 20 / 0%).
- **Optimisation.** Long-only, fully invested: maximise expected return subject to
  `volatility ≤ cap` and the client constraints, minus each asset's trading cost on the trades
  (cvxpy, CLARABEL solver).
- **Expected returns: Black-Litterman.** Prior = market-equilibrium returns from reverse
  optimisation, `pi = delta x Cov(asset, market)` with a 60% SPY / 40% AGG market portfolio and
  delta = 2.5, combined with a house view chosen in the client settings, with He & Litterman's
  uncertainty `Omega = diag(P tau Sigma P') (1 - c) / c` (tau = 0.05, c = confidence).
  - **Insider trading (default).** Breadth of insider buying: among US listed companies whose
    officers or directors traded their own shares on the open market over the last 3 months, the
    share with purchases. Compared with its last 5 years (z-score, capped at ±2), it gives one
    relative view: equities (SPY) beat bonds (AGG) by the equilibrium spread + 2% × z. Default
    confidence 50%. Breadth peaked at the market lows of 2008–09, 2011 and 2020; after the fifth of
    months with the broadest buying, SPY beat the T-bill by about 20% over the next year, against
    about 12% otherwise (modest and noisy evidence, shown in the app with its limits). Insider
    purchases of single large companies showed no edge, so they are displayed but not used.
    Backtest 2012–2026 against no view: slightly higher Sharpe in every profile (Balanced 1.09 vs
    1.07), smaller drawdowns, realised volatility at the budget (10.1% for 10%).
  - **Past performance.** Views = each asset's 60-month average return (default confidence 10%).
    Higher backtest Sharpe (Balanced 1.15), but more risk than promised (11.1% for a 10% budget),
    deeper drawdowns, twice the turnover, and partly hindsight: it favours past winners in a list
    of today's winners.
- **Risk:** Ledoit-Wolf covariance on standardised returns over a rolling 60-month window. An
  asset is used only once it has 60 months of history.
- **Constraints.** Volatility cap; minimum bonds; at most 10% cash and cash equivalents; at most
  8% per stock, 15% per bond fund, 20% per other fund, 40% per equity sector; emerging markets at
  most 1/3 of equities; bond duration, investment-grade and high-yield limits by profile;
  exclusions of sectors or companies.
- **Trading costs** by liquidity tier, from each asset's average daily dollar volume: 2 / 5 / 10 bps
  for stocks, 8 bps for high-yield and EM bond funds, 3 bps for other funds, 0 for cash.
- **Backtest.** Re-optimised every 3 (or 6 / 12) months using only past data, weights drift in
  between, costs charged on every trade. Benchmarks: SPY, a SPY / AGG mix per profile
  (30/70, 60/40, 80/20, 100/0) and an equal-weight portfolio of the same stocks and bond funds.
- **Simulation.** 5,000 paths built from random 12-month blocks of the strategy's out-of-sample
  backtest returns (2012–2026, each portfolio chosen with past data only, costs deducted), the
  benchmark on the same blocks. Risk shown as the chance of a one-year loss above the limit the
  client stated in the questionnaire, the worst fall from a peak (typical and 5% worst case), and
  the one-year value at risk and expected shortfall (CVaR) at 95%.
- **Risk analysis.** Risk contributions by asset and asset class (Euler decomposition of the
  volatility: marginal, absolute and relative contributions), compared with the benchmark;
  diversification measures (Herfindahl index and effective number of positions, Shannon entropy,
  Gini coefficient, diversification ratio). On the efficient frontier: the minimum-variance and
  maximum-Sharpe (tangency) portfolios under the client's constraints and the capital market line.

## Data

Everything is in USD, downloaded once and committed as `data/*.parquet`; the app never calls the
network.

- **Universe** (`data/universe.csv`): 150 US-listed stocks (all Nasdaq-100 members plus the
  largest other S&P 500 members, from Wikipedia constituent lists), international equity ETFs
  (VGK, EWJ, EEM), 13 USD bond ETFs as market segments (Treasury, corporate investment grade by
  maturity, high yield, municipal, mortgage-backed, EM USD, international hedged, TIPS), gold
  (GLD), cash equivalents (BIL, FLOT) and cash at the 3-month T-bill (`^IRX`).
- Monthly total returns from 1998; bond fund duration and ratings from the issuers' fact sheets
  (October 2026), with source and date in `universe.csv`.
- **Insider trades:** SEC Insider Transactions Data Sets (Form 4, 2006–2026): open-market
  purchases and sales by officers and directors of every US listed company (about 1.1 million
  filings), matched to our stocks by SEC registration number, including predecessors such as
  Google Inc. before Alphabet. Foreign companies (ASML, Shopify, ...) do not file Form 4.
- To refresh: `python scripts/build_universe.py`, `python scripts/download_data.py`, then
  `SEC_USER_AGENT="Your Name your.email@example.com" python scripts/download_insiders.py` (the SEC
  requires a contact in automated requests; raw files are cached outside the repo).

## Limitations

- **Survivorship bias.** The stock list is today's index membership applied to the past:
  companies that failed or left the indices are missing. Backtests and simulations of the
  strategy are therefore optimistic. SPY and the profile benchmarks are not affected; the
  equal-weight portfolio has the same bias, so the gap between it and the strategy is the fair
  measure of what the optimisation adds.
- Bond fund durations and ratings are current values applied to the whole history.
- The volatility caps are calibrated on the full 2003–2026 history, which includes years after the
  backtest start (2012). On the data available in January 2012 alone they would have been the same
  (Conservative 6%, Balanced 10%) or higher (Growth 13%, Dynamic 16%, because 2003–2011 includes
  2008), so this choice does not flatter the backtest.
- The volatility promised by the optimiser is exceeded out of sample on average (estimation
  error): 11.1% realised for a 10% budget at the default confidence, 10.0% with the pure
  equilibrium (0%).
- The optimised strategy does not clearly beat the equal-weight portfolio of the same universe
  (Balanced Sharpe 1.15 vs 1.16); a well-known result with short estimation windows (DeMiguel,
  Garlappi and Uppal, 2009).
- The simulation resamples history only; its range is only as wide as the past it was built from.
- Educational project, not investment advice.

## Run locally

Requires Python 3.12.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Tests: `pytest -m "not slow"` (fast, ~45 s) during development, `pytest` for the full suite
including the app scenarios (~7 min).

## Deploy on Streamlit Community Cloud

New app → this repository, branch `main`, main file `app.py`, and in **Advanced settings**
choose **Python 3.12**. No secrets are needed.

## Repository layout

```
app.py                 Streamlit UI
src/                   data, estimators, optimizer, frontier, diversification, benchmarks, bonds, backtest, risk, profiling, team
scripts/               one-off data download (universe snapshot, prices, costs)
data/                  committed data files and team.json
assets/team/           team photos
tests/                 pytest suite
```
