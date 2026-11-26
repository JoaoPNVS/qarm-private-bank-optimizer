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
| Client Profile | client settings (wealth, profile override, confidence in the insider house view) and the 7-question suitability questionnaire → Conservative / Balanced / Growth / Dynamic, with the reasons |
| Portfolio | every client constraint with a toggle and a slider (in the "Adjust constraints" panel), preset to the profile; conflicting constraints are named when the portfolio is impossible. Sub-tabs: *Overview* (expected return / volatility / Sharpe vs the profile benchmark and SPY, efficient frontier with and without the client's constraints and the return they cost, minimum-variance and maximum-Sharpe portfolios, capital market line), *Allocation* (by asset class and sector, where the risk comes from: risk contributions and diversification measures), *Bonds* (segments, duration, rating), *Holdings* (weights in USD, trading cost per asset, binding constraints) |
| Insider Signal | today's breadth of insider buying and the view it implies, its history against SPY, the evidence and its limits, the Black-Litterman view, recent insider purchases in our companies and why we do not bet on single stocks |
| Backtest | rolling backtest since 2012 vs an equal-weight portfolio of the same universe, the profile benchmark and SPY; estimated vs realised volatility; allocation over time; comparison with naive expected returns (past performance) and with no view |
| Simulation | Monte Carlo (block bootstrap) of the client's wealth over 5-20 years vs the benchmark: range of outcomes, loss beyond the client's limit, worst fall, one-year VaR and expected shortfall |

The "About Us" button (top right) presents the project and the team.

## Methodology

- **Profiles.** Risk capacity, risk willingness and knowledge are scored separately. The profile
  is the lower of capacity and willingness, capped by knowledge; a horizon under 3 years always
  gives Conservative. Each profile has a volatility budget (6 / 10 / 12 / 15%, its benchmark's
  historical volatility rounded up) and a minimum share in bond funds (70 / 40 / 20 / 0%).
- **Optimisation.** Long-only and fully invested: maximise expected return subject to
  `volatility ≤ budget` and the client constraints, net of each asset's trading cost (cvxpy,
  CLARABEL solver).
- **Constraints.** At most 10% cash and cash equivalents; at most 8% per stock, 15% per bond
  fund, 20% per other fund and 40% per equity sector; emerging markets at most 1/3 of equities;
  bond duration, investment-grade and high-yield limits by profile; optional exclusions.
- **Expected returns: Black-Litterman.** The prior is the market equilibrium, from reverse
  optimisation on a 60% SPY / 40% AGG market (`pi = 2.5 × Cov(asset, market)`). It is tilted by
  one house view, with He & Litterman's uncertainty `Omega = diag(P tau Sigma P') (1 - c) / c`
  (tau = 0.05); the client chooses only the confidence c (default 50%).
- **House view: insider trading.** Each month we count the US listed companies whose officers or
  directors bought their own shares on the open market, and those whose insiders sold. Breadth =
  purchases / (purchases + sales), each count added up over 3 months (so a company active in
  several months, or on both sides, counts each time). Its z-score against the last 5 years
  (capped at ±2) gives one view: SPY beats AGG by the equilibrium spread + 2% × z.
  - *Evidence:* breadth peaked at the market lows of 2008–09, 2011 and 2020. After the fifth of
    months with the broadest buying, SPY beat the T-bill by about 20% over the next year,
    against about 12% otherwise; modest, noisy evidence, shown in the app with its limits.
    Purchases in single large companies showed no edge, so they are displayed but not used.
  - *Backtest (May 2012 – Sep 2026) against no view:* slightly higher Sharpe in every profile
    (Balanced 1.05 vs 1.04), similar drawdowns. A small gain, not a proof.
  - *Naive comparison (Backtest tab only):* each asset's 60-month average return as a view
    (10% confidence), a bet that recent winners keep winning. Higher Sharpe (Balanced 1.13), but
    11.2% realised volatility for a 10% budget, deeper drawdowns, twice the turnover, and partly
    hindsight. Not offered to clients.
- **Risk.** Ledoit-Wolf covariance on standardised returns over a rolling 60-month window; an
  asset is used only once it has 60 months of history. Risk contributions (Euler decomposition)
  and diversification measures (Herfindahl, entropy, Gini, diversification ratio); efficient
  frontier with the minimum-variance and maximum-Sharpe portfolios and the capital market line.
- **Trading costs** by liquidity tier, from average daily dollar volume: 2 / 5 / 10 bps for
  stocks, 8 bps for high-yield and EM bond funds, 3 bps for other funds, 0 for cash.
- **Backtest.** Re-optimised every 3 (or 6 / 12) months, weights drift in between, costs charged
  on every trade. Each decision uses data up to the previous month-end and trades at the close of
  the rebalancing month (deciding and trading at the same close would add 0.01–0.03 of Sharpe).
  Compared with SPY, a SPY / AGG benchmark per profile (30/70, 60/40, 80/20, 100/0) and an
  equal-weight portfolio of the same stocks and bond funds.
- **Simulation.** 5,000 paths of random 12-month blocks of the strategy's out-of-sample backtest
  returns, with the benchmark on the same blocks. Risk shown as the chance of a one-year loss
  above the client's stated limit, the worst fall from a peak, and one-year VaR and expected
  shortfall at 95%.

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

- **Survivorship bias.** The stock list is today's index membership applied to the past, so
  companies that failed or left the indices are missing and the strategy's backtest and
  simulation are optimistic. SPY and the profile benchmarks are not affected.
- **Equal weight.** The strategy does not beat the equal-weight portfolio of the same universe
  (Balanced Sharpe 1.05 vs 1.14, every profile below it), a known result with short estimation
  windows (DeMiguel, Garlappi and Uppal, 2009). The gap is not the effect of optimisation alone:
  equal weight ignores the volatility budget and the bond limits (its bonds average 6.4 years of
  duration and 15% high yield, beyond the Conservative and Balanced limits). The strategy does
  beat its profile benchmark (Balanced 1.05 vs 0.84).
- **Volatility budget.** It caps the estimate at each rebalancing, not future volatility. For
  Balanced, realised volatility over the whole backtest is 10.3% for a 10% budget.
- **Calibration.** The volatility caps use 2003–2026 data, after the backtest start. On data
  available in January 2012 they would be the same or higher (Growth 13%, Dynamic 16%), so this
  does not flatter the backtest.
- Bond fund durations and ratings are current values applied to the whole history.
- The simulation only resamples the past; its range is as wide as that history.
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
including the app scenarios (~15 min).

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
