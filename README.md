# Private Client Portfolio Optimizer

A Streamlit web app built for the course *Quantitative Asset and Risk Management II* at
HEC Lausanne. We play the role of an **international private bank** advising clients whose
reference currency is the **US dollar**: the app profiles the client, builds the portfolio with
the highest expected return within the client's risk budget and constraints, backtests the
strategy and simulates the client's future wealth.

## The app

| Tab | What it shows |
|---|---|
| Client Profile | 7-question suitability questionnaire → Conservative / Balanced / Growth / Dynamic, with the reasons |
| Constraints | every client constraint with a toggle and a slider, preset to the profile; conflicting constraints are named when the portfolio is impossible |
| Portfolio | recommended weights, expected return / volatility / Sharpe vs the profile benchmark and SPY, allocation by asset class and sector, bond portfolio (segments, duration, rating), trading cost per asset |
| Backtest | rolling backtest since 2012 vs an equal-weight portfolio of the same universe, the profile benchmark and SPY; promised vs realised volatility; allocation over time |
| Simulation | Monte Carlo (block bootstrap) of the client's wealth over 5-20 years vs the benchmark |
| About Us | the project and the team |

## Methodology

- **Profiles.** Risk capacity, risk willingness and knowledge are scored separately; the profile
  is the lower of capacity and willingness, capped by knowledge (horizon < 3 years → Conservative).
  Each profile has a volatility cap (6 / 10 / 12 / 15%) equal to the historical volatility of its
  benchmark rounded up, and a minimum share in bond funds (70 / 40 / 20 / 0%).
- **Optimisation.** Long-only, fully invested: maximise expected return subject to
  `volatility ≤ cap` and the client constraints, minus each asset's trading cost on the trades
  (cvxpy, CLARABEL solver).
- **Estimates** on a rolling 60-month window: historical means shrunk 80% toward the mean of the
  asset's group (equities, gold, cash; bonds by maturity bucket) and a Ledoit-Wolf covariance
  applied to standardised returns. An asset is used only once it has 60 months of history.
- **Constraints.** Volatility cap; minimum bonds; at most 10% cash and cash equivalents; at most
  8% per stock, 15% per bond fund, 20% per other fund, 40% per equity sector; emerging markets at
  most 1/3 of equities; bond duration, investment-grade and high-yield limits by profile;
  exclusions of sectors or companies.
- **Trading costs** by liquidity tier, from each asset's average daily dollar volume: 2 / 5 / 10 bps
  for stocks, 8 bps for high-yield and EM bond funds, 3 bps for other funds, 0 for cash.
- **Backtest.** Re-optimised every 3 (or 6 / 12) months using only past data, weights drift in
  between, costs charged on every trade. Benchmarks: SPY, a SPY / AGG mix per profile
  (30/70, 60/40, 80/20, 100/0) and an equal-weight portfolio of the same stocks and bond funds.
- **Simulation.** 5,000 paths built from random 12-month blocks of historical monthly returns,
  the portfolio and its benchmark on the same blocks.

## Data

Everything is in USD, downloaded once with `yfinance` and committed as `data/*.parquet`; the app
never calls the network.

- **Universe** (`data/universe.csv`): 150 US-listed stocks (all Nasdaq-100 members plus the
  largest other S&P 500 members, from Wikipedia constituent lists), international equity ETFs
  (VGK, EWJ, EEM), 13 USD bond ETFs as market segments (Treasury, corporate investment grade by
  maturity, high yield, municipal, mortgage-backed, EM USD, international hedged, TIPS), gold
  (GLD), cash equivalents (BIL, FLOT) and cash at the 3-month T-bill (`^IRX`).
- Monthly total returns from 1998; bond fund duration and ratings from the issuers' fact sheets
  (October 2026), with source and date in `universe.csv`.
- To refresh: `python scripts/build_universe.py` then `python scripts/download_data.py`.

## Limitations

- **Survivorship bias.** The stock list is today's index membership applied to the past:
  companies that failed or left the indices are missing. Backtests and simulations of the
  strategy are therefore optimistic. SPY and the profile benchmarks are not affected; the
  equal-weight portfolio has the same bias, so the gap between it and the strategy is the fair
  measure of what the optimisation adds.
- Bond fund durations and ratings are current values applied to the whole history.
- The volatility promised by the optimiser is regularly exceeded out of sample (estimation error).
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
src/                   data, estimators, optimizer, benchmarks, bonds, backtest, risk, profiling, team
scripts/               one-off data download (universe snapshot, prices, costs)
data/                  committed data files and team.json
assets/team/           team photos
tests/                 pytest suite
```
