# NSE Stocks Signal Engine

A free-data research application for finding Indian companies whose fundamentals may be improving before the market fully recognises the change.

## V1.3 scope

- No paid market-data subscription
- No database
- Fresh data fetched at scan time
- Optional local cache can be added later
- Foundation scoring: Quality, Earnings, Cash Flow, Momentum
- Early Turnaround scoring: earnings, margin, debt, CFO, FCF and revenue inflection
- Turnaround stages: WEAK / WATCH / EARLY / CONFIRMING
- Sector Rotation scoring using constituent relative strength, breadth, earnings and turnaround signals
- Relative valuation context using PE, price/CFO and price/FCF proxies
- Sector-adjusted Opportunity Score including valuation
- Stock-level diagnostics and CSV export

## Important data note

V1 uses `yfinance` for market and financial data. The yfinance documentation states that it accesses Yahoo Finance's publicly available APIs and is intended for research/educational purposes; users must follow the underlying data provider's terms.

For production-scale use, replace the provider with permitted exchange/company filing feeds.

## Run locally

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Custom universe

Create `data/universe.csv`:

```csv
symbol,company,sector
RELIANCE,Reliance Industries,Energy
HDFCBANK,HDFC Bank,Financials
...
```

The application converts symbols to Yahoo Finance NSE tickers (`SYMBOL.NS`).

## Roadmap

1. Foundation engine
2. Turnaround detection
3. Sector rotation
4. Valuation disconnect
5. Market regime
6. Historical backtesting
7. Portfolio/risk engine
8. AI filing intelligence


## V1.4 — Market Regime Engine

V1.4 adds a market-context layer without allowing market timing to dominate stock selection.

### Market regime signals
- Nifty price vs 50-day and 200-day moving averages
- 50-day moving-average slope
- 60-day and 200-day market momentum
- breadth across the scanned universe
- India VIX when freely available through the market-data provider

### Regime stages
- **BEAR**: <35
- **BOTTOMING**: 35–49
- **EARLY RECOVERY**: 50–64
- **RECOVERY**: 65–79
- **BULL**: 80+

The regime score is used only as a small contextual modifier to the Opportunity Score. Fundamentals, turnaround evidence, sector recovery and valuation remain the core stock-selection inputs.

> The regime engine is a research signal, not a prediction of the exact market bottom or a personalized asset-allocation recommendation.


## V1.5 — Walk-Forward Backtesting

V1.5 introduces a no-database historical validation framework.

- Every successful scan saves a dated point-in-time ranking under `data/snapshots/`.
- Backtesting uses only those historical snapshots; it does **not** rebuild old scores from today's fundamentals.
- Entry is the first trading close strictly after the snapshot date.
- Tests 3/6/12-month forward returns, win rate and drawdown.
- The framework is intentionally conservative to reduce look-ahead bias.

A meaningful historical dataset must be accumulated across different market regimes before drawing conclusions about the screener's predictive edge.


### V1.5 Backtesting implementation
Point-in-time scan snapshots are stored in `data/snapshots/` and evaluated with forward returns and drawdown.


## V1.6 — Backtest Validation

Adds top-N excess return versus an equal-weight scanned-universe control and Opportunity Score rank-bucket analysis. This tests whether ranking order itself contains predictive information before changing score weights.


## V1.7 — Sector-Specific Valuation

- Financials: P/B + ROE.
- IT: P/E + FCF yield.
- Manufacturing: EV/EBITDA + ROCE + FCF yield.
- Consumer: P/E + FCF yield + growth context.
- Utilities: EV/EBITDA + FCF yield + leverage.
- Cyclicals: P/E/FCF with an explicit normalized-earnings warning.

Valuation remains a relative ranking signal, not an intrinsic-value or guaranteed-return estimate. Asset quality for financials and normalized-cycle earnings are planned refinements.


### Quote data reliability
Current price and market cap use multiple yfinance quote fallbacks. If quote metadata is unavailable, the screener keeps the value as N/A rather than fabricating it.


### Price display fix
The displayed stock price now comes from the latest close in the scan's downloaded market-price dataset, while market cap uses quote fallbacks. This separates price availability from Yahoo quote-metadata availability.


### Quote reliability
The displayed price is taken from the latest downloaded market close, independent of Yahoo quote metadata. Market cap uses yfinance quote fallbacks and is left blank when unavailable rather than fabricated.


## V1.8 — Bear / Base / Bull Scenario Valuation

Adds a transparent one-year scenario layer. Banks use P/B + book value per share, Manufacturing/Utilities use EV/EBITDA, and other sectors use P/E with modest growth assumptions. Scenario values are research estimates, not intrinsic values or guaranteed targets. Missing data remains missing.


## V1.8.2 — Scenario Data Pipeline Fix

The scenario valuation layer now receives the full raw fundamentals required for P/B, P/E, and EV/EBITDA calculations. The pipeline avoids relying on reduced ranking fields and preserves missing-data safeguards.

## V1.9 — Valuation Data Quality & Normalization Flags

- Keeps genuinely unavailable metrics as `N/A`; no fabricated values.
- Distinguishes `COMPLETE`, `PARTIAL`, and `INSUFFICIENT` valuation data.
- Reports which valuation metrics are applicable and which are actually available.
- Adds valuation data quality percentage and human-readable warnings.
- Flags very high P/E and EV/EBITDA multiples for investigation rather than silently treating them as normal.
- Flags extreme growth rates caused by potential base effects.
- Keeps an explicit normalization warning for cyclical businesses.
- Adds regression tests for missing-data handling and extreme-value warnings.


## V1.9 — Risk Engine & Investment Decision Matrix

V1.9 adds a capital-preservation layer above the opportunity score. It combines survival evidence, scenario risk/reward, valuation context, foundation, sector recovery and market regime into a transparent `risk_score`, then applies hard gates for insufficient data and excessive Bear-case downside.

Decision categories: `ATTRACTIVE RISK/REWARD`, `PROMISING — WATCH / CONFIRM`, `WATCH — NEED MORE EVIDENCE`, `HIGH DOWNSIDE RISK`, `DATA INSUFFICIENT`, `AVOID / NO EDGE`, and `LOW PRIORITY`.

The engine does not manufacture missing inputs and does not claim to predict exact bottoms or future prices.


## V1.10 — Investment Decision Matrix

Adds decision rationale, research priority, next confirmation step, and thesis invalidation conditions to the Risk & Decision layer. These are screening/research aids, not individualized investment advice.
