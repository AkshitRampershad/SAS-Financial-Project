# Credit Risk & Investment Portfolio Dashboard

A dynamic business dashboard for analyzing credit risk and optimizing investment portfolios for a mid-sized financial institution — a real, runnable rebuild of this project's original SAS-based design write-up, using free Python tools in place of the SAS toolset.

## What it does

1. **Overview** — loan book KPIs (total book, default rate, avg credit score), origination volume over time, and the institution's current (unoptimized) investment allocation.
2. **Credit Risk** — a real trained classifier predicting probability of default (PD), ROC curve, feature importance, risk breakdowns by loan purpose and credit score band, required regulatory capital, and an interactive "score a hypothetical loan" tool.
3. **Portfolio Optimization** — real mean-variance (Markowitz) optimization: the efficient frontier, the max-Sharpe tangency portfolio, and a risk-tolerance slider that blends cash and the tangency portfolio along the capital allocation line.
4. **Stress Testing** — mild/moderate/severe recession scenarios re-scored through the same trained credit model (real PD/capital shifts, not fabricated multipliers) and applied to both the current and optimized investment allocations.

## What's real, and what's illustrative

| This project's original SAS toolset | What this dashboard actually runs |
| --- | --- |
| `PROC LOGISTIC`, `PROC DISCRIM` (classification models) | A real trained `scikit-learn` `LogisticRegression`, compared against a `GradientBoostingClassifier` and selected by actual held-out test AUC — never hardcoded. Calibration (predicted PD vs. realized default rate) is checked and reported, not assumed. |
| `PROC HPFOREST`, SAS Enterprise Miner (ensemble/predictive workflow) | The gradient boosting candidate stands in for the ensemble method; model selection is a small local "mini-AutoML" comparison, logged to `models/run_log.json`. |
| `PROC OPTMODEL` (portfolio optimization) | Real mean-variance optimization via `scipy.optimize` (SLSQP) — efficient frontier, max-Sharpe tangency portfolio, and a risk-tolerance allocation via the capital allocation line (Tobin two-fund separation, not a naive joint optimization that degenerates toward cash). |
| SAS/ETS, `PROC ARIMA` (time series / macro modeling) | A synthetic but structured macro cycle (unemployment, GDP growth, with a modeled 2020-style shock) feeds the credit model's origination-time features and the stress scenarios. |
| Regulatory Capital Calculation | A real (simplified) Basel II/III IRB capital formula — not a placeholder number. Simplified to a single fixed asset correlation rather than Basel's full PD-dependent correlation curve; documented in `analytics/regulatory_capital.py`. |
| SAS Visual Analytics (dashboard) | This Streamlit app. |
| 2.5M real loan records, real bureau/Bloomberg/FRED data | A **synthetic**, seeded ~30,000-loan portfolio and a synthetic multi-asset return series, modeled on realistic credit-risk and capital-markets structure — not scraped or real institutional data. See `data/generate_synthetic_loans.py` and `data/generate_synthetic_market_data.py` for full disclosure of what's simulated and why. |

Two honesty principles held throughout: the credit model's AUC and calibration are whatever it actually measures on held-out data, never tuned to hit a target number; and the portfolio optimizer's expected returns are analytic assumptions computed from the market-data generator's own parameters (not a noisy trailing sample mean — with realistic volatility and only a decade of simulated data, a trailing mean's standard error would swamp the signal, which is why real institutions use separate capital-market assumptions rather than historical averages for return forecasts).

## Project Structure
```text
app.py                              # Streamlit dashboard (4 tabs)
data/
  generate_synthetic_loans.py       # Synthetic loan-level portfolio generator
  generate_synthetic_market_data.py # Synthetic multi-asset-class return generator
analytics/
  credit_risk_model.py              # Trains and selects the PD model
  portfolio_optimizer.py            # Mean-variance optimization (scipy)
  stress_testing.py                 # Scenario shocks, re-scored through the real model
  regulatory_capital.py             # Basel II/III IRB capital calculation
tests/                              # Pytest suite (data, model, optimizer, stress test invariants)
requirements.txt
```

## Getting Started

### Prerequisites
- Python 3.11+

No API keys or external data sources are required — everything runs on the synthetic generators above.

### Setup
```bash
git clone https://github.com/AkshitRampershad/sas-financial-project.git
cd sas-financial-project
pip install -r requirements.txt
```

### Run the dashboard
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`. First load takes a few seconds while the loan portfolio, market data, and credit risk model are generated/trained (cached after that).

### Run individual components from the command line
```bash
python -m data.generate_synthetic_loans --rows 30000 --out data/raw/loans.csv
python -m data.generate_synthetic_market_data --months 120 --out data/raw/market_returns.csv
python -m analytics.credit_risk_model      # trains and prints test AUC
python -m analytics.portfolio_optimizer    # prints the max-Sharpe portfolio and a risk-tolerance sweep
python -m analytics.stress_testing         # prints credit + portfolio impact per scenario
python -m analytics.regulatory_capital     # prints portfolio RWA and required capital
```

### Run the tests
```bash
pytest -q
```

### Deploying to Streamlit Community Cloud
1. Push this repo to GitHub (already done if you're reading this from the deployed app's source).
2. At [share.streamlit.io](https://share.streamlit.io), create a new app pointing at this repo, branch `main`, with `app.py` as the main file path.
3. Deploy — no secrets needed.
