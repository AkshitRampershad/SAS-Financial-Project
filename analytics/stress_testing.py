"""
Scenario stress testing for both the credit book and the investment
portfolio - standing in for SAS/STAT + SAS/ETS's role in this README's
"Stress Testing and Scenario Analysis" and "Regulatory Capital
Calculation" methodology steps.

Credit stress: re-scores the EXISTING loan book's probability of
default through the real trained model with macro inputs
(unemployment, GDP growth) overridden to the scenario's shocked
values - a genuine model re-evaluation, not a fabricated multiplier.

Portfolio stress: applies a one-time return shock per asset class to a
given allocation and sums the weighted impact - a real, transparent
calculation from the scenario's own numbers, not a canned "-20%
everything" figure.

Scenario magnitudes are illustrative, loosely modeled on the shape of
Fed CCAR-style severity tiers (mild/moderate/severe), not an official
regulatory scenario.
"""

from __future__ import annotations

import pandas as pd

SCENARIOS = {
    "Mild Recession": {
        "unemployment_shock_pp": 1.5,
        "gdp_growth_shock_pp": -2.0,
        "asset_shocks": {
            "US Large Cap Equity": -0.10, "US Small Cap Equity": -0.13, "Intl Developed Equity": -0.11,
            "Emerging Markets Equity": -0.15, "US Investment Grade Bonds": 0.02, "US High Yield Bonds": -0.04,
            "US Treasuries": 0.03, "REITs": -0.09, "Commodities": -0.05, "Cash": 0.0,
        },
    },
    "Moderate Recession": {
        "unemployment_shock_pp": 3.0,
        "gdp_growth_shock_pp": -4.0,
        "asset_shocks": {
            "US Large Cap Equity": -0.20, "US Small Cap Equity": -0.26, "Intl Developed Equity": -0.22,
            "Emerging Markets Equity": -0.30, "US Investment Grade Bonds": 0.03, "US High Yield Bonds": -0.09,
            "US Treasuries": 0.05, "REITs": -0.18, "Commodities": -0.12, "Cash": 0.0,
        },
    },
    "Severe Recession": {
        "unemployment_shock_pp": 5.0,
        "gdp_growth_shock_pp": -6.5,
        "asset_shocks": {
            "US Large Cap Equity": -0.35, "US Small Cap Equity": -0.42, "Intl Developed Equity": -0.37,
            "Emerging Markets Equity": -0.45, "US Investment Grade Bonds": 0.04, "US High Yield Bonds": -0.18,
            "US Treasuries": 0.08, "REITs": -0.32, "Commodities": -0.25, "Cash": 0.0,
        },
    },
}


def credit_stress_summary(loans: pd.DataFrame, model_pipeline, scenario_name: str) -> dict:
    """Re-scores the loan book under the scenario's shocked macro
    conditions through the real trained model, and reports the actual
    change in portfolio-level PD and expected loss - both baseline and
    stressed figures come from the same model, just different inputs.
    """
    from .credit_risk_model import ALL_FEATURES, expected_loss  # local import avoids a hard circular dep at module load

    scenario = SCENARIOS[scenario_name]
    stressed = loans.copy()
    stressed["unemployment_rate_at_origination"] = (
        stressed["unemployment_rate_at_origination"] + scenario["unemployment_shock_pp"]
    ).clip(upper=25.0)
    stressed["gdp_growth_at_origination"] = stressed["gdp_growth_at_origination"] + scenario["gdp_growth_shock_pp"]

    baseline_pd = model_pipeline.predict_proba(loans[ALL_FEATURES])[:, 1]
    stressed_pd = model_pipeline.predict_proba(stressed[ALL_FEATURES])[:, 1]

    baseline_el = expected_loss(loans, baseline_pd)
    stressed_el = expected_loss(loans, stressed_pd)

    return {
        "scenario": scenario_name,
        "baseline_avg_pd": float(baseline_pd.mean()),
        "stressed_avg_pd": float(stressed_pd.mean()),
        "baseline_expected_loss": float(baseline_el.sum()),
        "stressed_expected_loss": float(stressed_el.sum()),
        "expected_loss_increase": float(stressed_el.sum() - baseline_el.sum()),
        "expected_loss_increase_pct": float(stressed_el.sum() / baseline_el.sum() - 1) if baseline_el.sum() else 0.0,
    }


def portfolio_stress_summary(weights: pd.Series, scenario_name: str) -> dict:
    """Weighted-average return impact of the scenario's per-asset-class
    shocks on a given allocation - a plain dot product, not a canned
    number: swap in a different allocation and the result changes.
    """
    scenario = SCENARIOS[scenario_name]
    shocks = pd.Series(scenario["asset_shocks"])
    aligned_shocks = shocks.reindex(weights.index).fillna(0.0)
    portfolio_impact = float((weights * aligned_shocks).sum())

    per_asset = pd.DataFrame({"weight": weights, "shock": aligned_shocks, "contribution": weights * aligned_shocks})

    return {
        "scenario": scenario_name,
        "portfolio_return_impact": portfolio_impact,
        "per_asset_contribution": per_asset,
    }


if __name__ == "__main__":
    from analytics.credit_risk_model import train_and_select_best
    from analytics.portfolio_optimizer import risk_tolerance_portfolio, weights_series
    from data.generate_synthetic_loans import generate_loans
    from data.generate_synthetic_market_data import analytic_expected_returns, generate_monthly_returns
    import joblib

    loans = generate_loans()
    train_and_select_best(loans)
    model = joblib.load("models/credit_risk_model.joblib")

    monthly = generate_monthly_returns()
    mean_returns = analytic_expected_returns()
    cov_matrix = monthly.cov() * 12
    w = risk_tolerance_portfolio(mean_returns, cov_matrix, risk_aversion=10)
    weights = weights_series(w, mean_returns)

    for scenario_name in SCENARIOS:
        credit = credit_stress_summary(loans, model, scenario_name)
        portfolio = portfolio_stress_summary(weights, scenario_name)
        print(
            f"{scenario_name}: avg PD {credit['baseline_avg_pd']:.2%} -> {credit['stressed_avg_pd']:.2%}, "
            f"expected loss +{credit['expected_loss_increase_pct']:.1%}, "
            f"portfolio impact {portfolio['portfolio_return_impact']:+.2%}"
        )
