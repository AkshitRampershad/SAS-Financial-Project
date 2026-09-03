import joblib
import pandas as pd

from analytics.credit_risk_model import train_and_select_best
from analytics.regulatory_capital import portfolio_regulatory_capital
from analytics.stress_testing import SCENARIOS, credit_stress_summary, portfolio_stress_summary
from data.generate_synthetic_loans import generate_loans


def _model_and_loans(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    loans = generate_loans(n_loans=8000, seed=42)
    train_and_select_best(loans, seed=42)
    model = joblib.load("models/credit_risk_model.joblib")
    return loans, model


def test_credit_stress_increases_pd_and_loss_for_every_scenario(tmp_path, monkeypatch):
    loans, model = _model_and_loans(tmp_path, monkeypatch)
    for scenario_name in SCENARIOS:
        result = credit_stress_summary(loans, model, scenario_name)
        assert result["stressed_avg_pd"] > result["baseline_avg_pd"], scenario_name
        assert result["stressed_expected_loss"] > result["baseline_expected_loss"], scenario_name


def test_stress_severity_ordering(tmp_path, monkeypatch):
    """Severe should hurt more than Moderate, which should hurt more
    than Mild - not just "some increase", but ordered by severity.
    """
    loans, model = _model_and_loans(tmp_path, monkeypatch)
    increases = {name: credit_stress_summary(loans, model, name)["expected_loss_increase_pct"] for name in SCENARIOS}
    assert increases["Mild Recession"] < increases["Moderate Recession"] < increases["Severe Recession"], increases


def test_regulatory_capital_rises_under_stress(tmp_path, monkeypatch):
    loans, model = _model_and_loans(tmp_path, monkeypatch)
    from analytics.credit_risk_model import ALL_FEATURES

    baseline_pd = model.predict_proba(loans[ALL_FEATURES])[:, 1]
    baseline_cap = portfolio_regulatory_capital(loans, baseline_pd)

    scenario = SCENARIOS["Severe Recession"]
    stressed_loans = loans.copy()
    stressed_loans["unemployment_rate_at_origination"] += scenario["unemployment_shock_pp"]
    stressed_loans["gdp_growth_at_origination"] += scenario["gdp_growth_shock_pp"]
    stressed_pd = model.predict_proba(stressed_loans[ALL_FEATURES])[:, 1]
    stressed_cap = portfolio_regulatory_capital(loans, stressed_pd)

    assert stressed_cap["required_capital"] > baseline_cap["required_capital"]
    assert stressed_cap["avg_risk_weight"] > baseline_cap["avg_risk_weight"]


def test_portfolio_stress_impact_is_negative_for_recession_scenarios():
    weights = pd.Series({"US Large Cap Equity": 0.6, "US Investment Grade Bonds": 0.3, "Cash": 0.1})
    for scenario_name in SCENARIOS:
        result = portfolio_stress_summary(weights, scenario_name)
        assert result["portfolio_return_impact"] < 0, scenario_name


def test_portfolio_stress_impact_scales_with_allocation():
    """An all-equity portfolio should be hit harder than an
    all-bonds/cash portfolio under the same recession scenario - the
    calculation should be allocation-sensitive, not a fixed number."""
    aggressive = pd.Series({"US Large Cap Equity": 1.0})
    conservative = pd.Series({"US Investment Grade Bonds": 0.5, "Cash": 0.5})

    agg_impact = portfolio_stress_summary(aggressive, "Moderate Recession")["portfolio_return_impact"]
    con_impact = portfolio_stress_summary(conservative, "Moderate Recession")["portfolio_return_impact"]
    assert agg_impact < con_impact
