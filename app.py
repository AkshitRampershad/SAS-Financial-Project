"""
Credit Risk & Investment Portfolio Dashboard for a mid-sized financial
institution - a dynamic Streamlit business dashboard combining a real
trained credit risk model, real mean-variance portfolio optimization,
and real scenario stress testing.

See README.md for the full mapping from this project's original
SAS-toolset methodology to what each part of this dashboard actually
runs on.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analytics.credit_risk_model import ALL_FEATURES, TARGET, expected_loss, train_and_select_best
from analytics.portfolio_optimizer import efficient_frontier, max_sharpe_portfolio, portfolio_performance, risk_tolerance_portfolio, weights_series
from analytics.regulatory_capital import portfolio_regulatory_capital
from analytics.stress_testing import SCENARIOS, credit_stress_summary, portfolio_stress_summary
from data.generate_synthetic_loans import generate_loans
from data.generate_synthetic_market_data import ASSET_CLASSES, analytic_expected_returns, generate_monthly_returns


def fmt_currency_delta(value: float, unit: float = 1.0, suffix: str = "") -> str:
    """A signed currency string with the sign BEFORE the $ (e.g.
    "-$62,750,000", not "$-62,750,000") - st.metric's delta coloring
    reads the leading character for direction, so the sign has to come
    first or a negative delta silently renders as if it were positive.
    """
    scaled = value / unit
    sign = "-" if scaled < 0 else "+"
    return f"{sign}${abs(scaled):,.1f}{suffix}" if unit != 1.0 else f"{sign}${abs(scaled):,.0f}{suffix}"

st.set_page_config(page_title="Credit Risk & Portfolio Dashboard", layout="wide")
st.title("Credit Risk & Investment Portfolio Dashboard")
st.caption(
    "A dynamic business dashboard for analyzing credit risk and optimizing investment portfolios "
    "for a mid-sized financial institution."
)

RISKY_ASSETS = [a for a in ASSET_CLASSES if a != "Cash"]

# A plausible "as-is" legacy allocation an institution might hold before
# any optimization work - the baseline the Portfolio tab compares
# against. Illustrative, not derived from any real institution's books.
CURRENT_ALLOCATION = pd.Series({
    "US Large Cap Equity": 0.30, "US Small Cap Equity": 0.10, "Intl Developed Equity": 0.10,
    "Emerging Markets Equity": 0.05, "US Investment Grade Bonds": 0.20, "US High Yield Bonds": 0.05,
    "US Treasuries": 0.10, "REITs": 0.05, "Commodities": 0.00, "Cash": 0.05,
}, name="weight")[ASSET_CLASSES]


@st.cache_data
def load_loans() -> pd.DataFrame:
    return generate_loans()


@st.cache_data
def load_market_data():
    monthly = generate_monthly_returns()
    mean_returns = analytic_expected_returns()
    cov_matrix = monthly.cov() * 12
    return monthly, mean_returns, cov_matrix


@st.cache_resource
def get_credit_model():
    loans = load_loans()
    record = train_and_select_best(loans)
    model = joblib.load("models/credit_risk_model.joblib")
    return model, record


loans = load_loans()
monthly_returns, mean_returns, cov_matrix = load_market_data()
model, model_record = get_credit_model()

loans = loans.copy()
loans["pd_score"] = model.predict_proba(loans[ALL_FEATURES])[:, 1]
loans["expected_loss"] = expected_loss(loans, loans["pd_score"].to_numpy())

with st.expander("What's real here, and what's illustrative?", expanded=False):
    st.markdown(
        """
| README's original SAS toolset | What this dashboard actually runs |
| --- | --- |
| PROC LOGISTIC, PROC DISCRIM (classification models) | A real trained `scikit-learn` `LogisticRegression`, compared against a `GradientBoostingClassifier` and selected by actual held-out test AUC - never hardcoded |
| PROC HPFOREST, SAS Enterprise Miner (ensemble/predictive workflow) | The gradient boosting candidate above stands in for the ensemble method; model selection is a small local "mini-AutoML" comparison |
| PROC OPTMODEL (portfolio optimization) | Real mean-variance (Markowitz) optimization via `scipy.optimize` - efficient frontier, max-Sharpe tangency portfolio, and a risk-tolerance-driven allocation via the capital allocation line |
| SAS/ETS, PROC ARIMA (time series / macro modeling) | A synthetic but structured macro cycle (unemployment, GDP growth) feeds the credit model's origination-time features and the stress scenarios |
| Regulatory Capital Calculation | A real (simplified) Basel II/III IRB capital formula - not a placeholder number |
| SAS Visual Analytics (dashboard) | This Streamlit app |
| 2.5M real loan records, real bureau/Bloomberg/FRED data | A **synthetic**, seeded {n_loans:,}-loan portfolio and a synthetic multi-asset return series - modeled on realistic credit-risk and capital-markets structure, not scraped or real institutional data. See `data/generate_synthetic_loans.py` and `data/generate_synthetic_market_data.py` for full disclosure. |
        """.format(n_loans=len(loans))
    )

tab_overview, tab_credit, tab_portfolio, tab_stress = st.tabs(
    ["Overview", "Credit Risk", "Portfolio Optimization", "Stress Testing"]
)

# ---------------------------------------------------------------- Overview
with tab_overview:
    st.subheader("Loan book snapshot")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total loan book", f"${loans['loan_amount'].sum() / 1e9:.2f}B")
    c2.metric("Number of loans", f"{len(loans):,}")
    c3.metric("Realized default rate", f"{loans[TARGET].mean():.2%}")
    c4.metric("Avg credit score", f"{loans['credit_score'].mean():.0f}")
    c5.metric("Model-predicted portfolio PD", f"{loans['pd_score'].mean():.2%}")

    col1, col2 = st.columns(2)
    with col1:
        by_purpose = loans.groupby("loan_purpose")["loan_amount"].sum().sort_values(ascending=False)
        st.plotly_chart(px.bar(by_purpose, title="Loan book by purpose", labels={"value": "Total loan amount", "loan_purpose": ""}), use_container_width=True)
    with col2:
        by_quarter = loans.groupby("origination_quarter")["loan_amount"].sum().sort_index()
        st.plotly_chart(px.line(by_quarter, title="Origination volume by quarter", labels={"value": "Loan amount", "origination_quarter": ""}), use_container_width=True)

    st.subheader("Investment portfolio snapshot (current allocation)")
    current_ret, current_vol = portfolio_performance(CURRENT_ALLOCATION.to_numpy(), mean_returns, cov_matrix)
    c1, c2, c3 = st.columns(3)
    c1.metric("Expected return", f"{current_ret:.2%}")
    c2.metric("Volatility", f"{current_vol:.2%}")
    c3.metric("Sharpe ratio", f"{(current_ret - float(mean_returns['Cash'])) / current_vol:.2f}")
    st.plotly_chart(px.pie(values=CURRENT_ALLOCATION, names=CURRENT_ALLOCATION.index, title="Current allocation"), use_container_width=True)

# ------------------------------------------------------------- Credit Risk
with tab_credit:
    st.subheader("Model performance")
    st.caption(f"Selected model: **{model_record['selected_model']}** - trained on {model_record['n_train_rows']:,} loans, evaluated on {model_record['n_test_rows']:,} held out.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test AUC", f"{model_record['test_auc']:.4f}")
    calib_gap = abs(loans['pd_score'].mean() - loans[TARGET].mean())
    c2.metric("Calibration gap", f"{calib_gap:.2%}", help="Model's avg predicted PD vs the realized default rate - should be small.")
    c3.metric("Total expected loss", f"${loans['expected_loss'].sum() / 1e6:.1f}M", help="Sum of PD x LGD x loan amount across the book.")
    reg_cap = portfolio_regulatory_capital(loans, loans["pd_score"].to_numpy())
    c4.metric("Required regulatory capital", f"${reg_cap['required_capital'] / 1e6:.1f}M", help="Simplified Basel II/III IRB capital requirement (8% minimum ratio).")

    col1, col2 = st.columns(2)
    with col1:
        roc = model_record["roc_curve"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", name="Model"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash")))
        fig.update_layout(title=f"ROC curve (AUC = {model_record['test_auc']:.3f})", xaxis_title="False positive rate", yaxis_title="True positive rate")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fi = pd.DataFrame(model_record["feature_importance"]).head(10)
        st.plotly_chart(px.bar(fi, x="importance", y="feature", orientation="h", title="Top feature importance").update_layout(yaxis={"categoryorder": "total ascending"}), use_container_width=True)

    st.subheader("Portfolio risk breakdown")
    col1, col2 = st.columns(2)
    with col1:
        by_purpose = loans.groupby("loan_purpose").agg(avg_pd=("pd_score", "mean"), expected_loss=("expected_loss", "sum"))
        st.plotly_chart(px.bar(by_purpose, y="avg_pd", title="Avg predicted PD by loan purpose"), use_container_width=True)
    with col2:
        loans["credit_score_band"] = pd.cut(loans["credit_score"], bins=[300, 580, 640, 680, 720, 760, 850], labels=["<580", "580-640", "640-680", "680-720", "720-760", "760+"])
        by_band = loans.groupby("credit_score_band", observed=True)["pd_score"].mean()
        st.plotly_chart(px.bar(by_band, title="Avg predicted PD by credit score band"), use_container_width=True)

    st.subheader("Score a hypothetical loan")
    st.caption("Interactive: adjust a borrower's characteristics and get a live PD / expected loss prediction from the trained model.")
    c1, c2, c3 = st.columns(3)
    with c1:
        in_purpose = st.selectbox("Loan purpose", loans["loan_purpose"].unique())
        in_amount = st.number_input("Loan amount ($)", min_value=1000, value=25000, step=1000)
        in_term = st.selectbox("Term (months)", sorted(loans["term_months"].unique()))
        in_secured = in_purpose in ("auto", "mortgage")
    with c2:
        in_score = st.slider("Credit score", 300, 850, 680)
        in_income = st.number_input("Annual income ($)", min_value=10000, value=65000, step=5000)
        in_dti = st.slider("DTI ratio (%)", 0.0, 60.0, 25.0)
        in_ltv = st.slider("LTV ratio (%)", 0.0, 125.0, 75.0) if in_secured else 0.0
    with c3:
        in_delinq = st.number_input("Delinquencies (past 2 yrs)", min_value=0, value=0)
        in_emp_years = st.number_input("Employment length (years)", min_value=0, value=5)
        in_home = st.selectbox("Home ownership", ["own", "mortgage", "rent"])
        in_region = st.selectbox("Region", ["Northeast", "Midwest", "South", "West"])
        in_industry = st.selectbox("Industry", loans["industry_sector"].unique())

    hypothetical = pd.DataFrame([{
        "loan_amount": in_amount, "term_months": in_term,
        "interest_rate": float(loans["interest_rate"].median()),
        "credit_score": in_score, "annual_income": in_income, "dti_ratio": in_dti,
        "ltv_ratio": in_ltv, "secured": in_secured, "employment_length_years": in_emp_years,
        "delinquency_2yrs": in_delinq, "home_ownership": in_home, "region": in_region,
        "industry_sector": in_industry, "loan_purpose": in_purpose,
        "unemployment_rate_at_origination": float(loans["unemployment_rate_at_origination"].iloc[-1]),
        "gdp_growth_at_origination": float(loans["gdp_growth_at_origination"].iloc[-1]),
    }])
    hyp_pd = float(model.predict_proba(hypothetical[ALL_FEATURES])[:, 1][0])
    lgd_lookup = {"auto": 0.35, "mortgage": 0.25, "personal": 0.65, "small_business": 0.55, "credit_card": 0.80}
    hyp_el = hyp_pd * lgd_lookup[in_purpose] * in_amount

    c1, c2 = st.columns(2)
    c1.metric("Predicted probability of default", f"{hyp_pd:.2%}")
    c2.metric("Expected loss", f"${hyp_el:,.0f}")

# ------------------------------------------------------- Portfolio Optimization
with tab_portfolio:
    st.subheader("Efficient frontier")
    risky_mean, risky_cov = mean_returns[RISKY_ASSETS], cov_matrix.loc[RISKY_ASSETS, RISKY_ASSETS]
    frontier = efficient_frontier(risky_mean, risky_cov)

    aum = st.number_input("Total investable assets (AUM, $)", min_value=1_000_000, value=500_000_000, step=10_000_000, format="%d")

    st.markdown("**Risk tolerance**")
    risk_tolerance = st.slider("1 = conservative, 10 = aggressive", 1, 10, 5)
    risk_aversion = float(np.interp(risk_tolerance, [1, 10], [35, 2]))
    w_optimal = risk_tolerance_portfolio(mean_returns, cov_matrix, risk_aversion)
    opt_ret, opt_vol = portfolio_performance(w_optimal, mean_returns, cov_matrix)
    opt_weights = weights_series(w_optimal, mean_returns)

    tangency_w = max_sharpe_portfolio(risky_mean, risky_cov, risk_free_rate=float(mean_returns["Cash"]))
    tan_ret, tan_vol = portfolio_performance(tangency_w, risky_mean, risky_cov)
    cash_ret = float(mean_returns["Cash"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frontier["volatility"], y=frontier["return"], mode="lines", name="Efficient frontier (risky assets)"))
    # Capital allocation line: every cash/tangency-portfolio mix lies on
    # this straight line - it's why "optimized" points sit off the
    # curved risky-only frontier rather than on it once cash is blended in.
    cal_vol = [0, tan_vol * 1.3]
    cal_ret = [cash_ret, cash_ret + (tan_ret - cash_ret) / tan_vol * tan_vol * 1.3]
    fig.add_trace(go.Scatter(x=cal_vol, y=cal_ret, mode="lines", line=dict(dash="dash", color="gray"), name="Capital allocation line"))
    fig.add_trace(go.Scatter(x=[0], y=[cash_ret], mode="markers", marker=dict(size=10, symbol="diamond"), name="Cash (risk-free)"))
    fig.add_trace(go.Scatter(x=[tan_vol], y=[tan_ret], mode="markers", marker=dict(size=12, symbol="star"), name="Tangency portfolio (max Sharpe)"))
    fig.add_trace(go.Scatter(x=[current_vol], y=[current_ret], mode="markers", marker=dict(size=14, symbol="x"), name="Current allocation"))
    fig.add_trace(go.Scatter(x=[opt_vol], y=[opt_ret], mode="markers", marker=dict(size=14), name=f"Optimized (risk tolerance {risk_tolerance})"))
    fig.update_layout(
        title="Efficient frontier, capital allocation line, and current vs optimized allocations",
        xaxis_title="Volatility", yaxis_title="Expected return", xaxis_tickformat=".0%", yaxis_tickformat=".0%",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "The optimized point sits on the capital allocation line (a cash + tangency-portfolio mix), not on the "
        "risky-assets-only frontier curve - moving the risk tolerance slider slides it along that line."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Current allocation**")
        st.plotly_chart(px.pie(values=CURRENT_ALLOCATION, names=CURRENT_ALLOCATION.index), use_container_width=True)
    with col2:
        st.markdown(f"**Optimized allocation (risk tolerance {risk_tolerance})**")
        st.plotly_chart(px.pie(values=opt_weights, names=opt_weights.index), use_container_width=True)

    st.subheader("Current vs optimized")
    comparison = pd.DataFrame({
        "Current": {"Expected return": current_ret, "Volatility": current_vol, "Sharpe": (current_ret - float(mean_returns["Cash"])) / current_vol, "Expected annual $ return": current_ret * aum},
        "Optimized": {"Expected return": opt_ret, "Volatility": opt_vol, "Sharpe": (opt_ret - float(mean_returns["Cash"])) / opt_vol, "Expected annual $ return": opt_ret * aum},
    }).T
    st.dataframe(comparison.style.format({"Expected return": "{:.2%}", "Volatility": "{:.2%}", "Sharpe": "{:.2f}", "Expected annual $ return": "${:,.0f}"}), use_container_width=True)

# ----------------------------------------------------------- Stress Testing
with tab_stress:
    st.subheader("Scenario stress test")
    scenario_name = st.selectbox("Scenario", list(SCENARIOS.keys()), index=1)
    scenario = SCENARIOS[scenario_name]
    st.caption(f"Unemployment +{scenario['unemployment_shock_pp']}pp, GDP growth {scenario['gdp_growth_shock_pp']:+.1f}pp")

    credit_stress = credit_stress_summary(loans, model, scenario_name)
    reg_cap_baseline = portfolio_regulatory_capital(loans, loans["pd_score"].to_numpy())
    stressed_pd = model.predict_proba(
        loans.assign(
            unemployment_rate_at_origination=lambda d: (d["unemployment_rate_at_origination"] + scenario["unemployment_shock_pp"]).clip(upper=25.0),
            gdp_growth_at_origination=lambda d: d["gdp_growth_at_origination"] + scenario["gdp_growth_shock_pp"],
        )[ALL_FEATURES]
    )[:, 1]
    reg_cap_stressed = portfolio_regulatory_capital(loans, stressed_pd)

    st.markdown("**Credit book impact**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg PD", f"{credit_stress['stressed_avg_pd']:.2%}", f"{credit_stress['stressed_avg_pd'] - credit_stress['baseline_avg_pd']:+.2%}", delta_color="inverse")
    c2.metric("Expected loss", f"${credit_stress['stressed_expected_loss'] / 1e6:.1f}M", f"{credit_stress['expected_loss_increase_pct']:+.1%}", delta_color="inverse")
    c3.metric("Required capital", f"${reg_cap_stressed['required_capital'] / 1e6:.1f}M", fmt_currency_delta(reg_cap_stressed['required_capital'] - reg_cap_baseline['required_capital'], unit=1e6, suffix="M"), delta_color="inverse")
    c4.metric("Avg risk weight", f"{reg_cap_stressed['avg_risk_weight']:.1%}", f"{reg_cap_stressed['avg_risk_weight'] - reg_cap_baseline['avg_risk_weight']:+.1%}", delta_color="inverse")

    st.markdown("**Investment portfolio impact**")
    portfolio_stress_current = portfolio_stress_summary(CURRENT_ALLOCATION, scenario_name)
    portfolio_stress_optimized = portfolio_stress_summary(opt_weights, scenario_name)
    c1, c2 = st.columns(2)
    c1.metric("Current allocation impact", f"{portfolio_stress_current['portfolio_return_impact']:+.2%}", fmt_currency_delta(portfolio_stress_current['portfolio_return_impact'] * aum))
    c2.metric("Optimized allocation impact", f"{portfolio_stress_optimized['portfolio_return_impact']:+.2%}", fmt_currency_delta(portfolio_stress_optimized['portfolio_return_impact'] * aum))

    per_asset = portfolio_stress_current["per_asset_contribution"].copy()
    per_asset["allocation"] = "Current"
    per_asset_opt = portfolio_stress_optimized["per_asset_contribution"].copy()
    per_asset_opt["allocation"] = "Optimized"
    combined = pd.concat([per_asset, per_asset_opt]).reset_index(names="asset")
    st.plotly_chart(
        px.bar(combined, x="asset", y="contribution", color="allocation", barmode="group", title="Per-asset contribution to portfolio stress impact"),
        use_container_width=True,
    )
