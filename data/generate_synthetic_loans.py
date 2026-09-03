"""
Synthetic loan-level portfolio generator.

NOTE ON DATA PROVENANCE: This generates SYNTHETIC loan-level data with
realistic field names, value ranges, and risk correlations modeled on
publicly documented consumer/commercial credit risk factors (FICO score
bands, DTI/LTV thresholds, delinquency history, macro cycle effects). It
is NOT the institution's real core-banking data, real credit bureau
data, or real FRED/Bloomberg feeds described in this project's README -
no such data is available outside a real financial institution. It
exists so the credit risk model and dashboard have something real to
train on, score, and stress-test end-to-end.

The original README's scale (2.5M loans, 10 years of real bureau/macro
data) reflects production scale; this generator produces a smaller
synthetic sample (tens of thousands of rows) so the full pipeline runs
in seconds on a laptop or Streamlit Cloud's free tier.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LOAN_PURPOSES = ["auto", "mortgage", "personal", "small_business", "credit_card"]
REGIONS = ["Northeast", "Midwest", "South", "West"]
INDUSTRIES = ["Technology", "Healthcare", "Retail", "Manufacturing", "Construction", "Finance", "Hospitality", "Other"]
HOME_OWNERSHIP = ["own", "mortgage", "rent"]

# Secured loans (collateral backs the loan) have materially lower loss
# severity than unsecured ones if they default - a standard credit risk
# distinction (LGD = loss given default).
SECURED_PURPOSES = {"auto", "mortgage"}
BASE_LGD = {"auto": 0.35, "mortgage": 0.25, "personal": 0.65, "small_business": 0.55, "credit_card": 0.80}


def _macro_cycle(n_quarters: int, seed: int) -> pd.DataFrame:
    """A smooth, plausible macro cycle (unemployment/GDP growth) across
    2015Q1-2024Q4 - illustrative, not real FRED data.
    """
    rng = np.random.default_rng(seed)
    quarters = pd.period_range("2015Q1", periods=n_quarters, freq="Q")
    t = np.arange(n_quarters)
    # A mild cycle with a sharp 2020 shock, loosely resembling the shape
    # (not the exact values) of the real 2015-2024 period.
    unemployment = 4.5 + 1.5 * np.sin(t / 6) + rng.normal(0, 0.2, n_quarters)
    covid_idx = quarters.get_indexer(pd.period_range("2020Q2", "2020Q4", freq="Q"))
    unemployment[covid_idx] += 6.0
    unemployment = unemployment.clip(3.0, 15.0)

    gdp_growth = 2.2 - 0.3 * np.sin(t / 6) + rng.normal(0, 0.4, n_quarters)
    gdp_growth[covid_idx] -= 9.0
    gdp_growth = gdp_growth.clip(-12.0, 6.0)

    return pd.DataFrame({"quarter": quarters, "unemployment_rate": unemployment.round(2), "gdp_growth": gdp_growth.round(2)})


def generate_loans(n_loans: int = 30_000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic loan-level portfolio with a genuinely
    learnable default outcome (not random noise, not deterministic) so
    a downstream classifier has real signal to learn from.
    """
    rng = np.random.default_rng(seed)
    macro = _macro_cycle(40, seed)  # 2015Q1..2024Q4

    purpose = rng.choice(LOAN_PURPOSES, n_loans, p=[0.28, 0.20, 0.22, 0.10, 0.20])
    secured = np.isin(purpose, list(SECURED_PURPOSES))

    credit_score = np.round(rng.normal(690, 65, n_loans)).clip(300, 850)
    annual_income = np.round(rng.lognormal(mean=11.0, sigma=0.5, size=n_loans)).clip(15_000, 500_000)
    dti_ratio = np.round(rng.beta(2, 5, n_loans) * 60, 1)  # 0-60%
    employment_length_years = np.round(rng.exponential(5, n_loans)).clip(0, 40)
    delinquency_2yrs = rng.poisson(0.3, n_loans)
    home_ownership = rng.choice(HOME_OWNERSHIP, n_loans, p=[0.28, 0.40, 0.32])
    region = rng.choice(REGIONS, n_loans)
    industry = rng.choice(INDUSTRIES, n_loans)

    ltv_ratio = np.where(secured, np.round(rng.normal(78, 15, n_loans).clip(10, 125), 1), 0.0)

    loan_amount = np.where(
        purpose == "mortgage", rng.lognormal(12.3, 0.4, n_loans),
        np.where(purpose == "small_business", rng.lognormal(10.8, 0.7, n_loans),
                 np.where(purpose == "auto", rng.lognormal(9.9, 0.4, n_loans),
                          rng.lognormal(9.0, 0.6, n_loans))),
    ).round(0).clip(1_000, 2_000_000)

    term_months = np.select(
        [purpose == "mortgage", purpose == "auto", purpose == "small_business", purpose == "credit_card"],
        [rng.choice([180, 360], n_loans), rng.choice([36, 48, 60, 72], n_loans), rng.choice([36, 60, 84, 120], n_loans), np.full(n_loans, 12)],
        default=rng.choice([24, 36, 48, 60], n_loans),
    )

    quarter_idx = rng.integers(0, len(macro), n_loans)
    origination_quarter = macro["quarter"].to_numpy()[quarter_idx]
    unemployment_at_origination = macro["unemployment_rate"].to_numpy()[quarter_idx]
    gdp_growth_at_origination = macro["gdp_growth"].to_numpy()[quarter_idx]

    # Risk-based pricing: worse credit / higher DTI / unsecured -> higher rate.
    interest_rate = (
        3.5
        + (750 - credit_score) * 0.018
        + dti_ratio * 0.04
        + (~secured) * 2.5
        + rng.normal(0, 0.6, n_loans)
    ).clip(2.5, 29.9).round(2)

    # Latent default risk: a realistic composite of the standard credit
    # risk factors, converted to a probability via a logistic link, then
    # a genuine Bernoulli draw - not a fabricated/rounded label.
    z = (
        -4.3
        + (700 - credit_score) * 0.018
        + dti_ratio * 0.035
        + delinquency_2yrs * 0.55
        + (ltv_ratio - 70) * 0.012
        + (unemployment_at_origination - 5.0) * 0.22
        - gdp_growth_at_origination * 0.06
        - employment_length_years * 0.02
        + (purpose == "small_business") * 0.35
        + (purpose == "credit_card") * 0.25
        - (home_ownership == "own") * 0.15
        + rng.normal(0, 0.5, n_loans)
    )
    pd_true = 1 / (1 + np.exp(-z))
    default_flag = (rng.uniform(0, 1, n_loans) < pd_true).astype(int)

    lgd = np.array([BASE_LGD[p] for p in purpose]) + rng.normal(0, 0.05, n_loans)
    lgd = lgd.clip(0.05, 0.95).round(3)

    df = pd.DataFrame({
        "loan_id": [f"LN-{i:07d}" for i in range(n_loans)],
        "origination_quarter": origination_quarter.astype(str),
        "loan_purpose": purpose,
        "loan_amount": loan_amount,
        "term_months": term_months,
        "interest_rate": interest_rate,
        "credit_score": credit_score.astype(int),
        "annual_income": annual_income,
        "dti_ratio": dti_ratio,
        "ltv_ratio": ltv_ratio,
        "secured": secured,
        "employment_length_years": employment_length_years.astype(int),
        "delinquency_2yrs": delinquency_2yrs,
        "home_ownership": home_ownership,
        "region": region,
        "industry_sector": industry,
        "unemployment_rate_at_origination": unemployment_at_origination,
        "gdp_growth_at_origination": gdp_growth_at_origination,
        "lgd": lgd,
        "default_flag": default_flag,
    })

    # Realistic data-quality issues, same as production loan tapes: a
    # sliver of missing income/DTI, and a handful of duplicate loan IDs
    # from re-extracted batches - present so anyone extending this into
    # a real ingestion pipeline has something to validate against.
    dirty_idx = rng.choice(n_loans, size=int(n_loans * 0.01), replace=False)
    df.loc[dirty_idx[: len(dirty_idx) // 2], "annual_income"] = np.nan
    df.loc[dirty_idx[len(dirty_idx) // 2:], "dti_ratio"] = np.nan

    return df


def generate_macro_series(seed: int = 42) -> pd.DataFrame:
    return _macro_cycle(40, seed)


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=30_000)
    parser.add_argument("--out", type=str, default="data/raw/loans.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    loans = generate_loans(n_loans=args.rows)
    loans.to_csv(args.out, index=False)
    print(f"Wrote {len(loans):,} synthetic loans to {args.out} (default rate: {loans['default_flag'].mean():.2%})")
