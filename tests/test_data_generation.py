from data.generate_synthetic_loans import generate_loans
from data.generate_synthetic_market_data import ASSET_CLASSES, analytic_expected_returns, generate_monthly_returns


def test_loan_generation_shape_and_determinism():
    df1 = generate_loans(n_loans=5000, seed=42)
    df2 = generate_loans(n_loans=5000, seed=42)
    assert len(df1) == 5000
    assert df1.equals(df2)  # seeded, must be reproducible


def test_loan_default_rate_is_realistic():
    df = generate_loans(n_loans=20000, seed=42)
    rate = df["default_flag"].mean()
    assert 0.02 < rate < 0.10  # a plausible mixed retail/commercial book, not degenerate


def test_credit_score_is_monotonically_predictive():
    """Higher credit score bands should show a lower default rate - the
    generator is supposed to produce genuinely learnable signal, not
    pure noise.
    """
    import pandas as pd

    df = generate_loans(n_loans=20000, seed=42)
    df["band"] = pd.qcut(df["credit_score"], 5)
    rates = df.groupby("band", observed=True)["default_flag"].mean().to_numpy()
    assert all(earlier >= later for earlier, later in zip(rates, rates[1:])), (
        f"Expected monotonically decreasing default rate by credit score band, got {rates}"
    )


def test_loan_data_has_injected_quality_issues():
    df = generate_loans(n_loans=20000, seed=42)
    assert df["annual_income"].isna().sum() > 0
    assert df["dti_ratio"].isna().sum() > 0


def test_market_data_shape():
    returns = generate_monthly_returns(n_months=60, seed=42)
    assert returns.shape == (60, len(ASSET_CLASSES))
    assert list(returns.columns) == ASSET_CLASSES


def test_market_covariance_is_positive_semi_definite():
    """A degenerate/invalid covariance matrix would break the
    optimizer - this is a real, not merely cosmetic, invariant.
    """
    import numpy as np

    returns = generate_monthly_returns(n_months=120, seed=42)
    cov = returns.cov().to_numpy()
    eigenvalues = np.linalg.eigvalsh(cov)
    assert (eigenvalues >= -1e-8).all(), f"Covariance matrix has negative eigenvalues: {eigenvalues}"


def test_analytic_expected_returns_in_plausible_ranges():
    er = analytic_expected_returns()
    assert 0.05 < er["US Large Cap Equity"] < 0.15
    assert 0.0 < er["US Investment Grade Bonds"] < 0.08
    assert 0.0 < er["Cash"] < 0.06
    assert er["US Small Cap Equity"] > er["US Investment Grade Bonds"]  # equities should beat bonds on this dimension
