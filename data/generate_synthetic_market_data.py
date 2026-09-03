"""
Synthetic multi-asset-class return generator for portfolio optimization.

NOTE ON DATA PROVENANCE: These are SYNTHETIC monthly returns produced by
a 3-factor model (equity, rate, inflation/commodity factors + asset-
specific idiosyncratic noise), with factor loadings and volatilities
chosen so each asset class's simulated long-run annualized return and
volatility land in a plausible, illustrative range. This is NOT real
market data pulled from Bloomberg or any live feed (no live market data
API is reachable from this environment) - it exists so the portfolio
optimizer has a real, non-degenerate covariance matrix (a genuine
empirical covariance from simulated returns, not a hand-typed one) to
optimize against end-to-end.

Swap this module out for a real historical-returns download (e.g. from
a market data provider) to run the same optimizer on real data - the
optimizer only needs a returns DataFrame with these column names.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ASSET_CLASSES = [
    "US Large Cap Equity",
    "US Small Cap Equity",
    "Intl Developed Equity",
    "Emerging Markets Equity",
    "US Investment Grade Bonds",
    "US High Yield Bonds",
    "US Treasuries",
    "REITs",
    "Commodities",
    "Cash",
]

# (beta_equity, beta_rate, beta_inflation, monthly_idiosyncratic_vol)
_FACTOR_LOADINGS = {
    "US Large Cap Equity":        (1.00, -0.10,  0.05, 0.030),
    "US Small Cap Equity":        (1.25, -0.15,  0.05, 0.042),
    "Intl Developed Equity":      (0.90, -0.05,  0.10, 0.038),
    "Emerging Markets Equity":    (1.10,  0.00,  0.20, 0.055),
    "US Investment Grade Bonds":  (-0.05, 0.80, -0.10, 0.011),
    "US High Yield Bonds":        (0.30,  0.40, -0.05, 0.019),
    "US Treasuries":              (-0.15, 1.00, -0.15, 0.014),
    "REITs":                      (0.70, -0.30,  0.10, 0.043),
    "Commodities":                (0.10, -0.05,  0.70, 0.048),
    "Cash":                       (0.00,  0.00,  0.00, 0.001),
}

# Monthly factor means/vols, chosen so the resulting asset-level
# annualized figures land in a plausible illustrative range.
_FACTOR_MEANS = {"equity": 0.0072, "rate": 0.0018, "inflation": 0.0020}
_FACTOR_VOLS = {"equity": 0.042, "rate": 0.014, "inflation": 0.022}

# A fixed-income asset's expected return is mostly its running yield
# (carry), not just its price sensitivity to the rate factor - factor
# exposure alone understates bond/cash returns. Annual base drift added
# on top of each asset's factor exposure; 0 where factor exposure
# already captures the return (equities, REITs).
_BASE_DRIFT_ANNUAL = {
    "US Large Cap Equity": 0.0,
    "US Small Cap Equity": 0.0,
    "Intl Developed Equity": 0.0,
    "Emerging Markets Equity": 0.0,
    "US Investment Grade Bonds": 0.030,
    "US High Yield Bonds": 0.018,
    "US Treasuries": 0.024,
    "REITs": 0.0,
    "Commodities": 0.018,
    "Cash": 0.027,
}


def generate_monthly_returns(n_months: int = 120, seed: int = 42) -> pd.DataFrame:
    """Simulate n_months of monthly returns for each asset class via a
    3-factor model. Returns a DataFrame indexed by month with one
    column per asset class.
    """
    rng = np.random.default_rng(seed)

    equity_factor = rng.normal(_FACTOR_MEANS["equity"], _FACTOR_VOLS["equity"], n_months)
    rate_factor = rng.normal(_FACTOR_MEANS["rate"], _FACTOR_VOLS["rate"], n_months)
    inflation_factor = rng.normal(_FACTOR_MEANS["inflation"], _FACTOR_VOLS["inflation"], n_months)

    data = {}
    for asset, (b_eq, b_rate, b_infl, idio_vol) in _FACTOR_LOADINGS.items():
        idio = rng.normal(0, idio_vol, n_months)
        drift = _BASE_DRIFT_ANNUAL[asset] / 12
        data[asset] = b_eq * equity_factor + b_rate * rate_factor + b_infl * inflation_factor + idio + drift

    dates = pd.period_range(end=pd.Period.now("M"), periods=n_months, freq="M")
    return pd.DataFrame(data, index=dates)[ASSET_CLASSES]


def analytic_expected_returns() -> pd.Series:
    """Forward-looking annualized expected return per asset class,
    computed in closed form from the factor model's own parameters
    (beta_i . factor_mean + drift, annualized) rather than a trailing
    sample mean.

    This is standard practice in real portfolio construction, not a
    shortcut: with realistic asset volatility (15-25%/yr) and only a
    decade of monthly data, a trailing sample mean's standard error is
    large enough to swamp the true signal (a well-known result - it's
    why institutions use separate capital-market assumptions for
    expected return rather than relying on noisy historical averages).
    The covariance matrix, by contrast, IS estimated from the simulated
    data below, since covariance converges far faster than the mean.
    """
    rows = {}
    for asset, (b_eq, b_rate, b_infl, _idio_vol) in _FACTOR_LOADINGS.items():
        monthly = (
            b_eq * _FACTOR_MEANS["equity"]
            + b_rate * _FACTOR_MEANS["rate"]
            + b_infl * _FACTOR_MEANS["inflation"]
            + _BASE_DRIFT_ANNUAL[asset] / 12
        )
        rows[asset] = (1 + monthly) ** 12 - 1
    return pd.Series(rows, name="expected_return")[ASSET_CLASSES]


def summarize_assets(monthly_returns: pd.DataFrame) -> pd.DataFrame:
    """Per asset class: the analytic (forward-looking) expected return
    assumption, alongside volatility and the realized trailing return -
    both estimated from the simulated data. Showing both side by side
    is deliberate: it's the same "assumption vs. realized" comparison a
    real institution's risk committee would look at.
    """
    ann_vol = monthly_returns.std() * np.sqrt(12)
    realized_return = (1 + monthly_returns.mean()) ** 12 - 1
    expected_return = analytic_expected_returns()
    return pd.DataFrame({
        "expected_return": expected_return,
        "realized_return": realized_return,
        "volatility": ann_vol,
        "sharpe": expected_return / ann_vol,
    })


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", type=int, default=120)
    parser.add_argument("--out", type=str, default="data/raw/market_returns.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    returns = generate_monthly_returns(n_months=args.months)
    returns.to_csv(args.out)
    print(f"Wrote {len(returns)} months of synthetic returns for {len(ASSET_CLASSES)} asset classes to {args.out}")
    print(summarize_assets(returns).round(4))
