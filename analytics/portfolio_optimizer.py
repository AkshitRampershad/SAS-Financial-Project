"""
Mean-variance portfolio optimization (Markowitz), standing in for
SAS/OR's PROC OPTMODEL, using scipy.optimize's SLSQP solver.

Given a set of expected returns and a covariance matrix (see
data/generate_synthetic_market_data.py), computes:
- the minimum-variance portfolio for a target return (a point on the
  efficient frontier)
- the full efficient frontier
- the maximum-Sharpe-ratio portfolio
- a risk-tolerance-driven allocation (mean-variance utility
  maximization), which is what the dashboard's risk slider drives

All portfolios are long-only and fully invested (weights sum to 1,
each in [0, max_weight]) unless stated otherwise - the same constraint
set a real institutional mandate would typically impose.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def portfolio_performance(weights: np.ndarray, mean_returns: pd.Series, cov_matrix: pd.DataFrame) -> tuple[float, float]:
    ret = float(np.dot(weights, mean_returns))
    vol = float(np.sqrt(weights @ cov_matrix.to_numpy() @ weights))
    return ret, vol


def _bounds_and_base_constraints(n_assets: int, max_weight: float):
    bounds = tuple((0.0, max_weight) for _ in range(n_assets))
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    return bounds, constraints


def min_variance_for_target_return(
    mean_returns: pd.Series, cov_matrix: pd.DataFrame, target_return: float, max_weight: float = 0.40,
) -> np.ndarray:
    n = len(mean_returns)
    bounds, constraints = _bounds_and_base_constraints(n, max_weight)
    constraints.append({"type": "eq", "fun": lambda w: np.dot(w, mean_returns) - target_return})

    x0 = np.repeat(1 / n, n)
    result = minimize(
        lambda w: w @ cov_matrix.to_numpy() @ w,
        x0, method="SLSQP", bounds=bounds, constraints=constraints,
    )
    if not result.success:
        raise RuntimeError(f"Optimization failed for target_return={target_return}: {result.message}")
    return result.x


def efficient_frontier(mean_returns: pd.Series, cov_matrix: pd.DataFrame, n_points: int = 25, max_weight: float = 0.40) -> pd.DataFrame:
    lo, hi = float(mean_returns.min()), float(mean_returns.max())
    targets = np.linspace(lo + 1e-4, hi - 1e-4, n_points)

    rows = []
    for target in targets:
        try:
            w = min_variance_for_target_return(mean_returns, cov_matrix, target, max_weight)
        except RuntimeError:
            continue
        ret, vol = portfolio_performance(w, mean_returns, cov_matrix)
        rows.append({"target_return": target, "return": ret, "volatility": vol, "weights": w})
    return pd.DataFrame(rows)


def max_sharpe_portfolio(mean_returns: pd.Series, cov_matrix: pd.DataFrame, risk_free_rate: float = 0.025, max_weight: float = 0.40) -> np.ndarray:
    n = len(mean_returns)
    bounds, constraints = _bounds_and_base_constraints(n, max_weight)

    def neg_sharpe(w):
        ret, vol = portfolio_performance(w, mean_returns, cov_matrix)
        return -(ret - risk_free_rate) / vol

    x0 = np.repeat(1 / n, n)
    result = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    if not result.success:
        raise RuntimeError(f"Max-Sharpe optimization failed: {result.message}")
    return result.x


def risk_tolerance_portfolio(
    mean_returns: pd.Series, cov_matrix: pd.DataFrame, risk_aversion: float,
    risk_free_asset: str = "Cash", max_weight: float = 0.40,
) -> np.ndarray:
    """Two-fund (capital allocation line) approach: split between the
    risk-free-like asset and the max-Sharpe ("tangency") portfolio
    computed over the *remaining* risky assets, per Tobin separation.

    Deliberately NOT a single joint mean-variance optimization over all
    assets including cash: a near-zero-volatility asset's Sharpe ratio
    is dominated by its tiny denominator, so a joint optimizer tends to
    load up on it regardless of risk_aversion - a well-known pitfall,
    not a useful "risk tolerance" story. Separating the risk-free asset
    out and only varying the cash/risky-portfolio SPLIT by risk
    aversion is both the textbook-correct approach and what actually
    produces a sensible aggressive-to-conservative slider.
    """
    risky_assets = [a for a in mean_returns.index if a != risk_free_asset]
    risky_mean = mean_returns[risky_assets]
    risky_cov = cov_matrix.loc[risky_assets, risky_assets]
    rf = float(mean_returns[risk_free_asset])

    w_tangency = max_sharpe_portfolio(risky_mean, risky_cov, risk_free_rate=rf, max_weight=max_weight)
    ret_tan, vol_tan = portfolio_performance(w_tangency, risky_mean, risky_cov)

    # Fraction allocated to the risky tangency portfolio; the rest sits
    # in the risk-free asset. Clipped to [0, 1] - no leverage, no
    # short-selling cash to lever up the risky sleeve.
    y = (ret_tan - rf) / (risk_aversion * vol_tan**2)
    y = float(np.clip(y, 0.0, 1.0))

    weights = pd.Series(0.0, index=mean_returns.index)
    weights[risky_assets] = y * w_tangency
    weights[risk_free_asset] = 1 - y
    return weights.to_numpy()


def weights_series(weights: np.ndarray, mean_returns: pd.Series) -> pd.Series:
    return pd.Series(weights, index=mean_returns.index, name="weight").round(4)


if __name__ == "__main__":
    from data.generate_synthetic_market_data import ASSET_CLASSES, analytic_expected_returns, generate_monthly_returns

    monthly = generate_monthly_returns()
    mean_returns = analytic_expected_returns()
    cov_matrix = monthly.cov() * 12  # annualize

    risky_assets = [a for a in ASSET_CLASSES if a != "Cash"]
    risky_mean, risky_cov = mean_returns[risky_assets], cov_matrix.loc[risky_assets, risky_assets]

    w_sharpe = max_sharpe_portfolio(risky_mean, risky_cov, risk_free_rate=float(mean_returns["Cash"]))
    ret, vol = portfolio_performance(w_sharpe, risky_mean, risky_cov)
    print(f"Max-Sharpe portfolio (risky assets only): return={ret:.2%} vol={vol:.2%}")
    print(weights_series(w_sharpe, risky_mean))

    frontier = efficient_frontier(risky_mean, risky_cov)
    print(f"\nEfficient frontier: {len(frontier)} points, return range {frontier['return'].min():.2%}-{frontier['return'].max():.2%}")

    print("\nRisk-tolerance slider (capital allocation line, cash + tangency portfolio):")
    for risk_aversion in [1, 3, 6, 12, 25]:
        w = risk_tolerance_portfolio(mean_returns, cov_matrix, risk_aversion)
        ret, vol = portfolio_performance(w, mean_returns, cov_matrix)
        cash_weight = weights_series(w, mean_returns)["Cash"]
        print(f"  risk_aversion={risk_aversion:>3}: return={ret:.2%} vol={vol:.2%} cash_weight={cash_weight:.1%}")
