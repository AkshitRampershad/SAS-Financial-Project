import numpy as np

from analytics.portfolio_optimizer import (
    efficient_frontier,
    max_sharpe_portfolio,
    min_variance_for_target_return,
    portfolio_performance,
    risk_tolerance_portfolio,
)
from data.generate_synthetic_market_data import ASSET_CLASSES, analytic_expected_returns, generate_monthly_returns

RISKY_ASSETS = [a for a in ASSET_CLASSES if a != "Cash"]


def _market_data():
    monthly = generate_monthly_returns(n_months=120, seed=42)
    mean_returns = analytic_expected_returns()
    cov_matrix = monthly.cov() * 12
    return mean_returns, cov_matrix


def test_max_sharpe_weights_are_valid():
    mean_returns, cov_matrix = _market_data()
    risky_mean, risky_cov = mean_returns[RISKY_ASSETS], cov_matrix.loc[RISKY_ASSETS, RISKY_ASSETS]
    w = max_sharpe_portfolio(risky_mean, risky_cov, risk_free_rate=float(mean_returns["Cash"]))
    assert abs(w.sum() - 1.0) < 1e-6
    assert (w >= -1e-8).all()
    assert (w <= 0.40 + 1e-8).all()


def test_min_variance_hits_target_return():
    mean_returns, cov_matrix = _market_data()
    risky_mean, risky_cov = mean_returns[RISKY_ASSETS], cov_matrix.loc[RISKY_ASSETS, RISKY_ASSETS]
    target = float(risky_mean.median())
    w = min_variance_for_target_return(risky_mean, risky_cov, target)
    ret, _ = portfolio_performance(w, risky_mean, risky_cov)
    assert abs(ret - target) < 1e-4


def test_efficient_frontier_is_monotonic_in_return():
    mean_returns, cov_matrix = _market_data()
    risky_mean, risky_cov = mean_returns[RISKY_ASSETS], cov_matrix.loc[RISKY_ASSETS, RISKY_ASSETS]
    frontier = efficient_frontier(risky_mean, risky_cov, n_points=15)
    returns = frontier["return"].to_numpy()
    assert (np.diff(returns) > 0).all(), "Efficient frontier target returns should be strictly increasing"


def test_risk_tolerance_gradient_is_sensible():
    """More conservative (higher risk_aversion) should mean more cash,
    lower vol, lower return - and it should actually vary, not be flat.
    """
    mean_returns, cov_matrix = _market_data()
    cash_weights, vols = [], []
    for risk_aversion in [2, 10, 20, 35]:
        w = risk_tolerance_portfolio(mean_returns, cov_matrix, risk_aversion)
        assert abs(w.sum() - 1.0) < 1e-6
        assert (w >= -1e-8).all()
        cash_weights.append(w[mean_returns.index.get_loc("Cash")])
        _, vol = portfolio_performance(w, mean_returns, cov_matrix)
        vols.append(vol)

    assert cash_weights[0] <= cash_weights[-1], f"Cash weight should rise with risk aversion: {cash_weights}"
    assert vols[0] >= vols[-1], f"Volatility should fall with risk aversion: {vols}"
    assert cash_weights[0] != cash_weights[-1], "Risk tolerance had no effect on allocation"


def test_no_leverage_at_extreme_low_risk_aversion():
    """At very low risk aversion the investor wants to lever into the
    tangency portfolio, but weights must still respect the no-leverage,
    no-short-cash bound (y clipped to [0, 1]).
    """
    mean_returns, cov_matrix = _market_data()
    w = risk_tolerance_portfolio(mean_returns, cov_matrix, risk_aversion=0.1)
    assert abs(w.sum() - 1.0) < 1e-6
    assert (w >= -1e-8).all()
