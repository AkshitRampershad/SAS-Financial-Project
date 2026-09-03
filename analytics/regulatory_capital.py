"""
Simplified Basel II/III IRB (Internal Ratings-Based) regulatory capital
calculation - the standard formula banks use to convert a portfolio's
PD/LGD/EAD into required regulatory capital, matching this project's
README methodology step "Regulatory Capital Calculation."

Simplified deliberately in one place: the Basel retail/corporate
formulas use a PD-dependent asset correlation curve; this uses a single
fixed correlation instead (documented below). Everything else - the
unexpected-loss capital formula itself, the RWA scaling, the 8% minimum
capital ratio - is the real, standard Basel formula, not an invented
approximation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

# Basel's actual retail/corporate correlation is PD-dependent (roughly
# 0.03-0.24 depending on exposure type and PD); this uses a single fixed
# value near the middle of that range as an illustrative simplification.
DEFAULT_ASSET_CORRELATION = 0.12
MIN_CAPITAL_RATIO = 0.08  # Basel III minimum total capital ratio


def irb_capital_requirement(pd_: np.ndarray, lgd: np.ndarray, correlation: float = DEFAULT_ASSET_CORRELATION) -> np.ndarray:
    """Per-unit-exposure unexpected-loss capital requirement K, from the
    Basel IRB formula:

        K = LGD * [ N( (N^-1(PD) + sqrt(R) * N^-1(0.999)) / sqrt(1-R) ) - PD ]

    where N is the standard normal CDF. PD is clipped away from exactly
    0 or 1 since N^-1 is undefined there.
    """
    pd_clipped = np.clip(pd_, 1e-6, 1 - 1e-6)
    r = correlation
    inner = (norm.ppf(pd_clipped) + np.sqrt(r) * norm.ppf(0.999)) / np.sqrt(1 - r)
    k = lgd * (norm.cdf(inner) - pd_clipped)
    return np.clip(k, 0.0, None)


def portfolio_regulatory_capital(loans: pd.DataFrame, pd_scores: np.ndarray, correlation: float = DEFAULT_ASSET_CORRELATION) -> dict:
    """Portfolio-level Risk-Weighted Assets and required capital, using
    each loan's own EAD (approximated as outstanding loan_amount) and
    LGD alongside its (possibly stressed) PD.
    """
    ead = loans["loan_amount"].to_numpy()
    lgd = loans["lgd"].to_numpy()
    k = irb_capital_requirement(pd_scores, lgd, correlation)

    capital_required = k * ead
    rwa = capital_required / MIN_CAPITAL_RATIO

    return {
        "total_ead": float(ead.sum()),
        "total_rwa": float(rwa.sum()),
        "required_capital": float(capital_required.sum()),
        "capital_to_ead_ratio": float(capital_required.sum() / ead.sum()) if ead.sum() else 0.0,
        "avg_risk_weight": float(rwa.sum() / ead.sum()) if ead.sum() else 0.0,
    }


if __name__ == "__main__":
    import joblib

    from analytics.credit_risk_model import ALL_FEATURES, train_and_select_best
    from data.generate_synthetic_loans import generate_loans

    loans = generate_loans()
    train_and_select_best(loans)
    model = joblib.load("models/credit_risk_model.joblib")
    pd_scores = model.predict_proba(loans[ALL_FEATURES])[:, 1]

    result = portfolio_regulatory_capital(loans, pd_scores)
    print(f"Total EAD: ${result['total_ead']:,.0f}")
    print(f"Total RWA: ${result['total_rwa']:,.0f}")
    print(f"Required capital (8% minimum): ${result['required_capital']:,.0f}")
    print(f"Avg risk weight: {result['avg_risk_weight']:.1%}")
