import numpy as np

from analytics.credit_risk_model import ALL_FEATURES, TARGET, expected_loss, train_and_select_best
from data.generate_synthetic_loans import generate_loans


def _train():
    loans = generate_loans(n_loans=15000, seed=42)
    record = train_and_select_best(loans, seed=42)
    return loans, record


def test_model_beats_random_guessing():
    _, record = _train()
    assert record["test_auc"] > 0.65, f"AUC too low to be a meaningful model: {record['test_auc']}"


def test_model_is_calibrated_not_just_discriminative():
    """A model that ranks well but wildly overstates the average
    probability isn't fit for Expected Loss = PD x LGD x EAD - this is
    a regression test for the class_weight='balanced' miscalibration
    bug (it inflated mean predicted PD from ~5% to ~35%).
    """
    loans, record = _train()
    candidate_metrics = record["candidates_evaluated"][record["selected_model"]]
    true_rate = loans[TARGET].mean()
    assert abs(candidate_metrics["mean_predicted_pd"] - true_rate) < 0.02, (
        f"Selected model's mean predicted PD ({candidate_metrics['mean_predicted_pd']}) "
        f"is far from the true default rate ({true_rate}) - miscalibrated"
    )


def test_feature_importance_is_populated_and_credit_score_matters():
    _, record = _train()
    assert record["feature_importance"]
    top_features = {f["feature"] for f in record["feature_importance"][:5]}
    assert any("credit_score" in f for f in top_features), (
        f"Expected credit_score among top features, got {top_features}"
    )


def test_expected_loss_formula():
    loans = generate_loans(n_loans=100, seed=1)
    pd_scores = np.full(len(loans), 0.1)
    el = expected_loss(loans, pd_scores)
    expected = 0.1 * loans["lgd"].to_numpy() * loans["loan_amount"].to_numpy()
    assert np.allclose(el.to_numpy(), expected)


def test_run_log_accumulates_across_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    loans = generate_loans(n_loans=3000, seed=1)
    train_and_select_best(loans, seed=1)
    train_and_select_best(loans, seed=2)
    import json
    from pathlib import Path

    log = json.loads(Path("models/run_log.json").read_text())
    assert len(log) == 2
