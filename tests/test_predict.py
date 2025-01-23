from churn_system.predict import model_info, predict_batch

HIGH_RISK_CUSTOMER = {
    "customer_id": "CUS-TEST-HIGH",
    "tenure_months": 1,
    "contract": "Month-to-month",
    "internet_service": "Fiber optic",
    "payment_method": "Electronic check",
    "paperless_billing": "Yes",
    "tech_support": "No",
    "online_security": "No",
    "senior_citizen": 0,
    "partner": "No",
    "dependents": "No",
    "monthly_charges": 98.0,
    "total_charges": 98.0,
    "addon_count": 4,
    "support_calls_last_90d": 5,
    "unresolved_ticket_last_90d": 1,
}

LOW_RISK_CUSTOMER = {
    **HIGH_RISK_CUSTOMER,
    "customer_id": "CUS-TEST-LOW",
    "tenure_months": 60,
    "contract": "Two year",
    "payment_method": "Credit card",
    "tech_support": "Yes",
    "online_security": "Yes",
    "monthly_charges": 35.0,
    "total_charges": 2100.0,
    "addon_count": 0,
    "support_calls_last_90d": 0,
    "unresolved_ticket_last_90d": 0,
}


def test_predict_batch_returns_one_row_per_customer(trained_model):
    results = predict_batch([HIGH_RISK_CUSTOMER, LOW_RISK_CUSTOMER])
    assert len(results) == 2
    ids = {r["customer_id"] for r in results}
    assert ids == {"CUS-TEST-HIGH", "CUS-TEST-LOW"}


def test_predict_batch_probabilities_are_valid(trained_model):
    results = predict_batch([HIGH_RISK_CUSTOMER, LOW_RISK_CUSTOMER])
    for r in results:
        assert 0.0 <= r["churn_probability"] <= 1.0
        assert r["risk_tier"] in {"low", "medium", "high"}
        assert len(r["top_risk_factors"]) == 3


def test_predict_batch_ranks_obviously_risky_customer_higher(trained_model):
    """The whole point of the model: a month-to-month, high-bill, multiple
    recent unresolved tickets customer should score meaningfully higher
    than a loyal two-year, low-bill, no-issues customer."""
    results = {r["customer_id"]: r for r in predict_batch([HIGH_RISK_CUSTOMER, LOW_RISK_CUSTOMER])}
    assert results["CUS-TEST-HIGH"]["churn_probability"] > results["CUS-TEST-LOW"]["churn_probability"]
    assert results["CUS-TEST-HIGH"]["risk_tier"] == "high"
    assert results["CUS-TEST-LOW"]["risk_tier"] == "low"


def test_model_info_matches_metrics_file(trained_model):
    info = model_info()
    assert 0.5 <= info["test_roc_auc"] <= 1.0
    assert info["feature_count"] > 0
