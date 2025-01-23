import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(trained_model):
    from churn_system.api import app

    return TestClient(app)


def test_health_ok_when_model_present(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_model_info_endpoint(client):
    r = client.get("/model/info")
    assert r.status_code == 200
    body = r.json()
    assert "test_roc_auc" in body
    assert body["feature_count"] > 0


def test_predict_endpoint_rejects_empty_list(client):
    r = client.post("/predict", json={"customers": []})
    assert r.status_code == 422


def test_predict_endpoint_rejects_invalid_contract_value(client):
    bad = {
        "customer_id": "CUS-1",
        "tenure_months": 5,
        "contract": "Lifetime",  # not one of the allowed Literal values
        "internet_service": "DSL",
        "payment_method": "Credit card",
        "paperless_billing": "Yes",
        "tech_support": "No",
        "online_security": "No",
        "senior_citizen": 0,
        "partner": "No",
        "dependents": "No",
        "monthly_charges": 50.0,
        "total_charges": 250.0,
        "addon_count": 1,
        "support_calls_last_90d": 0,
        "unresolved_ticket_last_90d": 0,
    }
    r = client.post("/predict", json={"customers": [bad]})
    assert r.status_code == 422


def test_predict_endpoint_happy_path(client):
    good = {
        "customer_id": "CUS-1",
        "tenure_months": 5,
        "contract": "Month-to-month",
        "internet_service": "DSL",
        "payment_method": "Credit card",
        "paperless_billing": "Yes",
        "tech_support": "No",
        "online_security": "No",
        "senior_citizen": 0,
        "partner": "No",
        "dependents": "No",
        "monthly_charges": 50.0,
        "total_charges": 250.0,
        "addon_count": 1,
        "support_calls_last_90d": 0,
        "unresolved_ticket_last_90d": 0,
    }
    r = client.post("/predict", json={"customers": [good]})
    assert r.status_code == 200
    body = r.json()
    assert len(body["predictions"]) == 1
    assert 0.0 <= body["predictions"][0]["churn_probability"] <= 1.0


def test_segments_endpoint_valid_column(client):
    r = client.get("/segments/contract")
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 3
    assert {r["segment"] for r in rows} == {"Month-to-month", "One year", "Two year"}


def test_segments_endpoint_invalid_column(client):
    r = client.get("/segments/favorite_color")
    assert r.status_code == 422
