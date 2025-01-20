"""Inference: raw customer record(s) -> churn probability + explanation.

This is the module the FastAPI layer (api.py) calls. It's kept independent
of FastAPI so it can also be called from a batch job (e.g. a nightly script
that scores the whole customer base and writes results back to the
warehouse) without spinning up a web server.
"""

from __future__ import annotations

import functools
import json

import pandas as pd
import shap
import xgboost as xgb

from churn_system.config import PATHS
from churn_system.database import compute_features_inmemory
from churn_system.features import build_inference_frame

RISK_TIER_THRESHOLDS = {"low": 0.2, "medium": 0.5}  # >=0.5 is "high"


@functools.lru_cache(maxsize=1)
def get_model() -> xgb.XGBClassifier:
    if not PATHS.model_file.exists():
        raise FileNotFoundError(
            f"No trained model at {PATHS.model_file}. Run `python -m churn_system.train` first."
        )
    model = xgb.XGBClassifier()
    model.load_model(str(PATHS.model_file))
    return model


@functools.lru_cache(maxsize=1)
def get_explainer() -> shap.TreeExplainer:
    return shap.TreeExplainer(get_model())


def _risk_tier(prob: float) -> str:
    if prob < RISK_TIER_THRESHOLDS["low"]:
        return "low"
    if prob < RISK_TIER_THRESHOLDS["medium"]:
        return "medium"
    return "high"


def _top_factors(shap_row, columns: pd.Index, k: int = 3) -> list[str]:
    """Human-readable top-k SHAP contributors for one prediction, e.g.
    'contract_Month-to-month (+0.41)'."""
    pairs = sorted(zip(columns, shap_row, strict=True), key=lambda p: -abs(p[1]))[:k]
    return [f"{name} ({value:+.2f})" for name, value in pairs]


def predict_batch(records: list[dict]) -> list[dict]:
    """records: list of dicts shaped like schemas.CustomerRecord (minus
    validation -- the API layer validates before calling this)."""
    raw_df = pd.DataFrame(records)
    # The shared view SQL selects `churned` (it's a training-time label
    # column too) -- inference requests obviously don't carry a real label,
    # so stub one in. It's dropped again by build_inference_frame, which
    # only reads the feature manifest's columns.
    raw_df["churned"] = 0
    # Run the exact same SQL the warehouse view uses at training time so a
    # single request gets tenure_bucket / avg_historical_monthly_spend /
    # spend_trend / is_high_support_volume computed identically -- no
    # separate "reimplement the SQL in pandas for the hot path" copy to
    # drift out of sync with sql/schema.sql.
    featured_df = compute_features_inmemory(raw_df)
    X = build_inference_frame(featured_df)

    model = get_model()
    proba = model.predict_proba(X)[:, 1]

    explainer = get_explainer()
    shap_values = explainer(X)

    results = []
    for i, customer_id in enumerate(featured_df["customer_id"]):
        results.append(
            {
                "customer_id": customer_id,
                "churn_probability": round(float(proba[i]), 4),
                "risk_tier": _risk_tier(float(proba[i])),
                "top_risk_factors": _top_factors(shap_values.values[i], X.columns),
            }
        )
    return results


def model_info() -> dict:
    metrics = json.loads(PATHS.metrics_file.read_text())
    test = metrics["metrics"]["test"]
    return {
        "model_version": PATHS.model_file.stem,
        "trained_at_unix": metrics["trained_at_unix"],
        "feature_count": metrics["feature_count"],
        "test_roc_auc": test["roc_auc"],
        "test_precision": test["precision"],
        "test_recall": test["recall"],
    }
