"""Pydantic request/response models for the FastAPI service.

Field names intentionally mirror the raw `customers.csv` columns (minus the
label) so a caller can hand the API a record straight from an upstream CRM
export without a translation layer.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CustomerRecord(BaseModel):
    customer_id: str = Field(..., examples=["CUS-100123"])
    tenure_months: int = Field(..., ge=0, le=100)
    contract: Literal["Month-to-month", "One year", "Two year"]
    internet_service: Literal["DSL", "Fiber optic", "No"]
    payment_method: Literal["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
    paperless_billing: Literal["Yes", "No"]
    tech_support: Literal["Yes", "No", "No internet service"]
    online_security: Literal["Yes", "No", "No internet service"]
    senior_citizen: Literal[0, 1]
    partner: Literal["Yes", "No"]
    dependents: Literal["Yes", "No"]
    monthly_charges: float = Field(..., ge=0)
    total_charges: float = Field(..., ge=0)
    addon_count: int = Field(..., ge=0, le=20)
    support_calls_last_90d: int = Field(..., ge=0, le=50)
    unresolved_ticket_last_90d: Literal[0, 1]


class ChurnPrediction(BaseModel):
    customer_id: str
    churn_probability: float = Field(..., ge=0, le=1)
    risk_tier: Literal["low", "medium", "high"]
    top_risk_factors: list[str]


class BatchPredictRequest(BaseModel):
    customers: list[CustomerRecord]


class BatchPredictResponse(BaseModel):
    predictions: list[ChurnPrediction]
    model_version: str


class SegmentRow(BaseModel):
    segment: str
    n: int
    churn_rate: float


class ModelInfo(BaseModel):
    model_version: str
    trained_at_unix: int
    feature_count: int
    test_roc_auc: float
    test_precision: float
    test_recall: float
