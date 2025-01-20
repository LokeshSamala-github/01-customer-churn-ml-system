"""FastAPI serving layer.

Run with:
    uvicorn churn_system.api:app --reload

Endpoints:
    GET  /health                liveness probe
    GET  /model/info             model version + headline test-set metrics
    POST /predict                score one or more customers, with SHAP-based
                                  top risk factors per customer
    GET  /segments/{column}      churn rate by segment, straight from DuckDB
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from churn_system import __version__
from churn_system.database import churn_rate_by_segment
from churn_system.predict import get_model, model_info, predict_batch
from churn_system.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    ChurnPrediction,
    ModelInfo,
    SegmentRow,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once at startup instead of on the first request, so the
    # first real request isn't the one that pays for disk I/O + SHAP
    # explainer construction. Don't crash the process if it's missing --
    # /health should report the real problem instead of the container
    # failing to start (e.g. before `python -m churn_system.train` has run).
    try:
        get_model()
        logger.info("model loaded and warm")
    except FileNotFoundError as exc:
        logger.warning("startup warm-up skipped: %s", exc)
    yield


app = FastAPI(
    title="Customer Churn Risk API",
    description="Scores subscription customers for churn risk and explains why, using a DuckDB-backed feature pipeline and an XGBoost model.",
    version=__version__,
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    try:
        get_model()
        return {"status": "ok", "model_loaded": True}
    except FileNotFoundError:
        return JSONResponse(status_code=503, content={"status": "degraded", "model_loaded": False})


@app.get("/model/info", response_model=ModelInfo)
def get_model_info() -> ModelInfo:
    try:
        return ModelInfo(**model_info())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/predict", response_model=BatchPredictResponse)
def predict(request: BatchPredictRequest) -> BatchPredictResponse:
    if not request.customers:
        raise HTTPException(status_code=422, detail="customers list must not be empty")
    try:
        raw = [c.model_dump() for c in request.customers]
        results = predict_batch(raw)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return BatchPredictResponse(
        predictions=[ChurnPrediction(**r) for r in results],
        model_version=model_info()["model_version"],
    )


@app.get("/segments/{column}", response_model=list[SegmentRow])
def segments(column: str) -> list[SegmentRow]:
    try:
        df = churn_rate_by_segment(column)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return [
        SegmentRow(segment=str(row[column]), n=int(row["n"]), churn_rate=float(row["churn_rate"]))
        for _, row in df.iterrows()
    ]
