"""Shared pytest fixtures.

The whole test session runs against a throwaway project directory instead of
the real data/, models/, reports/ folders. This works by setting the
CHURN_* environment variables that `churn_system.config.Paths` already
supports (see config.py's `_env_path` helper) *before* anything in
churn_system gets imported -- conftest.py is guaranteed to run before any
test module's imports, so every `Paths()` instance constructed anywhere in
the app picks up the temp location automatically. No per-module monkeypatch
needed, and it's the same override mechanism a real deployment would use to
point at a different data directory.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (str(SRC), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

_TMP = Path(tempfile.mkdtemp(prefix="churn_system_tests_"))
os.environ.setdefault("CHURN_RAW_CSV", str(_TMP / "data" / "raw" / "customers.csv"))
os.environ.setdefault("CHURN_DUCKDB_FILE", str(_TMP / "data" / "warehouse.duckdb"))
os.environ.setdefault("CHURN_MODEL_FILE", str(_TMP / "models" / "churn_model.json"))
os.environ.setdefault("CHURN_FEATURE_MANIFEST", str(_TMP / "models" / "feature_manifest.json"))
os.environ.setdefault("CHURN_METRICS_FILE", str(_TMP / "reports" / "metrics.json"))
os.environ.setdefault("CHURN_SHAP_PNG", str(_TMP / "reports" / "shap_summary.png"))
os.environ.setdefault("CHURN_SHAP_CSV", str(_TMP / "reports" / "shap_feature_importance.csv"))

import pytest  # noqa: E402  (import order is deliberate -- env vars must be set first)

from churn_system.config import PATHS  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _raw_data() -> Path:
    """Generate a small synthetic dataset once per test session."""
    from data.generate_data import generate

    df = generate(n_customers=1200, seed=123)
    PATHS.raw_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PATHS.raw_csv, index=False)
    return PATHS.raw_csv


@pytest.fixture(scope="session")
def trained_model(_raw_data):
    """Actually run the real training pipeline once (small data, small
    search) so downstream tests exercise genuine trained-model behaviour
    rather than a mock."""
    import sys as _sys

    from churn_system import predict as predict_mod
    from churn_system import train as train_mod

    argv_backup = _sys.argv
    try:
        _sys.argv = ["train.py", "--search-iters", "3"]
        train_mod.main()
    finally:
        _sys.argv = argv_backup

    predict_mod.get_model.cache_clear()
    predict_mod.get_explainer.cache_clear()
    return PATHS
