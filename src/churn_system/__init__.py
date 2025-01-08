"""Production-grade customer churn ML system.

A self-contained pipeline that takes a subscription business's customer table
from raw records to a served, explainable churn-risk model:

    data generation/ingestion -> DuckDB feature views (SQL) -> feature
    engineering -> XGBoost training -> SHAP explainability -> FastAPI serving

Every stage is a plain, importable module so it can be run as a script,
called from a notebook, or wired into an orchestrator (Airflow, Dagster,
cron) without modification.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("churn-system")
except PackageNotFoundError:  # running from source without an editable install
    __version__ = "0.0.0+dev"

__all__ = ["__version__"]
