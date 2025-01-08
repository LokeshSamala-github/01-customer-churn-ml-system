"""Central configuration for the churn system.

Kept deliberately simple (a dataclass, not a settings framework) because the
whole point is that every path and hyperparameter used by every stage is
declared in exactly one place. Override any of these with environment
variables of the same name (upper-cased) if you wire this into a container.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else default


@dataclass(frozen=True)
class Paths:
    root: Path = PROJECT_ROOT
    raw_csv: Path = field(default_factory=lambda: _env_path("CHURN_RAW_CSV", PROJECT_ROOT / "data" / "raw" / "customers.csv"))
    duckdb_file: Path = field(default_factory=lambda: _env_path("CHURN_DUCKDB_FILE", PROJECT_ROOT / "data" / "warehouse.duckdb"))
    model_file: Path = field(default_factory=lambda: _env_path("CHURN_MODEL_FILE", PROJECT_ROOT / "models" / "churn_model.json"))
    feature_manifest: Path = field(default_factory=lambda: _env_path("CHURN_FEATURE_MANIFEST", PROJECT_ROOT / "models" / "feature_manifest.json"))
    metrics_file: Path = field(default_factory=lambda: _env_path("CHURN_METRICS_FILE", PROJECT_ROOT / "reports" / "metrics.json"))
    shap_summary_png: Path = field(default_factory=lambda: _env_path("CHURN_SHAP_PNG", PROJECT_ROOT / "reports" / "shap_summary.png"))
    shap_importance_csv: Path = field(default_factory=lambda: _env_path("CHURN_SHAP_CSV", PROJECT_ROOT / "reports" / "shap_feature_importance.csv"))


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters and split ratios. Values were picked by a small grid
    search recorded in reports/metrics.json (see `search_space` key)."""

    test_size: float = 0.15
    val_size: float = 0.15  # of the remaining train+val portion
    random_state: int = 42
    early_stopping_rounds: int = 30
    xgb_params: dict = field(
        default_factory=lambda: {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "max_depth": 4,
            "learning_rate": 0.06,
            "n_estimators": 600,
            "subsample": 0.85,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "reg_lambda": 1.5,
            "reg_alpha": 0.1,
            "random_state": 42,
            "n_jobs": -1,
        }
    )


PATHS = Paths()
TRAIN_CONFIG = TrainConfig()
