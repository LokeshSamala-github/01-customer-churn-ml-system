"""Feature engineering: turns `customer_features` (SQL output) into a numeric
matrix XGBoost can train on.

Kept as a small set of plain functions rather than a scikit-learn Pipeline
class hierarchy on purpose -- there's no preprocessing here complex enough to
need fit/transform state (one-hot categories are a fixed, known set), so a
class would just be indirection. The one piece of "fit state" that does
exist -- the final column order -- is saved to a manifest file so serving
can guarantee it matches training exactly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from churn_system.config import PATHS

TARGET_COL = "churned"
ID_COL = "customer_id"

CATEGORICAL_COLS = [
    "tenure_bucket",
    "contract",
    "internet_service",
    "payment_method",
    "paperless_billing",
    "tech_support",
    "online_security",
    "partner",
    "dependents",
]
NUMERIC_COLS = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "avg_historical_monthly_spend",
    "spend_trend",
    "addon_count",
    "support_calls_last_90d",
    "unresolved_ticket_last_90d",
    "is_high_support_volume",
    "senior_citizen",
]


def build_feature_matrix(df: pd.DataFrame, fit: bool, manifest_path: Path = PATHS.feature_manifest) -> pd.DataFrame:
    """One-hot encode categoricals + pass through numerics.

    fit=True (training time): derives the full set of one-hot columns from
        the data and writes it to `manifest_path`.
    fit=False (serving time): reads the same manifest and reindexes onto it,
        so a category unseen at inference time (or missing because a single
        row can't produce every dummy) can't silently shift column order --
        it just gets all-zero dummies for that field, which is the correct
        "unknown category" behaviour for a linear/tree model.
    """
    numeric = df[NUMERIC_COLS].copy()
    # DuckDB's CSV type inference turns Yes/No text columns into real bool
    # dtype, which pandas' get_dummies silently *skips* by default (it only
    # auto-selects object/string/category columns) -- passing `columns=`
    # explicitly forces every listed field to be dummied regardless of the
    # dtype DuckDB happened to infer for it.
    dummies = pd.get_dummies(df[CATEGORICAL_COLS].astype(str), columns=CATEGORICAL_COLS, prefix=CATEGORICAL_COLS)
    X = pd.concat([numeric, dummies], axis=1)

    if fit:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps({"columns": list(X.columns)}, indent=2))
        return X

    manifest = json.loads(manifest_path.read_text())
    return X.reindex(columns=manifest["columns"], fill_value=0)


def build_training_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Returns (X, y, customer_id) for the training pipeline."""
    X = build_feature_matrix(df, fit=True)
    y = df[TARGET_COL].astype(int)
    ids = df[ID_COL]
    return X, y, ids


def build_inference_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Same transform as training, but reindexed onto the saved manifest for
    a single request or a small batch at serving time."""
    return build_feature_matrix(df, fit=False)
