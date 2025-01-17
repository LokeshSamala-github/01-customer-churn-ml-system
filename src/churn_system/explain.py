"""SHAP explainability for the trained churn model.

Produces two artifacts under reports/:
  - shap_summary.png       a beeswarm plot of the top drivers, over a sample
                            of the test set (matplotlib, non-interactive)
  - shap_feature_importance.csv   mean |SHAP value| per feature, so "what
                            drives churn" can be answered without opening
                            a notebook

Run after `python -m churn_system.train` (needs a saved model + a fitted
feature manifest).
"""

from __future__ import annotations

import argparse
import logging

import matplotlib

matplotlib.use("Agg")  # headless: no display in CI/containers
import matplotlib.pyplot as plt
import pandas as pd
import shap
import xgboost as xgb

from churn_system.config import PATHS
from churn_system.database import load_customer_features
from churn_system.features import build_inference_frame

logger = logging.getLogger(__name__)


def load_model(model_file=PATHS.model_file) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier()
    model.load_model(str(model_file))
    return model


def compute_shap_values(model: xgb.XGBClassifier, X: pd.DataFrame) -> shap.Explanation:
    explainer = shap.TreeExplainer(model)
    return explainer(X)


def summarize_importance(shap_values: shap.Explanation, X: pd.DataFrame) -> pd.DataFrame:
    importance = pd.DataFrame(
        {
            "feature": X.columns,
            "mean_abs_shap": abs(shap_values.values).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    importance["rank"] = importance.index + 1
    return importance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=800, help="rows to explain (SHAP cost scales with this)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    df = load_customer_features()
    sample = df.sample(n=min(args.sample_size, len(df)), random_state=7)
    X = build_inference_frame(sample)

    model = load_model()
    shap_values = compute_shap_values(model, X)

    importance = summarize_importance(shap_values, X)
    PATHS.shap_importance_csv.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(PATHS.shap_importance_csv, index=False)
    logger.info("top 10 drivers of churn (mean |SHAP|):")
    for _, row in importance.head(10).iterrows():
        logger.info("  %2d. %-45s %.4f", row["rank"], row["feature"], row["mean_abs_shap"])

    plt.figure()
    shap.summary_plot(shap_values, X, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(PATHS.shap_summary_png, dpi=140)
    plt.close()

    logger.info("\nwrote %s", PATHS.shap_importance_csv)
    logger.info("wrote %s", PATHS.shap_summary_png)


if __name__ == "__main__":
    main()
