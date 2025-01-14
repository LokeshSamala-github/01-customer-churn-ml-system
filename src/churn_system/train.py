"""Train the churn model: DuckDB -> features -> XGBoost, with a proper
train/validation/test split, a small randomized hyperparameter search, and
metrics computed once, honestly, on a held-out test set the search never
touched.

Usage:
    python -m churn_system.train
    python -m churn_system.train --search-iters 15 --rows 20000
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterSampler, train_test_split

from churn_system.config import PATHS, TRAIN_CONFIG
from churn_system.database import load_customer_features
from churn_system.features import build_training_frame

logger = logging.getLogger(__name__)

SEARCH_SPACE = {
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.03, 0.05, 0.06, 0.08, 0.1],
    "subsample": [0.7, 0.8, 0.85, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 2, 3, 5, 8],
    "reg_lambda": [0.5, 1.0, 1.5, 2.0, 3.0],
}


def _split(X: pd.DataFrame, y: pd.Series, ids: pd.Series, cfg=TRAIN_CONFIG):
    X_temp, X_test, y_temp, y_test, ids_temp, ids_test = train_test_split(
        X, y, ids, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )
    val_fraction_of_temp = cfg.val_size / (1 - cfg.test_size)
    X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
        X_temp, y_temp, ids_temp, test_size=val_fraction_of_temp, random_state=cfg.random_state, stratify=y_temp
    )
    logger.info(
        "split sizes -> train=%d val=%d test=%d (churn rate train=%.1f%% val=%.1f%% test=%.1f%%)",
        len(X_train), len(X_val), len(X_test),
        y_train.mean() * 100, y_val.mean() * 100, y_test.mean() * 100,
    )
    return (X_train, y_train, ids_train), (X_val, y_val, ids_val), (X_test, y_test, ids_test)


def _fit_one(params: dict, train, val) -> xgb.XGBClassifier:
    X_train, y_train, _ = train
    X_val, y_val, _ = val
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def _val_auc(model: xgb.XGBClassifier, val) -> float:
    X_val, y_val, _ = val
    proba = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, proba)


def random_search(train, val, n_iter: int, base_params: dict, random_state: int) -> tuple[dict, list[dict]]:
    """Small randomized search over SEARCH_SPACE, early-stopped on the
    validation set. Returns the winning full param dict plus a trial log
    (written into metrics.json so the "why these hyperparameters" question
    has an actual answer instead of a hand-waved default)."""
    sampler = list(ParameterSampler(SEARCH_SPACE, n_iter=n_iter, random_state=random_state))
    trials = []
    best_auc = -1.0
    best_params = None

    for i, sampled in enumerate(sampler):
        params = {**base_params, **sampled}
        params["n_estimators"] = 400  # capped during search; early stopping decides the real count
        params["early_stopping_rounds"] = TRAIN_CONFIG.early_stopping_rounds
        model = _fit_one(params, train, val)
        auc = _val_auc(model, val)
        best_iteration = getattr(model, "best_iteration", None)
        trials.append({"trial": i, **sampled, "val_auc": round(float(auc), 5), "best_iteration": best_iteration})
        logger.info("trial %d/%d val_auc=%.4f params=%s", i + 1, len(sampler), auc, sampled)
        if auc > best_auc:
            best_auc = auc
            best_params = {**params, "n_estimators": (best_iteration or params["n_estimators"]) + 1}

    return best_params, sorted(trials, key=lambda t: -t["val_auc"])


def evaluate(model: xgb.XGBClassifier, split, threshold: float = 0.5) -> dict:
    X, y, _ = split
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "n": int(len(y)),
        "positive_rate": round(float(y.mean()), 4),
        "roc_auc": round(float(roc_auc_score(y, proba)), 4),
        "pr_auc": round(float(average_precision_score(y, proba)), 4),
        "brier_score": round(float(brier_score_loss(y, proba)), 4),
        "precision": round(float(precision_score(y, pred)), 4),
        "recall": round(float(recall_score(y, pred)), 4),
        "f1": round(float(f1_score(y, pred)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "threshold": threshold,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-iters", type=int, default=12)
    parser.add_argument("--skip-search", action="store_true", help="use TRAIN_CONFIG.xgb_params as-is")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    t0 = time.time()

    df = load_customer_features()
    X, y, ids = build_training_frame(df)
    train, val, test = _split(X, y, ids)

    base_params = {k: v for k, v in TRAIN_CONFIG.xgb_params.items() if k not in {"n_estimators"}}

    if args.skip_search:
        params = {**TRAIN_CONFIG.xgb_params, "early_stopping_rounds": TRAIN_CONFIG.early_stopping_rounds}
        trials = []
    else:
        params, trials = random_search(train, val, n_iter=args.search_iters, base_params=base_params, random_state=TRAIN_CONFIG.random_state)

    # Refit the winning config on train+val, evaluate once on test.
    X_trainval = pd.concat([train[0], val[0]])
    y_trainval = pd.concat([train[1], val[1]])
    final_params = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
    final_model = xgb.XGBClassifier(**final_params)
    final_model.fit(X_trainval, y_trainval, verbose=False)

    test_metrics = evaluate(final_model, test)
    val_metrics_of_final = evaluate(final_model, val)
    train_metrics_of_final = evaluate(final_model, train)

    PATHS.model_file.parent.mkdir(parents=True, exist_ok=True)
    final_model.save_model(str(PATHS.model_file))

    report = {
        "trained_at_unix": int(time.time()),
        "training_seconds": round(time.time() - t0, 1),
        "n_rows_total": int(len(df)),
        "search_space": SEARCH_SPACE if not args.skip_search else "skipped (--skip-search)",
        "search_trials": trials,
        "best_params": final_params,
        "metrics": {
            "train": train_metrics_of_final,
            "validation": val_metrics_of_final,
            "test": test_metrics,
        },
        "feature_count": int(X.shape[1]),
    }
    PATHS.metrics_file.parent.mkdir(parents=True, exist_ok=True)
    PATHS.metrics_file.write_text(json.dumps(report, indent=2))

    logger.info("\n=== TEST SET METRICS (held out from search entirely) ===")
    for k, v in test_metrics.items():
        logger.info("  %s: %s", k, v)
    logger.info("\nmodel saved to %s", PATHS.model_file)
    logger.info("metrics saved to %s", PATHS.metrics_file)


if __name__ == "__main__":
    main()
