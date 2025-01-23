from churn_system.database import load_customer_features
from churn_system.features import build_inference_frame, build_training_frame


def test_build_training_frame_shapes(_raw_data):
    df = load_customer_features()
    X, y, ids = build_training_frame(df)
    assert len(X) == len(y) == len(ids) == len(df)
    assert y.isin([0, 1]).all()
    assert X.select_dtypes(exclude=["number", "bool"]).empty, "all features must be numeric for XGBoost"


def test_build_inference_frame_matches_training_columns(_raw_data):
    df = load_customer_features()
    X_train, _, _ = build_training_frame(df)

    # A fresh, smaller batch must reindex onto the exact same manifest
    # columns even though pd.get_dummies alone would produce a different
    # (and differently ordered) set of columns for a smaller sample.
    small_batch = df.sample(n=5, random_state=1)
    X_infer = build_inference_frame(small_batch)

    assert list(X_infer.columns) == list(X_train.columns)


def test_inference_frame_handles_unseen_category_gracefully(_raw_data):
    df = load_customer_features()
    build_training_frame(df)  # ensures the manifest exists

    row = df.iloc[[0]].copy()
    row["contract"] = "Lifetime"  # a category the manifest has never seen
    X = build_inference_frame(row)
    # Every contract_* dummy should be 0 for this row (unknown category),
    # not raise, and not silently create a new "contract_Lifetime" column.
    contract_cols = [c for c in X.columns if c.startswith("contract_")]
    assert X.loc[row.index[0], contract_cols].sum() == 0
