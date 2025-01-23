import pandas as pd

from churn_system.config import PATHS
from churn_system.database import (
    build_warehouse,
    churn_rate_by_segment,
    compute_features_inmemory,
    load_customer_features,
)


def test_build_warehouse_row_count(_raw_data):
    con = build_warehouse()
    try:
        n = con.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
        assert n == 1200
    finally:
        con.close()


def test_load_customer_features_has_derived_columns(_raw_data):
    df = load_customer_features()
    for col in ["tenure_bucket", "avg_historical_monthly_spend", "spend_trend", "is_high_support_volume"]:
        assert col in df.columns


def test_churn_rate_by_segment_rejects_unknown_column(_raw_data):
    import pytest

    with pytest.raises(ValueError):
        churn_rate_by_segment("not_a_real_column")


def test_churn_rate_by_segment_sums_to_total_rows(_raw_data):
    warehouse_df = load_customer_features()
    seg = churn_rate_by_segment("contract")
    assert seg["n"].sum() == len(warehouse_df)


def test_inmemory_features_match_warehouse_features(_raw_data):
    """The whole point of compute_features_inmemory is that it must produce
    IDENTICAL derived columns to the persisted warehouse view for the same
    rows -- otherwise the model would see different features at serving
    time than it was trained on. This test is the guardrail against that
    drifting apart if either code path is edited later."""
    raw = pd.read_csv(PATHS.raw_csv)
    warehouse_df = load_customer_features().sort_values("customer_id").reset_index(drop=True)

    inmemory_df = compute_features_inmemory(raw).sort_values("customer_id").reset_index(drop=True)

    shared_cols = [c for c in warehouse_df.columns if c in inmemory_df.columns]
    pd.testing.assert_frame_equal(
        warehouse_df[shared_cols].reset_index(drop=True),
        inmemory_df[shared_cols].reset_index(drop=True),
        check_dtype=False,
    )
