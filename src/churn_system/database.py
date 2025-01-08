"""DuckDB warehouse layer.

Loads the raw customer CSV into a local DuckDB file and exposes the
`customer_features` SQL view defined in sql/schema.sql. Using a real SQL
engine (rather than doing everything in pandas) is deliberate: the
tenure-bucketing and spend-trend logic below is the kind of thing an
analyst maintains in SQL, and keeping it there means `churn_system.features`
and a BI tool can share the exact same definition.
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import pandas as pd

from churn_system.config import PATHS

logger = logging.getLogger(__name__)

SCHEMA_SQL_PATH = Path(__file__).resolve().parents[2] / "sql" / "schema.sql"


def _customer_features_select_sql() -> str:
    """Extract just the `SELECT ...` body of the `customer_features` view
    from sql/schema.sql, so training (which builds the view against the
    persisted warehouse) and online serving (which needs the exact same
    derived columns -- tenure_bucket, avg_historical_monthly_spend,
    spend_trend, is_high_support_volume -- computed for a single request
    that never touches the warehouse file) can share one definition instead
    of two copies that quietly drift apart. See `compute_features_inmemory`
    and tests/test_database.py::test_inmemory_features_match_warehouse.
    """
    schema_sql = SCHEMA_SQL_PATH.read_text()
    uncommented = "\n".join(
        line for line in schema_sql.splitlines() if not line.strip().startswith("--")
    )
    statements = [s.strip() for s in uncommented.split(";") if s.strip()]
    view_stmt = next(s for s in statements if s.upper().startswith("CREATE OR REPLACE VIEW CUSTOMER_FEATURES"))
    # Isolate the SELECT body from the "CREATE VIEW ... AS" wrapper by
    # locating the first SELECT keyword -- more robust than splitting on
    # " AS " (the wrapper's "AS" is followed by a newline, not a space, so
    # that naive split misses it entirely).
    select_idx = view_stmt.upper().index("SELECT")
    return view_stmt[select_idx:].strip()


def compute_features_inmemory(df: pd.DataFrame) -> pd.DataFrame:
    """Run the *same* customer_features SQL against an arbitrary in-memory
    DataFrame (e.g. one request's worth of customers hitting the API) --
    used by predict.py so serving-time feature derivation can never drift
    from what the model was trained on."""
    select_sql = _customer_features_select_sql()
    con = duckdb.connect(":memory:")
    try:
        con.register("customers", df)
        return con.execute(select_sql).fetchdf()
    finally:
        con.close()


def build_warehouse(
    raw_csv: Path = PATHS.raw_csv,
    duckdb_file: Path = PATHS.duckdb_file,
) -> duckdb.DuckDBPyConnection:
    """(Re)build the DuckDB warehouse from the raw CSV.

    Returns an open connection so callers can chain queries without a second
    round trip through disk.
    """
    if not raw_csv.exists():
        raise FileNotFoundError(
            f"{raw_csv} not found. Run `python data/generate_data.py` first."
        )

    duckdb_file.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(duckdb_file))

    schema_sql = SCHEMA_SQL_PATH.read_text()
    # Strip full-line SQL comments first, *then* split into statements --
    # otherwise a statement that merely starts with a `--` doc-comment line
    # (nearly all of them, here) would be dropped whole. The commented-out
    # analyst queries at the bottom of the file collapse to blank statements
    # and are skipped by the `if s.strip()` filter below.
    uncommented = "\n".join(
        line for line in schema_sql.splitlines() if not line.strip().startswith("--")
    )
    statements = [s.strip() for s in uncommented.split(";") if s.strip()]
    con.execute(statements[0], [str(raw_csv)])
    for statement in statements[1:]:
        con.execute(statement)

    n_rows = con.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
    logger.info("built warehouse at %s (%s rows)", duckdb_file, n_rows)
    return con


def load_customer_features(
    raw_csv: Path = PATHS.raw_csv,
    duckdb_file: Path = PATHS.duckdb_file,
) -> pd.DataFrame:
    """Convenience: build the warehouse (if needed) and return
    `customer_features` as a pandas DataFrame for the feature-engineering
    step."""
    con = build_warehouse(raw_csv, duckdb_file)
    try:
        return con.execute("SELECT * FROM customer_features").fetchdf()
    finally:
        con.close()


def churn_rate_by_segment(column: str, duckdb_file: Path = PATHS.duckdb_file) -> pd.DataFrame:
    """Run the "churn rate by X" analyst query for an arbitrary categorical
    column. Used by tests and by the /segments endpoint in the API."""
    allowed = {
        "contract",
        "internet_service",
        "payment_method",
        "tenure_bucket",
        "tech_support",
        "online_security",
    }
    if column not in allowed:
        raise ValueError(f"column must be one of {sorted(allowed)}, got {column!r}")

    con = duckdb.connect(str(duckdb_file))
    try:
        return con.execute(
            f"""
            SELECT {column}, COUNT(*) AS n, ROUND(AVG(churned), 4) AS churn_rate
            FROM customer_features
            GROUP BY {column}
            ORDER BY churn_rate DESC
            """
        ).fetchdf()
    finally:
        con.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = load_customer_features()
    print(df.head())
    print(f"\n{len(df):,} rows, churn rate {df['churned'].mean():.1%}")
