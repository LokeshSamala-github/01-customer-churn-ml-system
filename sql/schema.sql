-- DuckDB schema + analytical views for the churn system.
-- Loaded by src/churn_system/database.py against data/raw/customers.csv.
-- Written as plain, reviewable SQL (not ORM-generated) since this is the
-- artifact a data analyst on the team would actually read.

CREATE OR REPLACE TABLE customers AS
SELECT * FROM read_csv_auto(?, header = true);

-- One row per customer with a few analyst-friendly derived columns.
-- This is the table the feature-engineering step (features.py) reads from,
-- so business logic that's naturally expressed in SQL (bucketing, ratios)
-- lives here instead of being re-implemented in pandas.
CREATE OR REPLACE VIEW customer_features AS
SELECT
    customer_id,
    tenure_months,
    CASE
        WHEN tenure_months < 6 THEN 'new_0_6m'
        WHEN tenure_months < 12 THEN 'ramping_6_12m'
        WHEN tenure_months < 24 THEN 'established_1_2y'
        ELSE 'loyal_2y_plus'
    END AS tenure_bucket,
    contract,
    internet_service,
    payment_method,
    -- Normalize explicitly to the literal strings 'Yes'/'No': DuckDB's
    -- read_csv_auto infers these Yes/No text columns as native BOOLEAN when
    -- reading the warehouse CSV (-> CAST(...AS VARCHAR) would yield
    -- 'true'/'false'), but the same column arrives as the original VARCHAR
    -- 'Yes'/'No' when this view runs in-memory against a pandas DataFrame
    -- at serving time (compute_features_inmemory). Without normalizing both
    -- to the same strings, the two paths would one-hot encode these fields
    -- under different column names (paperless_billing_true vs
    -- paperless_billing_Yes) and inference would silently zero the feature
    -- out. See tests/test_database.py::test_inmemory_features_match_warehouse_features.
    CASE WHEN CAST(paperless_billing AS VARCHAR) IN ('true', 'Yes') THEN 'Yes' ELSE 'No' END AS paperless_billing,
    tech_support,
    online_security,
    senior_citizen,
    CASE WHEN CAST(partner AS VARCHAR) IN ('true', 'Yes') THEN 'Yes' ELSE 'No' END AS partner,
    CASE WHEN CAST(dependents AS VARCHAR) IN ('true', 'Yes') THEN 'Yes' ELSE 'No' END AS dependents,
    monthly_charges,
    total_charges,
    ROUND(total_charges / NULLIF(tenure_months, 0), 2) AS avg_historical_monthly_spend,
    ROUND(monthly_charges - (total_charges / NULLIF(tenure_months, 0)), 2) AS spend_trend,
    addon_count,
    support_calls_last_90d,
    unresolved_ticket_last_90d,
    CASE WHEN support_calls_last_90d >= 3 THEN 1 ELSE 0 END AS is_high_support_volume,
    churned
FROM customers;

-- Ad-hoc analyst queries kept here as documentation of "why these features":
-- run any of these with `duckdb data/warehouse.duckdb -c "..."` after
-- python -m churn_system.database has built the warehouse file.

-- Churn rate by contract type (the single strongest driver in this dataset).
-- SELECT contract, COUNT(*) AS n, ROUND(AVG(churned), 3) AS churn_rate
-- FROM customer_features GROUP BY contract ORDER BY churn_rate DESC;

-- Churn rate by tenure bucket x support-call volume (interaction effect).
-- SELECT tenure_bucket, is_high_support_volume, COUNT(*) AS n,
--        ROUND(AVG(churned), 3) AS churn_rate
-- FROM customer_features
-- GROUP BY tenure_bucket, is_high_support_volume
-- ORDER BY tenure_bucket, is_high_support_volume;

-- Revenue at risk: monthly revenue currently sitting on high-risk-looking
-- accounts (month-to-month + a recent unresolved ticket), independent of the
-- model — a sanity baseline to compare the model's flagged cohort against.
-- SELECT ROUND(SUM(monthly_charges), 2) AS at_risk_mrr, COUNT(*) AS n
-- FROM customer_features
-- WHERE contract = 'Month-to-month' AND unresolved_ticket_last_90d = 1;
