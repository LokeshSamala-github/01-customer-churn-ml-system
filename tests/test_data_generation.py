from data.generate_data import generate


def test_generate_is_deterministic_given_seed():
    df1 = generate(n_customers=500, seed=1)
    df2 = generate(n_customers=500, seed=1)
    assert df1.equals(df2)


def test_generate_different_seeds_differ():
    df1 = generate(n_customers=500, seed=1)
    df2 = generate(n_customers=500, seed=2)
    assert not df1["churned"].equals(df2["churned"])


def test_churn_rate_is_realistic():
    df = generate(n_customers=5000, seed=42)
    rate = df["churned"].mean()
    # A real subscription business's monthly/annual churn rate is nowhere
    # near 0% (no signal reaching the label) or 100% (broken generator) --
    # pin it to a believable band instead of an exact number so tuning the
    # hazard model doesn't require touching the test every time.
    assert 0.15 < rate < 0.40


def test_required_columns_present():
    df = generate(n_customers=100, seed=1)
    expected = {
        "customer_id", "tenure_months", "contract", "internet_service",
        "payment_method", "paperless_billing", "tech_support",
        "online_security", "senior_citizen", "partner", "dependents",
        "monthly_charges", "total_charges", "addon_count",
        "support_calls_last_90d", "unresolved_ticket_last_90d", "churned",
    }
    assert expected.issubset(df.columns)


def test_no_nulls():
    df = generate(n_customers=1000, seed=1)
    assert df.isna().sum().sum() == 0


def test_signal_is_directionally_sane():
    """Month-to-month contracts should churn more than two-year contracts --
    if this ever flips, the hazard model's sign got flipped by accident."""
    df = generate(n_customers=8000, seed=7)
    mtm_rate = df.loc[df["contract"] == "Month-to-month", "churned"].mean()
    two_year_rate = df.loc[df["contract"] == "Two year", "churned"].mean()
    assert mtm_rate > two_year_rate * 2
