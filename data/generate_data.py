"""Synthetic telecom-style customer churn dataset generator.

Real customer data can't ship in a public portfolio repo, so this generates
a statistically realistic stand-in: ~7,000 customers with the same shape of
signal you'd actually find in a subscription business (tenure and contract
type dominate; price sensitivity and support-call volume matter; demographics
barely matter) plus deliberate label noise, so a model trained on it has to
do real work and lands at a believable AUC (roughly 0.82-0.88), not a
suspicious 0.99.

Usage:
    python data/generate_data.py                 # writes data/raw/customers.csv
    python data/generate_data.py --rows 20000     # bigger dataset
    python data/generate_data.py --seed 7         # different draw
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CONTRACT_TYPES = ["Month-to-month", "One year", "Two year"]
PAYMENT_METHODS = ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
INTERNET_SERVICE = ["DSL", "Fiber optic", "No"]

# Multiplicative hazard weights: how much each factor pushes churn
# probability up (>1) or down (<1) relative to a baseline monthly hazard.
CONTRACT_HAZARD = {"Month-to-month": 2.6, "One year": 0.55, "Two year": 0.18}
INTERNET_HAZARD = {"DSL": 1.0, "Fiber optic": 1.35, "No": 0.7}
PAYMENT_HAZARD = {
    "Electronic check": 1.5,
    "Mailed check": 0.9,
    "Bank transfer": 0.75,
    "Credit card": 0.7,
}


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def generate(n_customers: int = 7000, seed: int = 42) -> pd.DataFrame:
    rng = _rng(seed)

    customer_id = np.array([f"CUS-{100000 + i}" for i in range(n_customers)])
    tenure_months = rng.gamma(shape=2.1, scale=14, size=n_customers).clip(0, 72).round().astype(int)

    contract = rng.choice(CONTRACT_TYPES, size=n_customers, p=[0.55, 0.24, 0.21])
    internet = rng.choice(INTERNET_SERVICE, size=n_customers, p=[0.34, 0.44, 0.22])
    payment = rng.choice(PAYMENT_METHODS, size=n_customers, p=[0.33, 0.19, 0.23, 0.25])

    senior_citizen = rng.choice([0, 1], size=n_customers, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], size=n_customers, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], size=n_customers, p=[0.3, 0.7])

    base_charge = rng.normal(45, 12, size=n_customers).clip(18, 90)
    internet_premium = np.where(internet == "Fiber optic", rng.normal(35, 8, n_customers).clip(10, 60), 0)
    internet_premium = np.where(internet == "No", 0, internet_premium)
    addon_count = rng.integers(0, 6, size=n_customers)
    addon_charge = addon_count * rng.normal(6, 1.5, size=n_customers).clip(2, 12)
    monthly_charges = (base_charge + internet_premium + addon_charge).round(2)

    tenure_noise = rng.normal(0, 3, size=n_customers)
    total_charges = (monthly_charges * np.maximum(tenure_months + tenure_noise, 0.5)).round(2)

    support_calls = rng.poisson(lam=np.clip(1.1 + (monthly_charges - 45) / 40, 0.2, None), size=n_customers)
    # a genuinely bad recent experience: 1+ call that took multiple contacts to resolve
    unresolved_ticket = rng.binomial(1, np.clip(0.08 + support_calls * 0.03, 0, 0.6))

    paperless_billing = rng.choice(["Yes", "No"], size=n_customers, p=[0.59, 0.41])
    tech_support = rng.choice(["Yes", "No", "No internet service"], size=n_customers, p=[0.29, 0.49, 0.22])
    online_security = rng.choice(["Yes", "No", "No internet service"], size=n_customers, p=[0.28, 0.5, 0.22])

    # --- latent hazard model -> churn probability -------------------------------
    contract_hazard = np.array([CONTRACT_HAZARD[c] for c in contract])
    internet_hazard = np.array([INTERNET_HAZARD[i] for i in internet])
    payment_hazard = np.array([PAYMENT_HAZARD[p] for p in payment])

    tenure_protection = np.exp(-tenure_months / 22.0)  # long tenure strongly protective
    price_pressure = np.clip((monthly_charges - 55) / 40, -0.4, 1.2)
    support_pressure = np.clip(support_calls / 6, 0, 1.0) + unresolved_ticket * 0.35
    tech_support_relief = np.where(tech_support == "Yes", -0.18, 0.0)
    security_relief = np.where(online_security == "Yes", -0.1, 0.0)
    senior_bump = senior_citizen * 0.08
    no_dependents_bump = np.where(dependents == "No", 0.05, 0.0)

    logit = (
        -2.85
        + np.log(contract_hazard) * 1.05
        + np.log(internet_hazard) * 0.55
        + np.log(payment_hazard) * 0.5
        + tenure_protection * 1.9
        + price_pressure * 0.9
        + support_pressure * 1.1
        + tech_support_relief
        + security_relief
        + senior_bump
        + no_dependents_bump
        + rng.normal(0, 0.55, size=n_customers)  # irreducible noise -> caps achievable AUC
    )
    churn_prob = 1 / (1 + np.exp(-logit))
    churned = rng.binomial(1, churn_prob)

    df = pd.DataFrame(
        {
            "customer_id": customer_id,
            "tenure_months": tenure_months,
            "contract": contract,
            "internet_service": internet,
            "payment_method": payment,
            "paperless_billing": paperless_billing,
            "tech_support": tech_support,
            "online_security": online_security,
            "senior_citizen": senior_citizen,
            "partner": partner,
            "dependents": dependents,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "addon_count": addon_count,
            "support_calls_last_90d": support_calls,
            "unresolved_ticket_last_90d": unresolved_ticket,
            "churned": churned,
        }
    )
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=7000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "raw" / "customers.csv",
    )
    args = parser.parse_args()

    df = generate(n_customers=args.rows, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df):,} rows to {args.out}")
    print(f"churn rate: {df['churned'].mean():.1%}")


if __name__ == "__main__":
    main()
