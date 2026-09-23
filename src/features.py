"""
features.py
-----------
All feature engineering transformations for the churn model.

Design principle: every function takes a DataFrame and returns a DataFrame.
This makes them composable and easy to test in isolation.
"""

import pandas as pd
import numpy as np


# ── Tenure ────────────────────────────────────────────────────────────────────

def add_tenure_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bucket days_tenure into 4 customer lifecycle stages.
    Research shows churn risk is highest in the first year and stabilizes
    after ~3 years for insurance customers.
    """
    bins = [0, 365, 1095, 2190, float("inf")]
    labels = ["New (< 1yr)", "Growing (1-3yr)", "Loyal (3-6yr)", "Long-term (6yr+)"]
    df["tenure_bucket"] = pd.cut(df["days_tenure"], bins=bins, labels=labels)
    df["tenure_years"] = (df["days_tenure"] / 365).round(1)
    return df


# ── Age ───────────────────────────────────────────────────────────────────────

def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """Segment customers into generational age groups."""
    bins = [0, 35, 50, 65, float("inf")]
    labels = ["Under 35", "35-50", "50-65", "65+"]
    df["age_group"] = pd.cut(df["age_in_years"], bins=bins, labels=labels)
    return df


# ── Income ────────────────────────────────────────────────────────────────────

def add_income_bracket(df: pd.DataFrame) -> pd.DataFrame:
    """Bin income into 5 brackets for segment analysis."""
    bins = [0, 30000, 60000, 100000, 150000, float("inf")]
    labels = ["< $30K", "$30-60K", "$60-100K", "$100-150K", "$150K+"]
    df["income_bracket"] = pd.cut(df["income"], bins=bins, labels=labels)
    return df


# ── Home Market Value ─────────────────────────────────────────────────────────

HOME_VALUE_MAP = {
    "Less than 50000": 25000,
    "50000 - 74999": 62500,
    "75000 - 99999": 87500,
    "100000 - 124999": 112500,
    "125000 - 149999": 137500,
    "150000 - 174999": 162500,
    "175000 - 199999": 187500,
    "200000 - 224999": 212500,
    "225000 - 249999": 237500,
    "250000 - 274999": 262500,
    "275000 - 299999": 287500,
    "300000 +": 325000,
}

def encode_home_market_value(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the range string to a numeric midpoint for modeling."""
    df["home_market_value_num"] = df["home_market_value"].map(HOME_VALUE_MAP)
    return df


# ── Marital Status ────────────────────────────────────────────────────────────

def encode_marital_status(df: pd.DataFrame) -> pd.DataFrame:
    """Binary encode marital status (1 = Married, 0 = Single)."""
    df["is_married"] = (df["marital_status"] == "Married").astype(int)
    return df


# ── Premium-to-Income Ratio ───────────────────────────────────────────────────

def add_premium_income_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Annual premium as % of income — a proxy for price sensitivity.
    Higher ratio = customer may feel more financial pressure from the policy.
    """
    df["premium_income_ratio"] = (df["curr_ann_amt"] / df["income"].replace(0, np.nan)).round(4)
    return df


# ── Pipeline ──────────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps in sequence.
    Call this on the raw loaded DataFrame to get a model-ready DataFrame.
    """
    df = df.copy()
    df = add_tenure_bucket(df)
    df = add_age_group(df)
    df = add_income_bracket(df)
    df = encode_home_market_value(df)
    df = encode_marital_status(df)
    df = add_premium_income_ratio(df)
    return df


def get_model_features() -> list:
    """
    Canonical list of numeric features for model training.
    Keeping this in one place avoids train/serve skew.
    """
    return [
        "curr_ann_amt",
        "days_tenure",
        "age_in_years",
        "income",
        "has_children",
        "length_of_residence",
        "home_owner",
        "college_degree",
        "good_credit",
        "is_married",
        "home_market_value_num",
        "premium_income_ratio",
        "tenure_years",
    ]
