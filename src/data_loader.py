"""
data_loader.py
--------------
Loads and merges the 5 raw CSV tables into a single analysis-ready DataFrame.

Why this exists:
  - The main autoinsurance_churn.csv already has most fields denormalized.
  - This module also shows how to join from the normalized tables (customer,
    demographic, address, termination) in case you want to rebuild from source.
"""

import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def load_main(nrows=None) -> pd.DataFrame:
    """
    Load the pre-joined autoinsurance_churn.csv.
    Use nrows during development to avoid loading 1.68M rows into memory.
    """
    path = RAW_DIR / "autoinsurance_churn.csv"
    df = pd.read_csv(path, nrows=nrows, low_memory=False)
    df.columns = df.columns.str.lower().str.strip()
    return df


def load_from_normalized(nrows=None) -> pd.DataFrame:
    """
    Alternative: build the dataset by joining the 4 normalized tables.
    Useful to understand the data model and for extending with new fields.

    Schema:
      customer (base)
        LEFT JOIN demographic ON individual_id
        LEFT JOIN address     ON address_id
        LEFT JOIN termination ON individual_id  → derives Churn flag
    """
    customer = pd.read_csv(RAW_DIR / "customer.csv", nrows=nrows, low_memory=False)
    demographic = pd.read_csv(RAW_DIR / "demographic.csv", nrows=nrows, low_memory=False)
    address = pd.read_csv(RAW_DIR / "address.csv", nrows=nrows, low_memory=False)
    termination = pd.read_csv(RAW_DIR / "termination.csv", nrows=nrows, low_memory=False)

    # Standardize column names
    for df in [customer, demographic, address, termination]:
        df.columns = df.columns.str.lower().str.strip()

    # Build churn flag from termination table
    termination["churn"] = 1

    # Join
    merged = (
        customer
        .merge(demographic, on="individual_id", how="left")
        .merge(address, on="address_id", how="left")
        .merge(termination[["individual_id", "acct_suspd_date", "churn"]],
               on="individual_id", how="left")
    )
    merged["churn"] = merged["churn"].fillna(0).astype(int)
    return merged


def save_processed(df: pd.DataFrame, filename: str = "churn_processed.csv"):
    """Save cleaned/engineered DataFrame to data/processed/."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / filename
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df):,} rows → {out_path}")
