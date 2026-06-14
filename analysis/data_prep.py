"""
data_prep.py — Data loading, cleaning, and feature engineering functions.

All raw data is defined as Python dicts (simulated from BLS OES, Numbeo,
USCIS H-1B Employer Data Hub, and LinkedIn/Indeed/Glassdoor, 2024-2025).
"""

import pandas as pd
import numpy as np
from typing import Optional

# ── Constants ────────────────────────────────────────────────────────────────
BASELINE_COST_INDEX: int = 70
DEFAULT_HOURLY_RATE: float = 40.0

# ── Raw data ─────────────────────────────────────────────────────────────────
_CITY_RAW = [
    {"city": "San Francisco", "state": "CA", "raw_salary": 95000, "cost_index": 100,
     "visa_friendly_score": 90, "h1b_approvals_pct": 88, "job_postings_count": 4200},
    {"city": "New York",      "state": "NY", "raw_salary": 88000, "cost_index": 92,
     "visa_friendly_score": 85, "h1b_approvals_pct": 82, "job_postings_count": 5100},
    {"city": "Seattle",       "state": "WA", "raw_salary": 86000, "cost_index": 88,
     "visa_friendly_score": 88, "h1b_approvals_pct": 85, "job_postings_count": 3100},
    {"city": "Chicago",       "state": "IL", "raw_salary": 72000, "cost_index": 70,
     "visa_friendly_score": 78, "h1b_approvals_pct": 74, "job_postings_count": 2800},
    {"city": "Austin",        "state": "TX", "raw_salary": 74000, "cost_index": 72,
     "visa_friendly_score": 80, "h1b_approvals_pct": 76, "job_postings_count": 2200},
    {"city": "Boston",        "state": "MA", "raw_salary": 80000, "cost_index": 85,
     "visa_friendly_score": 86, "h1b_approvals_pct": 84, "job_postings_count": 2600},
    {"city": "Washington DC", "state": "DC", "raw_salary": 83000, "cost_index": 82,
     "visa_friendly_score": 92, "h1b_approvals_pct": 91, "job_postings_count": 3400},
    {"city": "Atlanta",       "state": "GA", "raw_salary": 68000, "cost_index": 62,
     "visa_friendly_score": 74, "h1b_approvals_pct": 70, "job_postings_count": 1900},
    {"city": "Dallas",        "state": "TX", "raw_salary": 70000, "cost_index": 65,
     "visa_friendly_score": 77, "h1b_approvals_pct": 73, "job_postings_count": 2100},
    {"city": "Denver",        "state": "CO", "raw_salary": 75000, "cost_index": 76,
     "visa_friendly_score": 79, "h1b_approvals_pct": 76, "job_postings_count": 1800},
]

_CERT_RAW = [
    {"cert": "SQL Advanced (HackerRank)",        "weeks_to_earn": 2,  "cost_usd": 0,   "est_salary_lift": 4200},
    {"cert": "IBM Data Analysis with Python",    "weeks_to_earn": 6,  "cost_usd": 0,   "est_salary_lift": 5800},
    {"cert": "Microsoft Fabric (Applied Skills)","weeks_to_earn": 8,  "cost_usd": 165, "est_salary_lift": 7500},
    {"cert": "Tableau Desktop Specialist",       "weeks_to_earn": 4,  "cost_usd": 250, "est_salary_lift": 6100},
    {"cert": "Google Data Analytics Certificate","weeks_to_earn": 12, "cost_usd": 200, "est_salary_lift": 4500},
    {"cert": "AWS Cloud Practitioner",           "weeks_to_earn": 10, "cost_usd": 300, "est_salary_lift": 8200},
    {"cert": "PMP Certification",                "weeks_to_earn": 24, "cost_usd": 555, "est_salary_lift": 9800},
    {"cert": "CPA Exam (1 section)",             "weeks_to_earn": 20, "cost_usd": 800, "est_salary_lift": 11000},
]

_SKILL_RAW = [
    {"skill": "SQL",                "demand_pct": 87, "category": "Technical"},
    {"skill": "Python",             "demand_pct": 72, "category": "Technical"},
    {"skill": "Excel",              "demand_pct": 68, "category": "Technical"},
    {"skill": "Tableau / Power BI", "demand_pct": 64, "category": "Technical"},
    {"skill": "Communication",      "demand_pct": 61, "category": "Soft"},
    {"skill": "ETL / Pipelines",    "demand_pct": 48, "category": "Technical"},
    {"skill": "Statistics / ML",    "demand_pct": 44, "category": "Technical"},
    {"skill": "Cloud (AWS/Azure)",  "demand_pct": 39, "category": "Technical"},
]


# ── Loaders ──────────────────────────────────────────────────────────────────

def load_city_data() -> pd.DataFrame:
    """Load raw city salary dataset.

    Returns:
        pd.DataFrame: 10-row DataFrame with city salary and market attributes.
    """
    df = pd.DataFrame(_CITY_RAW)
    _validate_dataframe(df, ["city", "state", "raw_salary", "cost_index",
                              "visa_friendly_score", "h1b_approvals_pct",
                              "job_postings_count"])
    return df


def load_cert_data() -> pd.DataFrame:
    """Load raw certification ROI dataset.

    Returns:
        pd.DataFrame: 8-row DataFrame with certification cost and salary lift.
    """
    df = pd.DataFrame(_CERT_RAW)
    _validate_dataframe(df, ["cert", "weeks_to_earn", "cost_usd", "est_salary_lift"])
    return df


def load_skill_data() -> pd.DataFrame:
    """Load raw skill demand dataset.

    Returns:
        pd.DataFrame: 8-row DataFrame with skill demand percentages.
    """
    df = pd.DataFrame(_SKILL_RAW)
    _validate_dataframe(df, ["skill", "demand_pct", "category"])
    return df


# ── Feature engineering ───────────────────────────────────────────────────────

def compute_city_features(
    df: pd.DataFrame,
    baseline: int = BASELINE_COST_INDEX,
) -> pd.DataFrame:
    """Add derived columns to city DataFrame.

    Columns added:
        adjusted_salary  — purchasing-power-normalised salary
        salary_premium   — nominal minus adjusted (premium for accepting higher CoL)
        value_score      — composite 50/50 score: adjusted salary + visa friendliness
        high_value       — binary label: 1 if value_score >= median, else 0

    Args:
        df: Raw city DataFrame from load_city_data().
        baseline: Cost index value used as normalisation denominator (default 70).

    Returns:
        pd.DataFrame: Input DataFrame with four new columns appended.
    """
    df = df.copy()
    df["adjusted_salary"] = df["raw_salary"] / df["cost_index"] * baseline
    df["salary_premium"]  = df["raw_salary"] - df["adjusted_salary"]
    df["value_score"]     = (
        df["adjusted_salary"] / df["adjusted_salary"].max() * 0.5
        + df["visa_friendly_score"] / 100 * 0.5
    )
    df["high_value"] = (df["value_score"] >= df["value_score"].median()).astype(int)
    return df


def compute_cert_features(
    df: pd.DataFrame,
    hourly_rate: float = DEFAULT_HOURLY_RATE,
) -> pd.DataFrame:
    """Add derived columns to certification DataFrame.

    Columns added:
        roi_per_week           — salary lift divided by weeks to earn
        roi_per_dollar         — salary lift divided by (cost + $1 to avoid div/0)
        total_investment_score — opportunity cost in dollars: (weeks * hrs * rate) + exam fee

    Args:
        df: Raw cert DataFrame from load_cert_data().
        hourly_rate: Assumed hourly opportunity cost for study time (default $40).

    Returns:
        pd.DataFrame: Input DataFrame with three new columns appended.
    """
    df = df.copy()
    df["roi_per_week"]           = df["est_salary_lift"] / df["weeks_to_earn"]
    df["roi_per_dollar"]         = df["est_salary_lift"] / (df["cost_usd"] + 1)
    df["total_investment_score"] = (df["weeks_to_earn"] * hourly_rate) + df["cost_usd"]
    return df


# ── Quality reporting ─────────────────────────────────────────────────────────

def run_data_quality_report(df: pd.DataFrame, name: str) -> None:
    """Print a formatted data quality report for a DataFrame.

    Reports row/column counts, null totals, duplicate counts, and per-column
    descriptive statistics for numeric columns.

    Args:
        df: Any pandas DataFrame to audit.
        name: Human-readable label printed in the report header.

    Returns:
        None — output is printed to stdout.
    """
    width = 60
    print(f"\n{'─' * width}")
    print(f"  Data Quality Report — {name}")
    print(f"{'─' * width}")
    print(f"  Rows       : {len(df)}")
    print(f"  Columns    : {len(df.columns)}")
    print(f"  Nulls      : {df.isnull().sum().sum()}")
    print(f"  Duplicates : {df.duplicated().sum()}")
    print()

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        mn  = df[col].min()
        mx  = df[col].max()
        mu  = df[col].mean()
        std = df[col].std()

        if df[col].max() > 1000:
            print(f"  {col:<25} range: ${mn:,.0f}–${mx:,.0f} | mean: ${mu:,.0f} | std: ${std:,.0f}")
        else:
            print(f"  {col:<25} range: {mn:.1f}–{mx:.1f}   | mean: {mu:.1f}   | std: {std:.1f}")

    print(f"{'─' * width}\n")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _validate_dataframe(df: pd.DataFrame, required_cols: list) -> None:
    """Raise ValueError if required columns are missing or nulls are present.

    Args:
        df: DataFrame to validate.
        required_cols: List of column names that must exist and be non-null.

    Returns:
        None
    """
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    nulls = df[required_cols].isnull().sum().sum()
    if nulls > 0:
        raise ValueError(f"Found {nulls} null values in required columns.")
