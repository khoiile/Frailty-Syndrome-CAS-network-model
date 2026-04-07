import numpy as np
import pandas as pd
from pathlib import Path

# Dataset path
BASE_DIR = Path(__file__).parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "frailty_dataset.csv"


THRESHOLDS = {
    "weakness":     20.0,    # kg grip strength
    "slowness":      0.8,    # m/s gait speed
    "low_activity": 270.0,   # kcal/week
}


def load_dataset(path=DATASET_PATH):
    """Load and return the frailty dataset."""
    df = pd.read_csv(path, index_col="subject_id")
    return df


def binarize_subject(row):
    #Convert a single subject's raw measures to binary deficit states.

    return {
        "weakness":     1 if row["weakness"]     < THRESHOLDS["weakness"]     else 0,
        "slowness":     1 if row["slowness"]      < THRESHOLDS["slowness"]     else 0,
        "low_activity": 1 if row["low_activity"]  < THRESHOLDS["low_activity"] else 0,
        "exhaustion":   int(row["exhaustion"]),
    }


def get_group_initial_state(df, group):
    """
    Get the most common binary state pattern for a frailty group.
    Uses majority vote per node across all subjects in the group.

    Returns dict: {node_name: 0 or 1}
    """
    subset = df[df["frailty_group"] == group]
    binary_rows = subset.apply(binarize_subject, axis=1, result_type="expand")
    # Majority vote: if >50% of group has deficit, node starts damaged
    return {col: int(binary_rows[col].mean() > 0.5) for col in binary_rows.columns}


def get_group_damage_probability(df, group):
    """
    Get the fraction of subjects with each deficit in a frailty group.
    Useful for stochastic initialization.

    Returns dict: {node_name: probability}
    """
    subset = df[df["frailty_group"] == group]
    binary_rows = subset.apply(binarize_subject, axis=1, result_type="expand")
    return binary_rows.mean().to_dict()


def print_dataset_summary(df):
    """Print dataset overview."""
    print(f"  Subjects : {len(df)}  |  Columns: {list(df.columns)}")
    print(f"\n  Distribution:")
    print(df["frailty_group"].value_counts().to_string())
    print(f"\n  Fried thresholds used:")
    for k, v in THRESHOLDS.items():
        print(f"    {k}: < {v}")
    print(f"\n  Deficit prevalence by group:")
    for group in ["robust", "pre-frail", "frail"]:
        probs = get_group_damage_probability(df, group)
        print(f"    {group:10s}: {', '.join(f'{k}={v:.2f}' for k,v in probs.items())}")


try:
    dataset = load_dataset()
except FileNotFoundError:
    raise FileNotFoundError(
        f"Dataset not found at '{DATASET_PATH}'. "
        "Update DATASET_PATH at the top of data.py to the correct location."
    )


if __name__ == "__main__":
    print("── Dataset Summary ──")
    print_dataset_summary(dataset)
    print("\n── Initial states by group (majority vote) ──")
    for g in ["robust", "pre-frail", "frail"]:
        state = get_group_initial_state(dataset, g)
        print(f"  {g:10s}: {state}")
