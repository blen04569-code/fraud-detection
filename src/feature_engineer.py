import numpy as np
import pandas as pd


def build_ecommerce_features(df):
    """Generates transaction metrics and temporal latency deltas."""
    df = df.copy()

    # Compute account lifespan before purchase window
    df["time_since_signup"] = (
        df["purchase_time"] - df["signup_time"]
    ).dt.total_seconds() / 3600.0

    # Extract time traits
    df["hour_of_day"] = df["purchase_time"].dt.hour
    df["day_of_week"] = df["purchase_time"].dt.dayofweek

    # Transaction frequency count velocity window (10 Min Window)
    df = df.sort_values(by=["user_id", "purchase_time"])
    df["tx_count_past_10m"] = (
        df.groupby("user_id")
        .rolling(window="10min", on="purchase_time")["purchase_value"]
        .count()
        .reset_index(level=0, drop=True)
    )

    return df