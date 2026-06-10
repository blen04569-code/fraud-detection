import numpy as np
import pandas as pd


def clean_and_coerce(file_path):
    """Loads a dataset, drops duplicate entries, and runs missing value imputations."""
    df = pd.read_csv(file_path)
    df = df.drop_duplicates()

    if "signup_time" in df.columns and "purchase_time" in df.columns:
        df["signup_time"] = pd.to_datetime(df["signup_time"])
        df["purchase_time"] = pd.to_datetime(df["purchase_time"])

    # Protect boundaries against extreme monetary skew using Median
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    return df


def merge_geolocation(df_fraud, df_ip):
    """Vectorized geolocation country mapping utilizing range intervals."""
    df_fraud = df_fraud.copy()

    # Lambda-free bitwise IP integer parsing
    def ip_to_int(ip):
        try:
            p = list(map(int, str(ip).split(".")))
            return (p[0] << 24) + (p[1] << 16) + (p[2] << 8) + p[3] if len(p) == 4 else 0
        except Exception:
            return 0

    df_fraud["ip_int"] = df_fraud["ip_address"].apply(ip_to_int)
    df_ip = df_ip.sort_values(by="lower_bound_ip_address").reset_index(
        drop=True
    )

    df_fraud = df_fraud.sort_values(by="ip_int")
    merged_df = pd.merge_asof(
        df_fraud,
        df_ip,
        left_on="ip_int",
        right_on="lower_bound_ip_address",
        direction="backward",
    )

    # Invalidate out-of-bounds interval leakage
    mask = (merged_df["ip_int"] > merged_df["upper_bound_ip_address"]) | (
        merged_df["lower_bound_ip_address"].isna()
    )
    merged_df.loc[mask, "country"] = "Unknown"

    return merged_df.drop(
        columns=["lower_bound_ip_address", "upper_bound_ip_address"],
        errors="ignore",
    )