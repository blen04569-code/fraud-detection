import os
import sys
import numpy as np
import pandas as pd

# Ensure project root is available to runtime pathing variables
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_preprocessor import clean_and_coerce, merge_geolocation
from src.feature_engineer import build_ecommerce_features
from src.model_transformer import scale_encode_and_resample


def populate_empty_files(fraud_path, ip_path):
    """Checks if data files are empty or missing, and populates them

    with realistic mock data for testing.
    """
    np.random.seed(42)
    n_records = 1000

    # If Fraud_Data.csv is missing or completely empty (0 bytes)
    if not os.path.exists(fraud_path) or os.path.getsize(fraud_path) == 0:
        print(f"[Setup] '{fraud_path}' is empty/missing. Populating test data...")
        fraud_data = {
            "user_id": np.random.randint(10000, 12000, size=n_records),
            "signup_time": pd.date_range(start="2026-01-01", periods=n_records, freq="5min"),
            "purchase_time": pd.date_range(start="2026-01-02", periods=n_records, freq="4min"),
            "purchase_value": np.random.exponential(scale=60, size=n_records),
            "device_id": [f"DEV_{x}" for x in np.random.randint(100, 300, size=n_records)],
            "source": np.random.choice(["SEO", "Ads", "Direct"], size=n_records),
            "browser": np.random.choice(["Chrome", "Safari", "Firefox"], size=n_records),
            "sex": np.random.choice(["M", "F"], size=n_records),
            "age": np.random.randint(18, 70, size=n_records),
            "ip_address": [
                f"{np.random.choice([16, 33, 50, 67])}.10.20.{np.random.randint(1,254)}"
                for _ in range(n_records)
            ],
            "class": np.random.choice([0, 1], size=n_records, p=[0.96, 0.04]),
        }
        pd.DataFrame(fraud_data).to_csv(fraud_path, index=False)

    # If IpAddress_to_Country.csv is missing or completely empty
    if not os.path.exists(ip_path) or os.path.getsize(ip_path) == 0:
        print(f"[Setup] '{ip_path}' is empty/missing. Populating lookup data...")
        ip_map_data = {
            "lower_bound_ip_address": [268435456, 553648128, 838860800, 1124073472],
            "upper_bound_ip_address": [285212671, 570425343, 855638015, 1140850687],
            "country": ["United States", "Canada", "United Kingdom", "Ethiopia"],
        }
        pd.DataFrame(ip_map_data).to_csv(ip_path, index=False)


if __name__ == "__main__":
    print("[Pipeline] Commencing Full Execution Cycle...")

    RAW_DIR = "data/raw"
    PROCESSED_DIR = "data/processed"

    # Define resource paths
    fraud_path = os.path.join(RAW_DIR, "Fraud_Data.csv")
    ip_path = os.path.join(RAW_DIR, "IpAddress_to_Country.csv")

    # Create directories if they don't exist
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Run the safety populator to catch EmptyDataErrors
    populate_empty_files(fraud_path, ip_path)

    # Execute step-by-step modular transformations
    print("[Pipeline] Ingesting and cleaning files...")
    df_fraud = clean_and_coerce(fraud_path)
    df_ip = clean_and_coerce(ip_path)

    print("[Pipeline] Performing geolocation range-based lookup...")
    df_mapped = merge_geolocation(df_fraud, df_ip)
    
    print("[Pipeline] Engineering behavioral & velocity features...")
    df_features = build_ecommerce_features(df_mapped)

    # Save artifact checkpoints to disk
    processed_file_path = os.path.join(PROCESSED_DIR, "engineered_ecommerce.csv")
    df_features.to_csv(processed_file_path, index=False)
    print(f"[Pipeline] Clean processed artifacts exported to: {processed_file_path}")

    # Run downstream transform operations
    ignore_cols = [
        "user_id", "signup_time", "purchase_time", "device_id", 
        "ip_address", "ip_int", "class", "country", "source", "browser", "sex"
    ]
    
    print("[Pipeline] Preparing modeling matrices (Scaling & SMOTE)...")
    X_tr, X_te, y_tr, y_te = scale_encode_and_resample(
        df_features, target_col="class", drop_cols=ignore_cols
    )
    
    print(f"\n[Success] Final Matrix Dimensions:")
    print(f"   - Balanced Train Features Shape: {X_tr.shape}")
    print(f"   - Balanced Train Labels Count:   {len(y_tr)}")
    print(f"   - Pristine Test Features Shape:  {X_te.shape}")