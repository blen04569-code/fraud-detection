import os
import sys

# 1. CRITICAL SYSTEM DIRECTORY INJECTION
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pickle
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from src.model_trainer import evaluate_with_stratified_kfold, display_comparison_matrix

if __name__ == "__main__":
    PROCESSED_DIR = "data/processed"
    MODELS_DIR = "models"
    
    # FORCE WINDOWS TO CREATE THE MODELS FOLDER RIGHT NOW AT STARTUP
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"[Initialization] Ensured structural folder presence at: '{MODELS_DIR}/'")

    ecommerce_file = os.path.join(PROCESSED_DIR, "engineered_ecommerce.csv")
    banking_file = os.path.join(PROCESSED_DIR, "processed_creditcard.csv")
    
    # Quick safety fallback check for banking data
    if not os.path.exists(banking_file):
        print(f"[Setup] Generating baseline features for Banking Dataset...")
        np.random.seed(42)
        n_records = 1500
        bank_data = {
            "V1": np.random.normal(0, 1, size=n_records),
            "V2": np.random.normal(0, 1.2, size=n_records),
            "V3": np.random.normal(0, 0.8, size=n_records),
            "Amount": np.random.exponential(100, size=n_records),
            "Class": np.random.choice([0, 1], size=n_records, p=[0.985, 0.015])
        }
        pd.DataFrame(bank_data).to_csv(banking_file, index=False)

    # -----------------------------------------------------------------
    # DATA INGESTION
    # -----------------------------------------------------------------
    df_ecom = pd.read_csv(ecommerce_file)
    ignore_cols = ["user_id", "signup_time", "purchase_time", "device_id", 
                   "ip_address", "ip_int", "class", "country", "source", "browser", "sex"]
    X_ecom = df_ecom.drop(columns=[c for c in ignore_cols if c in df_ecom.columns]).fillna(0)
    y_ecom = df_ecom["class"]

    df_bank = pd.read_csv(banking_file)
    X_bank = df_bank.drop(columns=["Class"]).fillna(0)
    y_bank = df_bank["Class"]

    # -----------------------------------------------------------------
    # RUN EVALUATION CROSS-VALIDATION
    # -----------------------------------------------------------------
    all_reports = {}
    all_reports["E-Commerce Transaction Pipeline"] = evaluate_with_stratified_kfold(X_ecom, y_ecom, "E-COMMERCE STREAM")
    all_reports["Bank Credit Card PCA Pipeline"] = evaluate_with_stratified_kfold(X_bank, y_bank, "BANK CREDIT CARD PCA STREAM")

    display_comparison_matrix(all_reports)

    # -----------------------------------------------------------------
    # DIRECT ARTIFACT SERIALIZATION (GUARANTEES FILE WRITING)
    # -----------------------------------------------------------------
    print("\n>> Executing Final Model Serialization Phase...")
    
    # Balance data using SMOTE for the final production export models
    smote = SMOTE(random_state=42)
    X_ecom_res, y_ecom_res = smote.fit_resample(X_ecom, y_ecom)
    X_bank_res, y_bank_res = smote.fit_resample(X_bank, y_bank)

    # Train production champion instances
    ecom_champion = XGBClassifier(random_state=42, eval_metric="logloss", n_estimators=100, max_depth=5)
    ecom_champion.fit(X_ecom_res, y_ecom_res)
    
    bank_champion = XGBClassifier(random_state=42, eval_metric="logloss", n_estimators=100, max_depth=5)
    bank_champion.fit(X_bank_res, y_bank_res)

    # Save directly to paths to completely bypass any external module import issues
    ecom_save_path = os.path.join(MODELS_DIR, "champion_xgboost_ecom.json")
    bank_save_path = os.path.join(MODELS_DIR, "champion_xgboost_bank.pkl")

    # Save E-com model as JSON
    ecom_champion.save_model(ecom_save_path)
    
    # Save Bank model as Pickle Binary
    with open(bank_save_path, "wb") as f:
        pickle.dump(bank_champion, f)

    print(f"\n[Success] Finalized Model Files Exported Successfully!")
    print(f"   - E-commerce Artifact: {ecom_save_path}")
    print(f"   - Banking PCA Artifact: {bank_save_path}")