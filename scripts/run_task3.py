import os
import sys

# Append project root directory to system lookup paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from xgboost import XGBClassifier
from src.explainability import run_comprehensive_explainability

if __name__ == "__main__":
    PROCESSED_DIR = "data/processed"
    MODELS_DIR = "models"
    
    ecommerce_file = os.path.join(PROCESSED_DIR, "engineered_ecommerce.csv")
    model_file = os.path.join(MODELS_DIR, "champion_xgboost_ecom.json")
    
    if not os.path.exists(model_file) or not os.path.exists(ecommerce_file):
        print("[Error] Missing dependencies. Please run Task 2 first to generate models and processing matrices.")
        sys.exit(1)
        
    # Ingest data elements
    df = pd.read_csv(ecommerce_file)
    ignore_cols = ["user_id", "signup_time", "purchase_time", "device_id", 
                   "ip_address", "ip_int", "class", "country", "source", "browser", "sex"]
    
    X = df.drop(columns=[c for c in ignore_cols if c in df.columns]).fillna(0)
    y = df["class"]
    
    # Load your production champion model binary
    model = XGBClassifier()
    model.load_model(model_file)
    
    # Execute full task suite
    df_builtin, df_last_local = run_comprehensive_explainability(model, X, y)
    
    print("\n[Success] Task 3 complete! Comparison files and visual plots are exported to /notebooks.")