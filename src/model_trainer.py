import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


def evaluate_with_stratified_kfold(X, y, dataset_name, k=5):
    """Executes a 5-Fold Stratified Cross-Validation loop, training both

    Baseline Logistic Regression and an Advanced XGBoost Ensemble.
    Resampling (SMOTE) is applied strictly inside the training folds.
    """
    print(f"\n==================================================")
    print(f" RUNNING 5-FOLD STRATIFIED CV FOR: {dataset_name}")
    print(f"==================================================")

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    # Dictionary to collect scores across folds
    model_metrics = {
        "Logistic Regression": {"precision": [], "recall": [], "f1": [], "pr_auc": [], "cms": []},
        "XGBoost Ensemble": {"precision": [], "recall": [], "f1": [], "pr_auc": [], "cms": []}
    }

    # Instantiating estimators with basic hyperparameter tuning applied to XGBoost
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
        "XGBoost Ensemble": XGBClassifier(
            random_state=42,
            eval_metric="logloss",
            n_estimators=150,       
            max_depth=5,            
            learning_rate=0.08,     
            subsample=0.8           
        )
    }

    # Cross-validation loop
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        # Isolate split groups based on structural index partitions safely
        X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # CRITICAL PROTECTION PATCH: SMOTE cannot process NaNs natively.
        # Fill any runtime feature blanks with 0 to keep the coordinate space uniform.
        X_train = X_train.fillna(0)
        X_val = X_val.fillna(0)

        # Apply SMOTE ONLY on the training fold to prevent data leakage
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        for name, model in models.items():
            # Train model on the balanced fold arrays
            model.fit(X_train_res, y_train_res)

            # Predict on the pristine validation fold
            y_pred = model.predict(X_val)
            y_prob = model.predict_proba(X_val)[:, 1]

            # Calculate raw validation scores
            prec = precision_score(y_val, y_pred, zero_division=0)
            rec = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            
            # Compute Precision-Recall Curve Area (PR-AUC)
            p_curve, r_curve, _ = precision_recall_curve(y_val, y_prob)
            pr_auc = auc(r_curve, p_curve)
            
            # Store structural confusion matrices
            cm = confusion_matrix(y_val, y_pred)

            # Store fold outputs
            model_metrics[name]["precision"].append(prec)
            model_metrics[name]["recall"].append(rec)
            model_metrics[name]["f1"].append(f1)
            model_metrics[name]["pr_auc"].append(pr_auc)
            model_metrics[name]["cms"].append(cm)

    # Compute Summary Statistics
    summary_report = {}
    for name in models.keys():
        summary_report[name] = {
            "Precision Mean": np.mean(model_metrics[name]["precision"]),
            "Precision Std": np.std(model_metrics[name]["precision"]),
            "Recall Mean": np.mean(model_metrics[name]["recall"]),
            "Recall Std": np.std(model_metrics[name]["recall"]),
            "F1-Score Mean": np.mean(model_metrics[name]["f1"]),
            "F1-Score Std": np.std(model_metrics[name]["f1"]),
            "PR-AUC Mean": np.mean(model_metrics[name]["pr_auc"]),
            "PR-AUC Std": np.std(model_metrics[name]["pr_auc"]),
            # Accumulate full confusion matrices across all folds
            "Total Confusion Matrix": sum(model_metrics[name]["cms"])
        }
        
    return summary_report


def display_comparison_matrix(reports):
    """Outputs a side-by-side tabular report of models performance metrics."""
    for dataset_name, report in reports.items():
        print(f"\n--- SIDE-BY-SIDE COMPARISON MATRIX: {dataset_name} ---")
        rows = []
        for model_name, metrics in report.items():
            rows.append({
                "Model": model_name,
                "PR-AUC (Mean ± Std)": f"{metrics['PR-AUC Mean']:.4f} ± {metrics['PR-AUC Std']:.3f}",
                "F1-Score (Mean ± Std)": f"{metrics['F1-Score Mean']:.4f} ± {metrics['F1-Score Std']:.3f}",
                "Recall (Mean ± Std)": f"{metrics['Recall Mean']:.4f} ± {metrics['Recall Std']:.3f}",
                "Precision (Mean ± Std)": f"{metrics['Precision Mean']:.4f} ± {metrics['Precision Std']:.3f}"
            })
        df_table = pd.DataFrame(rows)
        print(df_table.to_string(index=False))
        
        # Display aggregated confusion matrix details
        for model_name, metrics in report.items():
            cm = metrics["Total Confusion Matrix"]
            print(f"\n[{model_name} Cumulative Confusion Matrix]")
            print(f"   TN: {cm[0][0]:<6} | FP: {cm[0][1]} (Customer Insults)")
            print(f"   FN: {cm[1][0]:<6} | TP: {cm[1][1]} (Caught Fraud)")