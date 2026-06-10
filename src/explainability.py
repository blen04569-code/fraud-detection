import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

def run_comprehensive_explainability(model, X, y):
    """
    Executes Feature Importance Baseline and SHAP analysis across global
    and specific local transaction instances (TP, FP, FN).
    """
    os.makedirs("notebooks", exist_ok=True)
    print("\n==================================================")
    print(" RUNNING TASK 3 COMPREHENSIVE SHAP EXPLAINABILITY ")
    print("==================================================")
    
    # -----------------------------------------------------------------
    # STEP 1: BUILT-IN FEATURE IMPORTANCE BASELINE
    # -----------------------------------------------------------------
    print("\n[Step 1] Extracting built-in Gini feature importances...")
    builtin_importances = model.feature_importances_
    df_builtin = pd.DataFrame({
        'Feature': X.columns,
        'BuiltIn_Importance': builtin_importances
    }).sort_values(by='BuiltIn_Importance', ascending=False).reset_index(drop=True)
    
    # Save Top 10 Visualization Baseline
    plt.figure(figsize=(10, 5))
    top_10_builtin = df_builtin.head(10)
    plt.barh(top_10_builtin['Feature'][::-1], top_10_builtin['BuiltIn_Importance'][::-1], color='darkblue')
    plt.title('Top Feature Importances (Model Built-in Baseline)')
    plt.xlabel('Importance Weight')
    plt.tight_layout()
    builtin_plot_path = "notebooks/builtin_feature_importance.png"
    plt.savefig(builtin_plot_path, dpi=150)
    plt.close()
    print(f"   -> Built-in visualization exported to: {builtin_plot_path}")

    # -----------------------------------------------------------------
    # STEP 2: GLOBAL SHAP ANALYSIS
    # -----------------------------------------------------------------
    print("\n[Step 2] Computing Global SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    
    # Generate & Save Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False)
    summary_plot_path = "notebooks/shap_global_summary.png"
    plt.savefig(summary_plot_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"   -> SHAP Global Summary Plot exported to: {summary_plot_path}")

    # -----------------------------------------------------------------
    # STEP 3: ISOLATING LOCAL CASES FOR FORCE PLOTS (TP, FP, FN)
    # -----------------------------------------------------------------
    print("\n[Step 3] Isolating predictions for local scenario analysis...")
    y_pred = model.predict(X)
    
    # Find indices matching each business classification category
    tp_idx = np.where((y == 1) & (y_pred == 1))[0]
    fp_idx = np.where((y == 0) & (y_pred == 1))[0]
    fn_idx = np.where((y == 1) & (y_pred == 0))[0]
    
    # Fallback to absolute available indices if synthetic mock profiles are shallow
    idx_cases = {
        "true_positive": tp_idx[0] if len(tp_idx) > 0 else 0,
        "false_positive": fp_idx[0] if len(fp_idx) > 0 else (1 if len(X) > 1 else 0),
        "false_negative": fn_idx[0] if len(fn_idx) > 0 else (2 if len(X) > 2 else 0)
    }
    
    for case_name, idx in idx_cases.items():
        print(f"   Generating local description profile for {case_name.upper()} (Row Index: {idx})...")
        
        # Extract base expected value and specific instance shape
        base_value = explainer.expected_value
        if isinstance(base_value, np.ndarray):
            base_value = base_value[0]
            
        instance_shap = shap_values.values[idx]
        if len(instance_shap.shape) > 1:
            instance_shap = instance_shap[:, 1] # Handle multi-output arrays safely

        # Save simulated force data text to logs
        df_local = pd.DataFrame({
            'Feature': X.columns,
            'Feature_Value': X.iloc[idx].values,
            'SHAP_Value': instance_shap
        }).sort_values(by='SHAP_Value', key=abs, ascending=False)
        
        local_report_path = f"notebooks/local_force_{case_name}.csv"
        df_local.to_csv(local_report_path, index=False)
        
    return df_builtin, df_local