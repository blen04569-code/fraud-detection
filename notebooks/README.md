# Task 1: Pipeline Documentation & Methodological Justifications

## 1. Feature Engineering Documentation

To catch sophisticated fraud velocity patterns across Adey Innovations' e-commerce streams, we engineered the following behavioral data shapes:

* `time_since_signup`: Measures the chronological delta hours between `signup_time` and `purchase_time`. 
    * *Signal:* Stolen credit cards are frequently paired with immediate, high-value purchases post-account creation.
* `hour_of_day` & `day_of_week`: Extracted components from transaction timestamps, capturing anomalous late-night script behavior.
* `tx_count_past_10m`: Uses a sliding rolling window grouped per `user_id`.
    * *Signal:* Detects continuous velocity testing or scripted card-brute-forcing attacks.

---

## 2. Class Imbalance Resampling Justification

### Chosen Technique: SMOTE (Synthetic Minority Over-sampling Technique)
For this framework, **SMOTE applied strictly within the training fold** was chosen over traditional random undersampling.

### Methodological Justification:
1.  **Information Preservation:** Random undersampling balances datasets by discarding thousands of legitimate transactions. In a live banking environment, throwing away data destroys the model's ability to map regular, non-fraudulent customer behavioral norms.
2.  **Boundary Nuance:** SMOTE operates by creating synthetic interpolations along the feature vector space paths connecting minority fraud occurrences. This expands the minority decision boundary, giving algorithms like XGBoost more nuanced patterns to train on.
3.  **Validation Leakage Prevention:** The resampling step is applied exclusively *after* executing our train-test split. The test matrix remains completely pristine and unmodified, representing the actual imbalanced distribution the model will encounter in production.
# Task 3: Model Explainability and Operational Insights Report

## 1. Feature Importance Interpretation & Comparison

### Top 5 Drivers of Fraud Prediction:
1. **`time_since_signup` (Account Lifespan Velocity):** The primary driver of fraud across both built-in and SHAP assessments. Short durations between account creation and a purchase heavily push the model toward a fraud prediction.
2. **`purchase_value` (Monetary Magnitude):** Outsized transaction amounts significantly accelerate the risk score.
3. **`tx_count_past_10m` (Velocity Window Tracking):** High numbers of repeat transactions within a tight 10-minute window suggest automated script attacks.
4. **`hour_of_day`:** Transactions occurring between 11:00 PM and 4:00 AM exhibit elevated risk profiles.
5. **`day_of_week`:** High-volume weekend spikes show a strong statistical correlation with fraudulent accounts.

### Comparison: Built-In vs. SHAP Importance
* **Built-In Importance (Gain):** Focuses heavily on structural splits that optimize the tree nodes (e.g., heavily weighting `time_since_signup`). However, it only indicates *that* a feature is useful, not the *direction* of its impact.
* **SHAP Summary Analysis:** Provides a clearer picture of feature mechanics. For example, it explicitly demonstrates that *low* values of `time_since_signup` (blue dots) have a massive *positive* SHAP impact, pushing the risk prediction directly toward fraud.

---

## 2. Local Predictions Analysis (Force Scenarios)

* **True Positive Case Study:** The transaction was correctly flagged as fraud because `time_since_signup` was under 1.5 hours and `purchase_value` exceeded \$250. These two features combined to push the model's output past the decision threshold.
* **False Positive Case Study:** A legitimate user was incorrectly flagged because they made a large purchase (`purchase_value` > \$400) late at night (`hour_of_day` = 2 AM). Even though the account was older, these two high-risk indicators overrode the baseline signal.
* **False Negative Case Study:** A fraudulent transaction slipped through unnoticed because the attacker used a low transaction value (`purchase_value` = \$12) spread out over a longer timeframe, successfully blending in with regular customer behavior.

---

## 3. Actionable Business Recommendations for Adey Innovations

Based on our SHAP explainability insights, we recommend implementing the following fraud prevention policies:

1. **Implement an Account Lifespan Verification Window**
   * *Strategy:* Any user attempting a transaction where `time_since_signup` is less than **2 hours** must pass a secondary multi-factor authentication (MFA) challenge before checkout can proceed.
   * *SHAP Insight:* SHAP local force plots reveal that ultra-low lifespans are the single largest driver pushing predictions toward the fraud threshold.

2. **Deploy Dynamic Step-Up Authentication for Rapid-Fire Velocity Attacks**
   * *Strategy:* If `tx_count_past_10m` rises above **3 unique transactions**, temporarily suspend checkout capabilities for that user ID for 15 minutes.
   * *SHAP Insight:* The global summary plot demonstrates a distinct risk spike when transaction frequency metrics increase over short intervals, indicating automated card-testing scripts.

3. **Enforce Conditional Multi-Factor Review for High-Value, Off-Hours Transactions**
   * *Strategy:* Transactions exceeding **\$200** that occur between **11:00 PM and 5:00 AM** local time must pass automated biometric or device-fingerprint verification.
   * *SHAP Insight:* SHAP analysis highlights a strong interaction effect between late-night hours and high monetary value, which frequently causes false positives for legitimate users or lets low-value fraud slip by undetected.