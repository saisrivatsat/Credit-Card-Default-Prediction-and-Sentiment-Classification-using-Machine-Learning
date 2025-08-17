
## Credit Card Default Prediction (UCI) — A Comparative Machine Learning Study

### Overview

This project builds and evaluates machine-learning models to predict **next-month default** for credit-card customers using demographic, behavioral, and financial attributes. We benchmark **Logistic Regression** against a **Random Forest** on the UCI “Default of Credit Card Clients” dataset and report results through accuracy, precision/recall/F1, ROC–AUC, confusion matrices, RMSE on probabilities, and **McNemar’s test** for paired comparison.

### Data

* **Source:** UCI Machine Learning Repository — *Default of Credit Card Clients* (Yeh & Lien, 2009)
* **Size:** 30,000 rows; **23 predictors + 1 target**
* **Target:** `default_payment_next_month` (1 = default, 0 = no default)
* **Feature groups:**

  * Demographics: `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`
  * Credit limit & bills: `LIMIT_BAL`, `BILL_AMT1..6`
  * Payments: `PAY_AMT1..6`
  * Repayment history: `PAY_0..PAY_6`
* **Class imbalance:** \~**22%** default, **78%** non-default

### Problem & Context

Credit-risk teams need early and reliable default predictions to reduce losses and tailor interventions (e.g., limit adjustments, proactive outreach). We frame this as a **binary classification** problem and compare interpretable linear baselines to ensemble methods.

### Methodology

* **EDA & Cleaning:**

  * Drop `ID`, standardize column names, confirm no missing values.
  * Persist raw and processed frames to SQLite (`credit_card_db.db`) for traceability.
* **Split:** Stratified **70/30** train/test (preserve class ratios), `random_state=42`.
* **Preprocessing:**

  * **Scaling** (StandardScaler) applied **only** to Logistic Regression.
* **Models:**

  * **Logistic Regression** (baseline; also tried `class_weight='balanced'`).
  * **Random Forest** tuned via GridSearchCV → **best:** `n_estimators=200`, `min_samples_split=5`, `max_depth=None`.
* **Evaluation:**

  * Metrics: **Accuracy**, **Precision/Recall/F1**, **ROC–AUC**, **Confusion Matrix**
  * Extras: **RMSE** on predicted probabilities; **McNemar’s test** for paired error comparison.

### Results

**Overall performance**

* **Accuracy:** \~**0.81** (both LR and RF)
* **ROC–AUC:** **LR = 0.715**, **RF = 0.757**
* **Recall (defaulters, class 1):** **LR = 0.24**, **RF = 0.36**
* **RMSE (probabilities):** **LR = 0.3822**, **RF = 0.3749**
* **McNemar’s test:** χ² ≈ **1.545**, p ≈ **0.214** → No statistically significant difference in error **patterns** between LR and RF.

**Interpretation**

* **Random Forest** achieves **higher recall on the minority class** and a **better AUC**, which is typically preferable for risk-mitigation scenarios where **missing a defaulter is costly**.
* **Top signals:** Repayment history—especially **`PAY_0` (most recent status)**—followed by `LIMIT_BAL` and magnitudes in `BILL_AMT*` / `PAY_AMT*`.

### Recommendation

Deploy **Random Forest** for production scoring and tune the decision threshold (or apply cost-sensitive learning) to align with your loss function. Consider monitoring stability and fairness across customer segments.

### Repro 

```bash
# Environment
pip install pandas numpy scikit-learn matplotlib seaborn joblib statsmodels

# Data steps (as in notebook/scripts)
# - Load UCI dataset
# - Drop ID, standardize names, sanity checks
# - Stratified split (70/30)

# Modeling
# - Fit Logistic Regression (scaled features)
# - GridSearchCV RandomForest (n_estimators, min_samples_split, max_depth)

# Evaluation
# - classification_report, confusion_matrix, ROC–AUC
# - RMSE on predicted probabilities
# - McNemar’s test for model comparison

# Save artifacts
python - << 'PY'
import joblib
# assume best_rf and lr_pipeline are trained
joblib.dump(best_rf, "random_forest_default_model.pkl")
joblib.dump(lr_pipeline, "logistic_regression_default_pipeline.pkl")
PY
```

### Limitations & Next Steps

* **Imbalance:** Explore **class weights**, resampling, and **threshold tuning** by business cost.
* **Explainability:** Add **SHAP** for feature contributions in individual predictions.
* **Calibration:** Check probability calibration (e.g., Platt/Isotonic).
* **Robustness:** Perform temporal validation if time stamps available; assess drift and monitoring in production.

### References

* Yeh, I.C., & Lien, C.H. (2009). *UCI Default of Credit Card Clients* dataset.
* Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python*, JMLR.

**Summary:** Random Forest and Logistic Regression reach similar accuracy, but **Random Forest** offers **higher AUC and materially better recall for defaulters**, making it the preferred choice for cost-sensitive credit-risk applications.
