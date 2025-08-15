# Credit Card Default Prediction (Machine Learning)

**Goal:** Predict whether a credit card client will default in the next billing cycle using demographic, behavioral, and financial features.

On the UCI “Default of Credit Card Clients” dataset (30,000 records), Random Forest outperforms Logistic Regression at identifying defaulters.  
- **Accuracy:** ≈ 0.81 (both)  
- **AUC:** RF **0.757** vs LR **0.715**  
- **Recall (class 1 – defaulters):** RF **0.36** vs LR **0.24**  
- **Key driver:** Recent repayment status (**PAY_0**)

---

## 1) Problem & Context
Credit risk teams need early, reliable default predictions to reduce losses and tailor interventions (e.g., limit changes, outreach). We build and compare models to predict next-month default as a binary classification problem.

---

## 2) Data
- **Source:** UCI Machine Learning Repository — *Default of Credit Card Clients* (Yeh & Lien, 2009).  
- **Size:** 30,000 rows; 23 features + target.  
- **Target:** `default_payment_next_month` (1 = default, 0 = no default).  
- **Feature groups:**  
  - Demographics: `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`  
  - Credit limit & bills: `LIMIT_BAL`, `BILL_AMT1..6`  
  - Payments: `PAY_AMT1..6`  
  - Repayment history (status codes): `PAY_0..PAY_6`  

> Note: The dataset is imbalanced (~22% default, ~78% non-default).

---

## 3) Approach
1. **EDA & Cleaning**
   - Drop `ID`, standardize column names, sanity checks for missing values (none found).
   - Store raw and processed data in SQLite (`credit_card_db.db`) for traceability.
2. **Train/Test Split**
   - Stratified split (70/30) to preserve class ratios; `random_state=42`.
3. **Preprocessing**
   - **Scaling** applied only to Logistic Regression (StandardScaler).
4. **Models**
   - **Logistic Regression** (baseline; also tried `class_weight='balanced'`).
   - **Random Forest** (tuned with GridSearchCV; best: `n_estimators=200`, `min_samples_split=5`, `max_depth=None`).
5. **Evaluation**
   - Metrics: Accuracy, Precision/Recall/F1, ROC–AUC, Confusion Matrix.
   - Additional: McNemar’s test for paired comparison; RMSE on probabilities.

---

## 4) Results
- **Overall**
  - **Accuracy:** ~0.81 (LR and RF)
  - **AUC:** LR **0.715**, RF **0.757**
  - **Recall (defaulters):** LR **0.24**, RF **0.36**
  - **RMSE (probas):** LR **0.3822**, RF **0.3749**
  - **McNemar’s test:** χ² ≈ **1.545**, p ≈ **0.214** (no statistically significant difference in error patterns)

- **Interpretation**
  - **RF** provides materially higher **recall** for the minority (defaulters) with a better AUC—preferable for risk mitigation.
  - **Top signals:** Repayment history, especially **`PAY_0`** (most recent status); also `LIMIT_BAL` and billing/payment magnitudes.

- **Recommendation**
  - Deploy **Random Forest**; perform **threshold tuning** and/or **cost-sensitive** adjustments to align with business loss functions.

