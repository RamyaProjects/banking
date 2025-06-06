#  Bank Marketing Prediction using Logistic Regression

**Project Overview**

This project aims to predict whether a client will subscribe to a term deposit based on direct marketing campaign data provided by a Portuguese bank. The model is built using Logistic Regression, with a special focus on feature selection, data balancing (SMOTE), and cross-validation to ensure robustness.

**Dataset Description**

- Size: ~41,000 records, 51 features after one-hot encoding
- Target Variable: y (binary: yes or no)
- Problem Type: Binary Classification

**Project Workflow**
1. Data Preprocessing & EDA
- Converted target variable y: yes → 1, no → 0
- Categorical variables transformed using one-hot encoding
- Boolean fields mapped: True → 1, False → 0
- Scaled numerical features using StandardScaler to ensure consistent feature magnitude

2. Handling Class Imbalance
- Applied SMOTE (Synthetic Minority Oversampling Technique) to balance training data:
  - Original ratio: no (89%) vs yes (11%)
  - After SMOTE: 50% / 50%

3.  Feature Selection
Applied Recursive Feature Elimination (RFE) with Logistic Regression to select the top 20 features contributing to prediction accuracy.

4.  Model Building & Evaluation

**Model 1: Logistic Regression with SMOTE**
- Trained using balanced dataset
- Evaluated on original (imbalanced) test set

Results:
  - Accuracy: 88.40%
  - Precision (class 1): 0.47
  - Recall (class 1): 0.22
  - F1 Score (class 1): 0.30
 Drawback: SMOTE caused the model to over-predict positives, leading to lower precision and recall on actual test data.

**Model 2: Logistic Regression (No SMOTE)**
- Used imbalanced training set with class ratio preserved
- Selected features using RFE

Results:
- Accuracy: 91.13%
- Precision (class 1): 0.67
- Recall (class 1): 0.42
- F1 Score (class 1): 0.52
-  Best model based on generalization performance and AUC score

5. Cross-Validation
- Performed 10-fold cross-validation to ensure model generalizes well.
- Achieved consistent accuracy across folds.

6.  ROC & AUC Analysis
- Plotted ROC Curve
- Achieved AUC = 0.91, indicating a strong classification capability.

**Final Recommendation**
- Best Model: Logistic Regression without SMOTE, trained on imbalanced data
- Evaluation Metric Focus: Precision, Recall, F1-score, AUC
- Ideal for situations where false positives/negatives have significant consequences (e.g., offering financial products)

