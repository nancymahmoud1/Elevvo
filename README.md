# Elevvo Tasks

## 📊 Loan Approval Prediction

This notebook builds a machine learning model to **predict loan approval status** based on applicant financial, credit, and employment information. The project uses a cleaned dataset from Kaggle and compares Logistic Regression and Decision Tree model.

---

### 🔧 Project Workflow

1. **📥 Data Loading**

   * Dataset (Loan-Approval-Prediction-Dataset) is downloaded using `kagglehub`.
   * The data includes columns like: `income_annum`, `loan_amount`, `cibil_score`, `loan_term`, `education`, `self_employed`, and more.

2. **🧼 Data Preprocessing**

   * Missing values are handled.
   * Categorical features like `education` and `self_employed` are encoded using One-Hot or Label Encoding.
   * Whitespace is stripped from column names.
   * Target variable `loan_status` is encoded as 0 (Rejected) and 1 (Approved).
   * Features and labels are split into `X` and `y`.

3. **🧪 Train-Test Split**

   * The dataset is split into 80% training and 20% testing using `train_test_split` from scikit-learn.
   * `stratify=y` ensures the class distribution is preserved in both sets.

4. **📈 Model Training**

   * **Logistic Regression** with:

     * `solver="liblinear"`
     * `class_weight="balanced"`
     * `random_state=42`, `max_iter=500`
   * **Decision Tree Classifier** is also trained for comparison.

5. **📊 Evaluation**

   * Accuracy and ROC-AUC are computed for both models.
   * Classification reports are generated.
   * Visualization includes:

     * Confusion matrices
     * ROC Curves

---

## 🤖 Model Comparison: Logistic Regression vs. Decision Tree

| Model               | Accuracy | ROC-AUC |
| ------------------- | -------- | ------- |
| Logistic Regression | 0.7400   | 0.8287  |
| Decision Tree       | 0.9707   | 0.9668  |

---

### 📦 Requirements

* Python 3.7+
* Libraries:

  * `pandas`
  * `numpy`
  * `scikit-learn`
  * `matplotlib`
  * `seaborn`
  * `kagglehub` (for dataset download)

---
