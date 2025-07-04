# 🏦 Predict the Success of Bank Telemarketing  
**MLP Project T32024** | Competition by `iitmbscs2008p`

## 📌 Objective  
Build a classification model to predict whether a client will subscribe to a term deposit based on marketing call data. The goal is to maximize the **F1 score (macro)** on the test set.

---

## 📁 Dataset Description  

The data is based on direct marketing campaigns of a banking institution via phone calls. Often, multiple contacts with the same client were made. The goal is to predict whether the client will subscribe to a term deposit (`"yes"` or `"no"`).

### 🔢 Files Provided
- `train.csv`: Training dataset
- `test.csv`: Test dataset (no target column)
- `sample_submission.csv`: Required submission format

### 🎯 Target Variable
- `target`: Whether the client subscribed to the deposit plan (binary: `"yes"` / `"no"`)

### 🔍 Input Variables
- `last contact date`: Date of last contact
- `age`: Age of client (numeric)
- `job`, `marital`, `education`, `default`, `housing`, `loan`: Client profile attributes (categorical/binary)
- `balance`, `duration`, `campaign`, `pdays`, `previous`: Numeric campaign stats
- `contact`, `poutcome`: Contact type & previous campaign result

---

## 🧠 Modeling Approach  

### 🔨 Tools Used  
- Python libraries: `pandas`, `numpy`, `scikit-learn`, `imbalanced-learn`, `xgboost`, `lightgbm`, `seaborn`, `matplotlib`

### 🧪 Data Preprocessing
- Imputed missing categorical values with `"unknown"`
- Extracted date features: `contact_month`, `contact_day`, `contact_dayofweek`, `contact_period`
- Engineered features:
  - Interaction features: `job_marital`, `housing_loan`, `campaign_outcome`
  - Log transformation on skewed numerical features: `balance_log`, `duration_log`, etc.
- One-hot encoding of categorical variables with `OneHotEncoder`
- Converted target values: `"yes"` → `1`, `"no"` → `0`

### ⚖️ Class Imbalance  
Used **SMOTE** to oversample the minority class before model training.

### 🤖 Models Trained  
- **XGBoostClassifier**
- **LightGBMClassifier**
- **RandomForestClassifier**

All models were wrapped in a pipeline (`ImbPipeline`) with SMOTE and tuned using optimal hyperparameters.

### 🧩 Ensemble Strategies  
Tested multiple ensemble methods:
- Weighted Average
- Simple Average
- Soft Voting
- Hard Voting
- Stacking (meta-model with Logistic Regression)

Each method was evaluated using **F1 Score (macro)** on validation data. The best-performing strategy was selected for final test prediction.

---

## 📊 Evaluation & Visualization
- Plotted ROC curves for each ensemble method
- Visualized confusion matrices
- Displayed feature importance for XGBoost, LightGBM, and Random Forest
- Compared F1 scores across ensemble methods using bar plots

---

## 📤 Submission Format

| id | target |
|----|--------|
| 0  | "yes"  |
| 1  | "no"   |
| 2  | "no"   |

File saved as `submission.csv`.

---

## 🔗 Notes
- Notebook name: `YourRollNo-notebook-t32024` (e.g., `22f2001468-notebook-t32024`)
- Submission shared privately with `iitmbscs2008p`
- Deadline: **November 30, 2024**
- Metric: `f1_score(average='macro')`

---

## 🏁 Final Remarks  
This competition gave hands-on experience with:
- Feature engineering & handling imbalanced data
- Model ensembling and performance visualization
- Real-world business classification challenges in banking

The ensemble model with optimized thresholding provided the best generalization performance on validation data.
