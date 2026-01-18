# Predict the Success of Bank Telemarketing 📞🏦

![Status](https://img.shields.io/badge/Status-Completed-success) ![Rank](https://img.shields.io/badge/LB_Rank-6th_%2F_1256-gold) ![Metric](https://img.shields.io/badge/Metric-Macro_F1-blue) ![Language](https://img.shields.io/badge/Python-3.x-yellow)

## 🏆 Achievements
**Finished 6th on the Public Leaderboard** and **32nd on the Private Leaderboard** out of **1,256 participants** (16,543 total submissions).

* **Competition:** [Predict the Success of Bank Telemarketing (MLP Project T32024)](https://www.kaggle.com/competitions/predict-the-success-of-bank-telemarketing/overview)
* **Host:** iitmbscs2008p
* **Participant:** Sathvik V (Roll No: 22f2001468)

## 📖 Overview
This project was developed for the MLP Project T32024 competition. The objective was to build a machine learning model to predict whether a client would subscribe to a bank term deposit based on direct marketing campaign data (phone calls).

The solution utilizes a sophisticated pipeline involving **advanced feature engineering**, **automated feature selection**, and a **Rank Averaging Ensemble** of three GPU-accelerated models (XGBoost, LightGBM, Random Forest).

## 📊 Dataset
The dataset relates to direct marketing campaigns of a Portuguese banking institution.
* **Train Set:** 39,211 samples
* **Test Set:** 10,000 samples
* **Input Features:** 15 variables including client demographics (`age`, `job`, `marital`), financial status (`balance`, `loan`), and campaign details (`duration`, `pdays`, `previous`).
* **Target:** Binary variable (`yes`/`no`) indicating subscription success.

## 🛠️ Methodology

### 1. Data Processing & EDA
* **Missing Values:** Handled categorical missing values by filling them with an `'unknown'` token, preserving the signal of missingness.
* **Exploratory Data Analysis:** Analyzed target imbalance, feature distributions, and correlations to identify key predictors like `duration` and `poutcome`.

### 2. Advanced Feature Engineering
We expanded the feature space significantly to capture hidden signals:
* **Date Decompostion:** Extracted `month`, `year`, `day`, `day_of_week`, and created a binary `weekend` flag from the last contact date.
* **Log Transformations:** Applied `log(x + 1)` to skewed numerical features (`balance`, `duration`, `campaign`, `pdays`) to improve model convergence.
* **Interaction Features:** Created crossed features (e.g., `job_education`, `housing_loan`) to capture complex non-linear relationships.
* **Domain Specifics:** Converted the `-1` value in `pdays` (client not previously contacted) into a specific binary flag `pdays_contacted`.

### 3. Feature Selection
To counter the "Curse of Dimensionality" from creating hundreds of interaction features:
* Used **LightGBM** to estimate feature importance.
* Applied `SelectFromModel` to automatically retain only the **Top 100** most predictive features, discarding noise and preventing overfitting.

### 4. Model Training & Ensembling
We trained three distinct classifiers using **Stratified K-Fold Cross-Validation** and **RandomizedSearchCV**:
1.  **XGBoost:** (GPU-enabled) Optimized for speed and accuracy.
2.  **LightGBM:** (GPU-enabled) Efficient gradient boosting for categorical data.
3.  **Random Forest:** (CPU-parallelized) Bagging ensemble to reduce variance.

**Ensemble Strategy:**
We validated multiple strategies (Simple Average, Weighted Average) and selected **Rank Averaging** as the winner. This method averages the *ranks* of the predicted probabilities rather than the raw values, making the ensemble robust to calibration differences between models.

### 5. Optimization (The "Secret Sauce")
* **Metric:** Optimized for **ROC AUC** during training to ensure high-quality probability ranking.
* **Threshold Tuning:** Instead of a default 0.5 cut-off, we performed a dynamic search on the validation set to find the exact probability threshold (approx **0.79**) that maximized the **Macro F1 Score**.

## 📈 Performance
* **Validation F1 Macro:** 0.7837
* **Optimal Threshold:** 0.79
* **Public Leaderboard Rank:** 6th

## 🚀 How to Run
1.  Clone the repository.
2.  Ensure you have the required libraries: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `imbalanced-learn`.
3.  Place `train.csv` and `test.csv` in the input directory.
4.  Run the notebook `22f2001468-notebook-t32024.ipynb`.

---
*Created by Sathvik V*
