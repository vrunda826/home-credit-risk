# 🏦 Home Credit Default Risk Prediction

An end-to-end machine learning project that predicts whether a loan applicant is likely to default using historical financial and credit information from the Home Credit dataset.

The project covers the complete machine learning workflow—from exploratory data analysis and feature engineering on multiple relational tables to model training, explainability, and deployment through an interactive Streamlit application.

---

## Project Overview

Financial institutions need to evaluate the risk associated with every loan application. Incorrect decisions can either increase financial losses (approving risky customers) or reduce business opportunities (rejecting reliable customers).

This project builds a predictive model that estimates the probability of customer default using historical credit behavior and engineered features from multiple data sources.

---

## Dataset

**Home Credit Default Risk** (Kaggle)

The project combines information from multiple relational tables:

* Application Data
* Bureau
* Previous Applications
* Installment Payments
* POS Cash Balance
* Credit Card Balance

More than 300 engineered features were created to summarize each customer's historical financial behavior.

---

## Project Workflow

```text
Raw Data
    │
    ▼
Exploratory Data Analysis
    │
    ▼
Feature Engineering
    │
    ▼
Dataset Merging
    │
    ▼
Preprocessing Pipeline
    │
    ▼
Model Training
    │
    ▼
Threshold Optimization
    │
    ▼
Model Explainability (SHAP)
    │
    ▼
Streamlit Deployment
```

---

## Feature Engineering

Features were engineered from multiple customer history tables, including:

* Active and closed loan statistics
* Debt-to-credit ratios
* Previous application approval/refusal behavior
* Installment payment delays
* Late payment ratios
* POS cash repayment behavior
* Credit card utilization statistics
* Aggregated financial indicators

These engineered features significantly improved predictive performance over using the application data alone.

---

## Models Evaluated

| Model                      |    ROC-AUC |
| -------------------------- | ---------: |
| Logistic Regression        |     0.7707 |
| LightGBM                   |     0.7799 |
| XGBoost                    |     0.7830 |
| **CatBoost (Final Model)** | **0.7857** |

CatBoost was selected as the final model based on overall validation performance.

---

## Threshold Optimization

Instead of using the default probability threshold of **0.50**, the decision threshold was optimized using the validation set.

**Final Threshold:** **0.15**

This improves recall for identifying high-risk customers, which is often more valuable in credit risk assessment than maximizing overall accuracy.

---

## Model Explainability

Model predictions were interpreted using **SHAP (SHapley Additive Explanations)**.

The project includes:

* Feature Importance
* SHAP Summary Plot
* SHAP Waterfall Plot
* ROC Curve
* Precision–Recall Curve

These visualizations help explain why the model classifies a customer as low or high risk.

---

## Streamlit Application

The project includes an interactive Streamlit dashboard where users can:

* Enter customer information
* Predict default probability
* View predicted risk level
* Explore model insights
* Visualize feature importance and SHAP explanations

---

## Project Structure

```text
home-credit-risk/
│
├── app.py
├── data/
├── models/
├── notebooks/
├── reports/
│   └── figures/
├── src/
│   └── prediction.py
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/home-credit-risk.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the Streamlit application

```bash
streamlit run app.py
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* CatBoost
* XGBoost
* LightGBM
* SHAP
* Matplotlib
* Streamlit
* Joblib

---

## Results

* ROC-AUC: **0.7857**
* Feature engineering from multiple relational datasets
* SHAP-based model interpretability
* Threshold optimization for improved risk detection
* Interactive Streamlit deployment

---

## Future Improvements

* Hyperparameter optimization using Optuna
* Probability calibration
* Cost-sensitive learning
* Model monitoring pipeline
* REST API deployment using FastAPI

---

## Acknowledgements

* Home Credit Group
* Kaggle Home Credit Default Risk Competition

