# Project Notes

## Dataset Overview

- Dataset: Home Credit Default Risk
- Training Samples: **307,511**
- Features: **122**
- Target Variable: `TARGET`
- Class Distribution:
  - Non-Default (0): **92%**
  - Default (1): **8%**

### Feature Types

- Numerical Features: **106**
- Categorical Features: **16**

---

# Initial Observations

- The dataset is highly imbalanced; therefore, **accuracy is not an appropriate evaluation metric**.
- ROC-AUC was selected as the primary metric throughout the project.
- Many housing/property-related variables contain **50–70% missing values**.
- Several external tables contain one-to-many relationships requiring aggregation before merging.
- `DAYS_EMPLOYED` contains the placeholder value **365243**, which represents missing information and must be handled during preprocessing.

---

# Data Preprocessing

Performed the following preprocessing steps:

- Removed duplicate records
- Replaced placeholder values
- Handled missing values
- Built preprocessing pipelines
- One-Hot Encoded categorical variables
- Standardized numerical features where required
- Applied identical preprocessing during training and inference using Scikit-learn Pipelines

---

# Feature Engineering

## Application Dataset

Created several domain-inspired features including:

- Credit-to-Income Ratio
- Annuity-to-Income Ratio
- Credit-to-Goods Ratio
- Income per Family Member
- Employment Age Ratio

---

## Bureau Dataset

Aggregated bureau information into customer-level features.

Created features including:

- Number of active loans
- Number of closed loans
- Total credit
- Total debt
- Average debt
- Maximum debt
- Bureau credit mean
- Active loan ratio
- Closed loan ratio
- Debt-to-credit ratio

These bureau-derived features contributed significantly to model performance.

---

## Previous Applications

Created customer-level aggregation features including:

- Previous application count
- Average previous credit
- Average previous annuity
- Approval ratio
- Refusal ratio

These features appeared among important predictors in the final model.

---

## Installment Payments

Engineered features:

- Average days late
- Maximum days late
- Delay volatility
- Payment ratio statistics
- Payment difference statistics
- 30/60/90-day late ratios
- Total installment amount
- Total payment amount
- Late payment ratio

Observation:

- Overall ROC-AUC improvement was minimal.
- Some installment features became moderately important in the final CatBoost model.
- Further temporal feature engineering may improve performance.

---

## POS Cash Balance

Created aggregated features including:

- Average remaining installments
- Maximum remaining installments
- Oldest record month
- Latest record month
- Average DPD
- Maximum DPD

These features contributed meaningful predictive information.

---

## Credit Card Balance

Aggregated customer credit card history into:

- Average balance
- Maximum balance
- Average credit limit
- Credit utilization ratio
- Payment ratio
- Drawings statistics

Contribution was limited but retained in the final dataset.

---

# Models Evaluated

## Baseline Models

| Model | Test ROC-AUC |
|---------|-------------:|
| Decision Tree | 0.538 |
| Random Forest | 0.733 |
| Logistic Regression | 0.750 |

### Observations

- Decision Tree severely overfitted.
- Logistic Regression surprisingly outperformed Random Forest.
- Linear models established a strong baseline.

---

## Gradient Boosting Models

| Model | Test ROC-AUC |
|---------|-------------:|
| LightGBM | 0.7799 |
| XGBoost | 0.7830 |
| CatBoost | **0.7857** |

CatBoost achieved the highest overall performance.

---

# Cross Validation

Performed **5-Fold Stratified Cross Validation**.

Final CatBoost Fold Scores:

- Fold 1: 0.7830
- Fold 2: 0.7691
- Fold 3: 0.7836
- Fold 4: 0.7756
- Fold 5: 0.7859

Mean ROC-AUC ≈ **0.779**

---

# Feature Selection

Compared:

- Full feature set
- CatBoost Feature Importance
- SHAP Feature Importance

Removing low-importance features slightly improved performance.

Final model retained approximately the most informative 30% of engineered features.

---

# SHAP Analysis

Generated:

- SHAP Summary Plot
- SHAP Feature Importance

Most influential features included:

- EXT_SOURCE_2
- EXT_SOURCE_3
- EXT_SOURCE_1
- DAYS_BIRTH
- POS balance features
- Bureau debt features
- Installment payment statistics
- Previous application refusal ratio

SHAP confirmed that engineered features from external datasets provided meaningful predictive value.

---

# Threshold Optimization

Optimized the decision threshold using the validation set.

Default threshold:

```
0.50
```

Best threshold:

```
0.15
```

Reason:

- Improved recall for default customers.
- Better balance between Precision and Recall.
- More suitable for real-world credit risk applications.

---

# Final Model Performance

Model:

```
CatBoost Classifier
```

Evaluation Metric:

```
ROC-AUC
```

Test ROC-AUC:

```
0.7857
```

Optimized Threshold:

```
0.15
```

---

# Explainability

Implemented:

- Feature Importance
- SHAP Explainability
- ROC Curve
- Precision-Recall Curve

These visualizations were integrated into the deployed Streamlit application.

---

# Deployment

Developed an interactive Streamlit application featuring:

- Customer information form
- Default probability prediction
- Risk level estimation
- Confidence score
- Feature importance visualization
- SHAP summary visualization
- ROC Curve
- Precision-Recall Curve
- Project overview
- Model documentation

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- CatBoost
- LightGBM
- XGBoost
- SHAP
- Matplotlib
- Streamlit
- Joblib

---

# Key Learnings

This project provided hands-on experience with:

- End-to-end Machine Learning workflow
- Feature Engineering
- Data preprocessing pipelines
- Cross Validation
- Hyperparameter Tuning
- Ensemble Tree Models
- Model Explainability (SHAP)
- Threshold Optimization
- Streamlit Deployment
- GitHub Project Organization
- Production-ready ML pipelines

---

# Future Improvements

Potential future enhancements include:

- Time-series feature engineering
- Automated hyperparameter optimization using Optuna
- Probability calibration
- Model monitoring
- FastAPI deployment
- Docker containerization
- CI/CD pipeline