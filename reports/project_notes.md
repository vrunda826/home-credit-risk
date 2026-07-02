dataset size:307511
columns:122
target: 0-92%  1-8%
Observation:
Many housing/property-related features have 50–70% missing values.
Missingness strategy will be required during preprocessing.
accuracy can not be a good metric as dataset id imbalanced.
numerical-106,categorical 16
days_employed specific cleaning required as A huge number of rows have 365243
Logistic Regression consistently outperformed Random Forest on the Home Credit dataset, achieving a mean 5-fold ROC-AUC of 0.744 compared to 0.724 for Random Forest.
Logistic Regression performed best among baseline models.
Decision Trees overfit heavily.
Random Forest improved upon a single tree but remained weaker than Logistic Regression.
Class imbalance significantly affected Recall and Precision
| Model               | CV ROC-AUC | Test ROC-AUC |
| ------------------- | ---------- | ------------ |
| Logistic Regression | 0.744      | 0.750        |
| Decision Tree       | 0.706      | 0.538        |
| Random Forest       | 0.724      | 0.733        |
| LightGBM            | 0.754      | 0.7590       |