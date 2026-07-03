import joblib
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "catboost_pipeline.pkl"
THRESHOLD_PATH = BASE_DIR / "models" / "best_threshold.pkl"
template = pd.read_csv(
    BASE_DIR /"template_customer.csv"
)
pipeline = joblib.load(MODEL_PATH)
threshold = joblib.load(THRESHOLD_PATH)

def build_customer(user_inputs):
    """
    Create a complete customer record by starting from
    the template and replacing only the fields entered
    by the user.
    """

    customer = template.copy()

    for feature, value in user_inputs.items():
        customer.loc[0, feature] = value

    return customer


def predict_customer(customer_df):
    """
    Predict default probability for one or more customers.

    Parameters
    ----------
    customer_df : pandas.DataFrame

    Returns
    -------
    probability
    prediction
    """

    probability = pipeline.predict_proba(customer_df)[:,1]

    prediction = (
        probability >= threshold
    ).astype(int)

    return probability, prediction

def get_risk_level(probability):

    if probability < 0.15:
        return "Low Risk 🟢"

    elif probability < 0.40:
        return "Medium Risk 🟡"

    else:
        return "High Risk 🔴"

def predict(customer_df):

    probability, prediction = predict_customer(customer_df)

    probability = probability[0]
    prediction = prediction[0]

    result = {

        "Default Probability": round(probability * 100, 2),

        "Prediction":
            "Default" if prediction == 1 else "Non Default",

        "Risk Level":
            get_risk_level(probability),

        "Confidence":
            round(
                max(probability, 1 - probability) * 100,
                2
            )
    }

    return result