import streamlit as st
import pandas as pd
from src.prediction import predict,build_customer

st.set_page_config(
    page_title="Home Credit Default Risk Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>

section[data-testid="stSidebar"]{
    background-color:#F7F9FC;
}

section[data-testid="stSidebar"] h1{
    font-size:34px;
}

section[data-testid="stSidebar"] p{
    font-size:18px;
}

div[data-testid="stSidebarNav"]{
    font-size:18px;
}

</style>
""",unsafe_allow_html=True)

st.sidebar.markdown("# 🏦 Home Credit")

st.sidebar.caption("Loan Default Risk Prediction")

st.sidebar.markdown("---")


page = st.sidebar.radio(
    " 📑 Navigation",
    [
        "🏠 Home",
        "🔍 Predict Customer",
        "📊 Model Insights",
        "👨‍💻 About"
    ]
)
st.sidebar.markdown("---")

st.sidebar.link_button(
    "🌐 GitHub Repository",
    "https://github.com/vrunda826/home-credit-risk"
)

if page == "🏠 Home":

    st.title("🏦 Home Credit Default Risk Prediction")
    st.sidebar.markdown("---")

    st.sidebar.success(
    """
    **Machine Learning Loan Default Prediction**

    CatBoost + SHAP Explainability
    """
    )

    
    st.markdown("""
### 📌 Project Overview

This project predicts whether a customer is likely to default on a loan using historical financial and credit information from the Home Credit dataset.

The model combines information from multiple data sources including:

- Bureau records
- Previous applications
- Installment payments
- POS Cash Balance
- Credit Card Balance

The final model is a **CatBoost Classifier** trained on engineered features.
""")

    col1, col2, col3 = st.columns(3)

    col1.metric("ROC-AUC", "0.7857")
    col2.metric("Best Threshold", "0.15")
    col3.metric("Final Model", "CatBoost")

    st.success("Navigate using the sidebar to predict customer default risk.")
elif page == "🔍 Predict Customer":

    st.title("🔍 Customer Risk Prediction")
    st.markdown(
        "Fill in the customer details below and click **Predict** to estimate the probability of loan default."
    )

    with st.container(border=True):

        st.subheader("📝 Loan Information")

        col1, col2 = st.columns(2)

        with col1:

            income = st.number_input(
                "Annual Income (₹)",
                min_value=0.0,
                value=180000.0,
                step=10000.0,
                help="Customer's annual income."
            )

            credit = st.number_input(
                "Credit Amount (₹)",
                min_value=0.0,
                value=500000.0,
                step=50000.0
            )

            annuity = st.number_input(
                "Loan Annuity (₹)",
                min_value=0.0,
                value=25000.0,
                step=1000.0
            )

            goods = st.number_input(
                "Goods Price (₹)",
                min_value=0.0,
                value=450000.0,
                step=50000.0
            )

            age = st.slider(
                "Age (Years)",
                min_value=18,
                max_value=70,
                value=35
            )

        with col2:

            employment = st.number_input(
                "Employment Duration (Days)",
                value=-2000,
                step=365,
                help="Negative values indicate days before today."
            )

            ext1 = st.slider(
    "Credit Score 1",
    0.0, 1.0, 0.50, 0.01,
    help="Internal creditworthiness score."
)

            ext2 = st.slider(
                "Credit Score 2",
                0.0, 1.0, 0.60, 0.01,
                help="Internal creditworthiness score."
            )

            ext3 = st.slider(
                "Credit Score 3",
                0.0, 1.0, 0.70, 0.01,
                help="Internal creditworthiness score."
            )

        st.write("")

        predict_btn, reset_btn = st.columns(2)

        with predict_btn:
            predic = st.button(
                "🔮 Predict Default Risk",
                use_container_width=True
            )

        with reset_btn:
            if st.button(
                "🔄 Reset Form",
                use_container_width=True
            ):
                st.rerun()

        if predic:
            user_inputs = {

                "AMT_INCOME_TOTAL": income,
                "AMT_CREDIT": credit,
                "AMT_ANNUITY": annuity,
                "AMT_GOODS_PRICE": goods,
                "DAYS_BIRTH": -age * 365,
                "DAYS_EMPLOYED": employment,
                "EXT_SOURCE_1": ext1,
                "EXT_SOURCE_2": ext2,
                "EXT_SOURCE_3": ext3

            }

            customer = build_customer(user_inputs)

            result = predict(customer)

            st.success("Prediction Completed Successfully!")

            st.divider()

            st.subheader("📊 Prediction Result")

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.metric(
                    "Prediction",
                    result["Prediction"]
                )

            with c2:
                st.metric(
                    "Default Probability",
                    f"{result['Default Probability']}%"
                )

            with c3:
                st.metric(
                    "Risk Level",
                    result["Risk Level"]
                )

            with c4:
                st.metric(
                    "Confidence",
                    f"{result['Confidence']}%"
                )

            st.write("### 📈 Default Probability")

            st.progress(
                result["Default Probability"] / 100
            )
            if result["Default Probability"]<15:

                st.success("🟢 Low Risk")

            elif result["Default Probability"]<40:

                st.warning("🟡 Medium Risk")

            else:

                st.error("🔴 High Risk")
            st.caption(
                f"{result['Default Probability']}% probability of loan default."
            )

            if result["Prediction"] == "Default":

                st.error(
                    "⚠️ High Risk Customer. Loan approval should be reviewed carefully."
                )

            else:

                st.success(
                    "✅ Customer is likely to repay the loan."
                )

        
elif page == "📊 Model Insights":

    st.title("📊 Model Insights")
    st.subheader("Top Important Features")

    st.write("""
    The model relies primarily on external credit scores,
    loan amount, bureau history,
    installment payment behaviour,
    and previous loan applications.
    """)

    st.subheader("Feature Importance")

    st.image(
        "reports/figures/feature_importance.png",
        use_container_width=True
    )

    st.subheader("SHAP Summary")

    st.image(
        "reports/figures/shap_summary.png",
        use_container_width=True
    )

    st.subheader("ROC Curve")

    st.image(
        "reports/figures/roc_curve.png",
        use_container_width=True
    )
    st.info(
"""
SHAP explains why the model predicts a customer as risky or safe.

Positive SHAP → Higher default risk

Negative SHAP → Lower default risk
"""
)
    st.subheader("Precision Recall Curve")

    st.image(
        "reports/figures/pr_curve.png",
        use_container_width=True
    )
elif page == "👨‍💻 About":

    st.title("👨‍💻 About This Project")

    st.markdown("""

### Objective

Predict whether a customer will default on a loan using historical financial information.

---

### Dataset

Home Credit Default Risk (Kaggle)

---

### Models Compared

- Logistic Regression
- LightGBM
- XGBoost
- CatBoost

---

### Final Model

CatBoost Classifier

ROC-AUC: **0.7857**

---

### Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- CatBoost
- SHAP
- Streamlit

---
""")