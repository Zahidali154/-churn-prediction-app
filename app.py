import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ===============================
# LOAD MODEL
# ===============================
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Churn Intelligence System", layout="wide")

# ===============================
# HEADER
# ===============================
st.markdown("<h1 style='text-align: center;'>🏦 Churn Intelligence Dashboard</h1>", unsafe_allow_html=True)
st.markdown("### Predict churn probability and classify customers into risk segments")

# ===============================
# SIDEBAR INPUT
# ===============================
st.sidebar.header("Customer Input Features")

credit_score = st.sidebar.slider("Credit Score", 300, 900, 600)
age = st.sidebar.slider("Age", 18, 80, 40)
tenure = st.sidebar.slider("Tenure (Years)", 0, 10, 3)
balance = st.sidebar.number_input("Balance", 0.0, 250000.0, 50000.0)
products = st.sidebar.selectbox("Number of Products", [1,2,3,4])
has_card = st.sidebar.selectbox("Has Credit Card", [0,1])
active = st.sidebar.selectbox("Is Active Member", [0,1])
salary = st.sidebar.number_input("Estimated Salary", 10000.0, 200000.0, 50000.0)

geography = st.sidebar.selectbox("Geography", ["France","Germany","Spain"])
gender = st.sidebar.selectbox("Gender", ["Male","Female"])

st.sidebar.markdown("---")
st.sidebar.info("Adjust inputs to simulate customer behavior and predict churn risk.")

# ===============================
# FEATURE ENGINEERING
# ===============================
balance_salary_ratio = balance / salary if salary != 0 else 0
engagement_score = active + products
tenure_age_ratio = tenure / age if age != 0 else 0

geo_germany = 1 if geography == "Germany" else 0
geo_spain = 1 if geography == "Spain" else 0
gender_male = 1 if gender == "Male" else 0

input_data = np.array([[ 
    credit_score, age, tenure, balance, products,
    has_card, active, salary,
    balance_salary_ratio, engagement_score, tenure_age_ratio,
    geo_germany, geo_spain, gender_male
]])

input_data = scaler.transform(input_data)

# ===============================
# TABS
# ===============================
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📈 Analysis", "ℹ️ About"])

# ===============================
# DASHBOARD
# ===============================
with tab1:

    st.header("📊 Customer Risk Dashboard")

    if st.button("🔍 Predict Churn Risk"):

        prob = model.predict_proba(input_data)[0][1]

        st.success("✅ Model successfully evaluated customer risk")
        st.markdown("---")

        # RISK LEVEL
        if prob > 0.7:
            risk = "🔴 High Risk"
            st.error("🚨 High Risk Customer – Immediate Action Required")
        elif prob > 0.4:
            risk = "🟠 Medium Risk"
            st.warning("⚠️ Medium Risk – Monitor Closely")
        else:
            risk = "🟢 Low Risk"
            st.success("✅ Low Risk – Stable Customer")

        # METRICS
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Churn Probability", f"{prob:.2f}")

            if prob > 0.7:
                st.progress(prob, text="High Risk Zone")
            elif prob > 0.4:
                st.progress(prob, text="Medium Risk Zone")
            else:
                st.progress(prob, text="Low Risk Zone")

        with col2:
            st.metric("Risk Level", risk)

        # RECOMMENDATION
        st.subheader("📌 Recommended Action")

        if prob > 0.7:
            st.write("• Offer retention bonus")
            st.write("• Assign relationship manager")
            st.write("• Reduce service charges")
        elif prob > 0.4:
            st.write("• Send personalized offers")
            st.write("• Increase engagement campaigns")
        else:
            st.write("• Maintain current strategy")

                # SMART INSIGHT
        st.subheader("🧠 Smart Insight")

        if prob > 0.7:
            if active == 0 and products == 1:
                st.error("Very low engagement and product usage driving high churn risk.")
            elif age > 50:
                st.warning("Older inactive customers show higher churn tendency.")
            else:
                st.warning("Multiple risk factors contributing to churn.")

        elif prob > 0.4:
            if products == 1:
                st.info("Increasing product usage may help reduce churn risk.")
            elif active == 0:
                st.info("Improving customer engagement can lower churn risk.")
            else:
                st.info("Moderate risk detected. Monitor customer behavior.")

        else:
            if products == 1:
                st.success("Customer is stable, but increasing product usage can further reduce risk.")
            else:
                st.success("Strong engagement and product usage indicate a loyal customer.")

    # WHAT-IF
    st.subheader("📊 What-if Scenario")

    new_products = st.slider("Adjust Products", 1, 4, products)
    new_engagement = active + new_products

    whatif_input = np.array([[ 
        credit_score, age, tenure, balance, new_products,
        has_card, active, salary,
        balance_salary_ratio, new_engagement, tenure_age_ratio,
        geo_germany, geo_spain, gender_male
    ]])

    whatif_input = scaler.transform(whatif_input)
    whatif_prob = model.predict_proba(whatif_input)[0][1]

    st.write(f"New Churn Probability: **{whatif_prob:.2f}**")

# ===============================
# ANALYSIS
# ===============================
with tab2:

    st.header("📈 Model Analysis")

    importance = model.feature_importances_
    features = [
        "CreditScore","Age","Tenure","Balance","Products",
        "HasCard","Active","Salary","BalanceSalaryRatio",
        "EngagementScore","TenureAgeRatio","GeoGermany",
        "GeoSpain","GenderMale"
    ]

    imp_df = pd.Series(importance, index=features).sort_values()

    fig, ax = plt.subplots()
    imp_df.plot(kind='barh', ax=ax)
    st.pyplot(fig)

    st.info("Customers with low engagement & fewer products show higher churn risk.")

# ===============================
# ABOUT
# ===============================
with tab3:

    st.header("ℹ️ About This Project")

    st.write("""
    This project predicts customer churn using machine learning.

    Key Highlights:
    • Feature engineering (Engagement Score, Ratios)  
    • Risk segmentation (High / Medium / Low)  
    • What-if simulation  
    • Streamlit deployment  

    Business Value:
    Helps banks reduce churn by targeting high-risk customers early.
    """)

# ===============================
# FOOTER
# ===============================
st.markdown("---")

st.subheader("📊 Risk Interpretation Guide")

st.write("""
🔴 High Risk (>0.7): Immediate retention required  
🟠 Medium Risk (0.4–0.7): Monitor and engage  
🟢 Low Risk (<0.4): Stable customers  
""")

st.subheader("🧠 Business Impact")

st.write("""
Improves Customer Lifetime Value, Revenue Stability, and Marketing ROI.
""")

st.caption("⚠️ This system is designed to assist decision-making, not replace human judgment.")
