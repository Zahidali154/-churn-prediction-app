import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ===============================
# LOAD MODEL
# ===============================
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
st.set_page_config(page_title="Churn Intelligence System", layout="wide")

st.title("🏦 Customer Churn Risk Intelligence System")

st.markdown("Predict churn probability and classify customers into actionable risk segments.")

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

# ===============================
# FEATURE ENGINEERING
# ===============================
balance_salary_ratio = balance / salary if salary != 0 else 0
engagement_score = active + products
tenure_age_ratio = tenure / age if age != 0 else 0

# ===============================
# ENCODING (MATCH TRAINING)
# ===============================
geo_germany = 1 if geography == "Germany" else 0
geo_spain = 1 if geography == "Spain" else 0
gender_male = 1 if gender == "Male" else 0

# ===============================
# FINAL INPUT ARRAY (IMPORTANT)
# ===============================
input_data = np.array([[
    credit_score,
    age,
    tenure,
    balance,
    products,
    has_card,
    active,
    salary,
    balance_salary_ratio,
    engagement_score,
    tenure_age_ratio,
    geo_germany,
    geo_spain,
    gender_male
]])
input_data = scaler.transform(input_data)
# ===============================
# PREDICTION
# ===============================
if st.button("🔍 Predict Churn Risk"):
    st.write(input_data)
    prob = model.predict_proba(input_data)[0][1]

    st.markdown("---")

    # ===============================
    # RESULT
    # ===============================
    st.subheader("📊 Churn Risk Result")

    if prob > 0.7:
        st.error("🚨 High Risk Customer – Immediate Action Required")
    elif prob > 0.4:
        st.warning("⚠️ Medium Risk – Monitor Closely")
    else:
        st.success("✅ Low Risk – Stable Customer")

    # ===============================
    # RECOMMENDATION
    # ===============================
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

    # ===============================
    # METRICS
    # ===============================
    if prob > 0.7:
        risk = "🔴 High Risk"
    elif prob > 0.4:
        risk = "🟠 Medium Risk"
    else:
        risk = "🟢 Low Risk"

    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
    	st.metric("Churn Probability", f"{prob:.2f}")
    	st.progress(prob)

    with col2:
    	st.metric("Risk Level", risk)

    # ===============================
    # BUSINESS INSIGHT
    # ===============================
    st.subheader("Business Insight")

    if prob > 0.7:
        st.error("High risk customer. Immediate retention action recommended.")
    elif prob > 0.4:
        st.warning("Moderate risk. Targeted engagement strategies needed.")
    else:
        st.success("Low risk customer. No immediate action required.")
# ===============================
# WHAT-IF ANALYSIS
# ===============================
st.subheader("📊 What-if Scenario")

new_products = st.slider("Adjust Products", 1, 4, products)

new_engagement = active + new_products

whatif_input = np.array([[
    credit_score,
    age,
    tenure,
    balance,
    new_products,
    has_card,
    active,
    salary,
    balance_salary_ratio,
    new_engagement,
    tenure_age_ratio,
    geo_germany,
    geo_spain,
    gender_male
]])
whatif_input = scaler.transform(whatif_input)
whatif_prob = model.predict_proba(whatif_input)[0][1]

st.write(f"New Churn Probability if products = {new_products}: **{whatif_prob:.2f}**")
# ===============================
# FEATURE IMPORTANCE (STATIC TEXT)
# ===============================
st.subheader("📌 Key Churn Drivers")

st.markdown("""
- Age  
- Number of Products  
- Engagement Score  
- Balance  
- Geography (Germany)  

These features strongly influence customer churn behavior.
""")
st.markdown("---")

st.subheader("📊 Risk Interpretation Guide")

st.write("""
🔴 High Risk (>0.7): Immediate retention required  
🟠 Medium Risk (0.4–0.7): Monitor and engage  
🟢 Low Risk (<0.4): Stable customers  
""")
st.markdown("---")

st.subheader("🧠 Business Impact")

st.write("""
The model enables targeted retention strategies by identifying high-risk customers early.

Instead of targeting all customers, banks can focus on a small high-risk segment,
reducing marketing cost and increasing retention efficiency.

This improves:
• Customer Lifetime Value (CLV)
• Revenue stability
• Marketing ROI
""")
st.caption("⚠️ This system is designed to assist decision-making, not replace human judgment.")
