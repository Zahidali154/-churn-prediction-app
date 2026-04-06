import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="AI Customer Retention System", layout="wide")

# ===============================
# LOGIN SYSTEM
# ===============================
def login():
    st.title("🔐 Login")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        if user == "admin" and pwd == "1234":
            st.session_state["login"] = True
        else:
            st.error("Invalid credentials")

if "login" not in st.session_state:
    st.session_state["login"] = False

if not st.session_state["login"]:
    login()
    st.stop()

# ===============================
# UI STYLE
# ===============================
st.markdown("""
<style>
.main {background-color: #0E1117;}
h1, h2, h3 {color: #00ADB5;}
.stButton>button {
    background-color: #00ADB5;
    color: white;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# LOAD MODEL
# ===============================
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

# ===============================
# HEADER
# ===============================
st.title("🏦 AI-Powered Customer Retention System")
st.markdown("### Predict churn, analyze risk, and take action")

# ===============================
# SIDEBAR INPUT
# ===============================
st.sidebar.header("Customer Input")

credit_score = st.sidebar.slider("Credit Score", 300, 900, 600)
age = st.sidebar.slider("Age", 18, 80, 40)
tenure = st.sidebar.slider("Tenure", 0, 10, 3)
balance = st.sidebar.number_input("Balance", 0.0, 250000.0, 50000.0)
products = st.sidebar.selectbox("Products", [1,2,3,4])
has_card = st.sidebar.selectbox("Credit Card", [0,1])
active = st.sidebar.selectbox("Active Member", [0,1])
salary = st.sidebar.number_input("Salary", 10000.0, 200000.0, 50000.0)
geography = st.sidebar.selectbox("Geography", ["France","Germany","Spain"])
gender = st.sidebar.selectbox("Gender", ["Male","Female"])

st.sidebar.markdown("---")
st.sidebar.info("Adjust inputs to simulate customer behavior")

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
# PDF FUNCTION
# ===============================
def create_pdf(prob, risk):
    doc = SimpleDocTemplate("report.pdf")
    styles = getSampleStyleSheet()

    content = []
    content.append(Paragraph("Customer Churn Report", styles['Title']))
    content.append(Paragraph(f"Churn Probability: {prob:.2f}", styles['Normal']))
    content.append(Paragraph(f"Risk Level: {risk}", styles['Normal']))

    doc.build(content)

# ===============================
# SESSION STATE FIX
# ===============================
if "prob" not in st.session_state:
    st.session_state["prob"] = None

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
        st.session_state["prob"] = model.predict_proba(input_data)[0][1]

    prob = st.session_state["prob"]

    if prob is not None:

        # Risk Logic
        if prob > 0.7:
            risk = "🔴 High"
            st.error("🚨 High Risk Customer – Immediate Action Required")
        elif prob > 0.4:
            risk = "🟠 Medium"
            st.warning("⚠️ Medium Risk – Monitor Closely")
        else:
            risk = "🟢 Low"
            st.success("✅ Low Risk – Stable Customer")

        # Metrics
        col1, col2, col3 = st.columns(3)

        col1.metric("Churn Probability", f"{prob:.2f}")
        col2.metric("Risk Level", risk)
        col3.metric("Engagement Score", engagement_score)

        # Chart
        st.subheader("📊 Risk Visualization")

        fig, ax = plt.subplots()
        ax.bar(["Churn Risk"], [prob])
        ax.set_ylim(0,1)
        st.pyplot(fig)

        # ===============================
        # SMART INSIGHT (FIXED)
        # ===============================
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
                st.info("Moderate risk detected. Monitor behavior.")

        else:
            if products == 1:
                st.success("Customer is stable, but increasing product usage can further reduce risk.")
            else:
                st.success("Strong engagement and product usage indicate a loyal customer.")

        # Recommendation
        st.subheader("📌 Recommended Action")

        if prob > 0.7:
            st.write("• Offer retention bonus")
            st.write("• Assign relationship manager")
        elif prob > 0.4:
            st.write("• Increase engagement campaigns")
        else:
            st.write("• Maintain strategy")

        # PDF
        if st.button("📄 Generate Report"):
            create_pdf(prob, risk)
            with open("report.pdf", "rb") as f:
                st.download_button("Download Report", f, "report.pdf")

# ===============================
# ANALYSIS
# ===============================
with tab2:

    st.header("📈 Model Analysis")

    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    else:
        importance = np.abs(model.coef_[0])

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

# ===============================
# ABOUT
# ===============================
with tab3:

    st.header("ℹ️ About This Project")

    st.write("""
    This project predicts customer churn using machine learning.

    Key Features:
    • Feature engineering  
    • Risk segmentation  
    • What-if analysis  
    • Dashboard visualization  

    Business Value:
    Helps banks reduce churn by targeting high-risk customers.
    """)

# ===============================
# FOOTER
# ===============================
st.markdown("---")

st.subheader("📊 Risk Interpretation Guide")

st.write("""
🔴 High Risk (>0.7): Immediate action required  
🟠 Medium Risk (0.4–0.7): Monitor closely  
🟢 Low Risk (<0.4): Stable customers  
""")

st.caption("⚠️ AI system for decision support only")
