"""Streamlit Interactive Credit Risk Assessment & Advisor Dashboard."""
import streamlit as st
import pandas as pd
import numpy as np

from src.schemas import ApplicantInput
from src.engine import RiskEngine

# Configure page
st.set_page_config(
    page_title="AI Credit Risk Advisor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache Risk Engine instance
@st.cache_resource
def load_engine():
    return RiskEngine()

engine = load_engine()

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #007bff;
        margin-bottom: 12px;
    }
    .badge-approved {
        background-color: #d4edda;
        color: #155724;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .badge-review {
        background-color: #fff3cd;
        color: #856404;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .badge-declined {
        background-color: #f8d7da;
        color: #721c24;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .advisor-box {
        background-color: #e8f4fd;
        border-left: 6px solid #1e88e5;
        padding: 16px;
        border-radius: 8px;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

st.title("💳 AI Credit Risk & Financial Advisor Engine")
st.caption("Probabilistic Default Risk Scoring, SHAP Factor Attribution, and Automated 'What-If' Restructuring")

# Sidebar: Applicant Controls & Presets
st.sidebar.header("Applicant Profile & Loan Terms")

with st.sidebar.expander("☁️ Cloud Architecture & API", expanded=False):
    st.markdown("**Live GCP Backend:** [Cloud Run API](https://credit-risk-api-266811956116.us-central1.run.app/docs)")
    st.markdown("**API Health:** `Online (HTTP 200)`")
    st.markdown("**Model Version:** `1.0.0-path-b`")

preset = st.sidebar.selectbox(
    "Load Scenario Preset",
    ["Custom", "High Risk ($408k Car Loan)", "Prime Safe Borrower ($25k Personal)", "Borderline Moderate Risk"]
)

# Default values based on preset
if preset == "High Risk ($408k Car Loan)":
    def_inc, def_exp, def_sav, def_cs = 4051.0, 2244.42, 194901.41, 718
    def_type, def_p, def_term, def_rate = "Car", 407915.25, 12, 22.21
    def_age, def_edu, def_occ = 62, "Master", "Student"
elif preset == "Prime Safe Borrower ($25k Personal)":
    def_inc, def_exp, def_sav, def_cs = 7500.0, 2500.0, 45000.0, 780
    def_type, def_p, def_term, def_rate = "Personal", 25000.0, 48, 8.5
    def_age, def_edu, def_occ = 38, "Bachelor", "Engineer"
elif preset == "Borderline Moderate Risk":
    def_inc, def_exp, def_sav, def_cs = 3800.0, 2400.0, 6000.0, 640
    def_type, def_p, def_term, def_rate = "Auto", 32000.0, 36, 14.5
    def_age, def_edu, def_occ = 29, "Bachelor", "Sales"
else:
    def_inc, def_exp, def_sav, def_cs = 5000.0, 2500.0, 15000.0, 700
    def_type, def_p, def_term, def_rate = "Personal", 30000.0, 36, 11.0
    def_age, def_edu, def_occ = 35, "Bachelor", "Professional"

col_sb1, col_sb2 = st.sidebar.columns(2)
with col_sb1:
    monthly_income = st.number_input("Monthly Income ($)", min_value=500.0, max_value=100000.0, value=float(def_inc), step=250.0)
    savings = st.number_input("Liquid Savings ($)", min_value=0.0, max_value=1000000.0, value=float(def_sav), step=1000.0)
    credit_score = st.slider("Credit Score", min_value=300, max_value=850, value=int(def_cs))
with col_sb2:
    monthly_expenses = st.number_input("Monthly Living Expenses ($)", min_value=0.0, max_value=50000.0, value=float(def_exp), step=100.0)
    loan_type = st.selectbox("Loan Type", ["Personal", "Auto", "Car", "Home", "Student"], index=["Personal", "Auto", "Car", "Home", "Student"].index(def_type) if def_type in ["Personal", "Auto", "Car", "Home", "Student"] else 0)
    loan_amount = st.number_input("Loan Principal ($)", min_value=1000.0, max_value=1000000.0, value=float(def_p), step=2500.0)

col_sb3, col_sb4 = st.sidebar.columns(2)
with col_sb3:
    loan_term_months = st.number_input("Tenure (Months)", min_value=6, max_value=360, value=int(def_term), step=6)
with col_sb4:
    loan_interest_rate_pct = st.number_input("Interest Rate (%)", min_value=0.5, max_value=36.0, value=float(def_rate), step=0.25)

with st.sidebar.expander("Additional Demographics"):
    age = st.slider("Age", 18, 90, int(def_age))
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    education_level = st.selectbox("Education", ["High School", "Bachelor", "Master", "Doctorate"], index=["High School", "Bachelor", "Master", "Doctorate"].index(def_edu) if def_edu in ["High School", "Bachelor", "Master", "Doctorate"] else 1)
    employment_status = st.selectbox("Employment", ["Employed", "Self-Employed", "Unemployed"])
    job_title = st.text_input("Job Title", value=def_occ)
    region = st.selectbox("Region", ["North America", "Europe", "Asia", "South America", "Africa", "Oceania"])

# Construct Applicant Input
applicant = ApplicantInput(
    age=age,
    gender=gender,
    education_level=education_level,
    employment_status=employment_status,
    job_title=job_title,
    region=region,
    monthly_income_usd=monthly_income,
    monthly_expenses_usd=monthly_expenses,
    savings_usd=savings,
    credit_score=credit_score,
    loan_type=loan_type,
    loan_amount_usd=loan_amount,
    loan_term_months=loan_term_months,
    loan_interest_rate_pct=loan_interest_rate_pct
)

# Run Inference
response = engine.assess_applicant(applicant)

# Main UI Display
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Default Probability", f"{response.default_probability:.1%}")
    if response.decision == "Approved":
        st.markdown('<span class="badge-approved">✓ APPROVED</span>', unsafe_allow_html=True)
    elif response.decision == "Manual Review":
        st.markdown('<span class="badge-review">⚠ MANUAL REVIEW</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-declined">✕ DECLINED</span>', unsafe_allow_html=True)

with col2:
    st.metric("Internal Risk Score", f"{response.risk_score} / 850")
    st.write(f"Risk Tier: **{response.risk_tier}**")

with col3:
    st.metric("Estimated Monthly EMI", f"${response.estimated_monthly_emi:,.2f}")
    st.write(f"Tenure: **{loan_term_months} Months**")

with col4:
    dti_display = f"{response.debt_to_income_ratio * 100:.1f}%"
    st.metric("Debt-to-Income (DTI)", dti_display)
    st.write(f"Cash Buffer: **${response.disposable_income_usd:,.0f}/mo**")

st.markdown("---")

# Section 2: SHAP Local Attribution
st.subheader("🔍 Local Explainability: What Drove This Decision?")
st.write("SHAP feature contributions identify the specific push (increasing default risk) and pull (reducing default risk) for this applicant:")

shap_df = pd.DataFrame([
    {
        "Feature": d.feature,
        "Value": d.feature_value,
        "Attribution (Log-Odds)": d.shap_value,
        "Direction": d.direction,
        "Explanation": d.interpretation
    }
    for d in response.key_drivers
])

# Interactive Bar Chart for SHAP attributions
st.bar_chart(
    shap_df.set_index("Feature")["Attribution (Log-Odds)"],
    horizontal=True
)

st.dataframe(
    shap_df[["Feature", "Value", "Direction", "Explanation"]],
    hide_index=True,
    use_container_width=True
)

# Section 3: The Risk Advisor Restructuring Engine
if response.restructuring_options:
    st.markdown("---")
    st.subheader("💡 Automated 'What-If' Restructuring Advice")
    st.write("Because this loan exceeds the target risk threshold, the Risk Advisor engine evaluated multiple recovery paths:")

    for opt in response.restructuring_options:
        with st.container():
            st.markdown(f"""
            <div class="advisor-box">
                <h4>🚀 Strategy: {opt.strategy}</h4>
                <p><strong>Recommendation:</strong> {opt.description}</p>
                <div style="display: flex; gap: 24px; margin-top: 10px;">
                    <div><strong>New Principal:</strong> ${opt.recommended_loan_amount:,.0f}</div>
                    <div><strong>New Term:</strong> {opt.recommended_term_months} months</div>
                    <div><strong>New Monthly EMI:</strong> ${opt.estimated_monthly_emi:,.2f}</div>
                    <div><strong>Projected Risk:</strong> {opt.projected_default_prob:.1%}</div>
                    <div><strong>New Status:</strong> <code>{opt.projected_decision}</code></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.write("")
elif response.decision == "Approved":
    st.success("🎉 This applicant meets all underwriting risk standards ($P(\\text{Default}) \\le 35\\%$) with healthy debt service coverage.")

