# ============================================================
# AI Loan Approval Predictor - Web Application
# Author: Sanath | York University
# ============================================================
# I built this Streamlit app as the frontend for my loan
# prediction model. It lets users input loan details, get
# instant predictions, and see SHAP explanations for why
# the model made each decision. I chose Streamlit because
# it lets me create a production-quality UI purely in Python.
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import shap
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="AI Loan Approval Predictor", page_icon="🏦", layout="wide", initial_sidebar_state="expanded")

# I cache the model loading so it doesn't reload on every interaction
@st.cache_resource
def load_model():
    model = pickle.load(open("models/xgb_model.pkl", "rb"))
    le_purpose = pickle.load(open("models/le_purpose.pkl", "rb"))
    le_home = pickle.load(open("models/le_home.pkl", "rb"))
    feature_cols = json.load(open("models/feature_columns.json"))
    metrics = json.load(open("models/metrics.json"))
    return model, le_purpose, le_home, feature_cols, metrics

model, le_purpose, le_home, feature_cols, metrics = load_model()

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; background: linear-gradient(90deg, #1e3a5f, #2563eb); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; margin-bottom: 0; }
    .sub-header { text-align: center; color: #6b7280; font-size: 1.1rem; margin-top: -10px; margin-bottom: 30px; }
    .approved-box { background: linear-gradient(135deg, #059669, #10b981); padding: 30px; border-radius: 16px; text-align: center; color: white; }
    .denied-box { background: linear-gradient(135deg, #dc2626, #ef4444); padding: 30px; border-radius: 16px; text-align: center; color: white; }
    .stButton > button { width: 100%; background: linear-gradient(90deg, #1e3a5f, #2563eb); color: white; border: none; padding: 12px; font-size: 1.1rem; font-weight: 600; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🏦 AI Loan Approval Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Powered by XGBoost Machine Learning with Explainable AI (SHAP)</p>', unsafe_allow_html=True)

# Sidebar — I display model performance metrics here so reviewers can see accuracy at a glance
with st.sidebar:
    st.markdown("### 📊 Model Performance")
    st.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    st.metric("Precision", f"{metrics['precision']:.1%}")
    st.metric("Recall", f"{metrics['recall']:.1%}")
    st.metric("F1 Score", f"{metrics['f1']:.1%}")
    st.divider()
    st.markdown("### 🔬 About This Model")
    st.markdown(f"""
    - **Algorithm:** XGBoost (Gradient Boosting)
    - **Explainability:** SHAP values
    - **Training Data:** {metrics['train_size']:,} samples
    - **Test Data:** {metrics['test_size']:,} samples
    - **Features:** {len(feature_cols)} input variables
    """)
    st.divider()
    st.markdown("### 🏆 Top Predictive Features")
    for feat, imp in list(metrics['feature_importance'].items())[:5]:
        clean_name = feat.replace("_encoded", "").replace("_", " ").title()
        st.progress(float(imp) / 0.35, text=f"{clean_name}: {float(imp):.1%}")
    st.divider()
    st.markdown("<div style='text-align:center;color:#9ca3af;font-size:0.85rem;'>Built by Sanath | York University<br>Python • XGBoost • SHAP • Streamlit</div>", unsafe_allow_html=True)

# Main input form
st.markdown("### 📝 Enter Loan Application Details")
col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("👤 Age", 21, 65, 35)
    annual_income = st.number_input("💰 Annual Income ($)", min_value=20000, max_value=200000, value=60000, step=5000)
    credit_score = st.slider("📈 Credit Score", 300, 850, 680)
    employment_years = st.slider("💼 Years at Current Job", 0, 30, 5)

with col2:
    loan_amount = st.number_input("🏷️ Loan Amount ($)", min_value=1000, max_value=50000, value=15000, step=1000)
    dti_ratio = st.slider("📊 Debt-to-Income Ratio", min_value=0.05, max_value=0.80, value=0.30, step=0.05, help="Monthly debt payments ÷ Monthly income. Lower is better.")
    num_credit_lines = st.slider("💳 Number of Credit Accounts", 1, 14, 5)
    previous_default = st.selectbox("⚠️ Previous Loan Default?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

with col3:
    loan_purpose = st.selectbox("🎯 Loan Purpose", options=["debt_consolidation", "home_improvement", "business", "education", "medical", "other"], format_func=lambda x: x.replace("_", " ").title())
    home_ownership = st.selectbox("🏠 Home Ownership", options=["rent", "own", "mortgage"], format_func=lambda x: x.title())
    st.markdown("---")
    st.markdown("**📋 Quick Summary**")
    income_to_loan = annual_income / loan_amount
    st.write(f"Income-to-Loan Ratio: **{income_to_loan:.1f}x**")
    dti_status = "✅ Good" if dti_ratio < 0.36 else "⚠️ High" if dti_ratio < 0.5 else "🔴 Very High"
    st.write(f"DTI Status: **{dti_status}**")
    credit_status = "✅ Excellent" if credit_score >= 750 else "✅ Good" if credit_score >= 670 else "⚠️ Fair" if credit_score >= 580 else "🔴 Poor"
    st.write(f"Credit Rating: **{credit_status}**")

st.markdown("---")

if st.button("🚀 Predict Loan Approval", use_container_width=True):
    # I prepare the input in the same format the model was trained on
    input_data = pd.DataFrame({
        "age": [age], "annual_income": [annual_income], "credit_score": [credit_score],
        "employment_years": [employment_years], "loan_amount": [loan_amount],
        "dti_ratio": [dti_ratio], "num_credit_lines": [num_credit_lines],
        "previous_default": [previous_default],
        "loan_purpose_encoded": [le_purpose.transform([loan_purpose])[0]],
        "home_ownership_encoded": [le_home.transform([home_ownership])[0]]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    confidence = probability.max() * 100

    st.markdown("---")
    result_col1, result_col2 = st.columns([1, 1])

    with result_col1:
        if prediction == 1:
            st.markdown(f'<div class="approved-box"><h1 style="color:white;margin:0;">✅ APPROVED</h1><p style="font-size:1.3rem;margin-top:10px;">Confidence: {confidence:.1f}%</p></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="denied-box"><h1 style="color:white;margin:0;">❌ DENIED</h1><p style="font-size:1.3rem;margin-top:10px;">Confidence: {confidence:.1f}%</p></div>', unsafe_allow_html=True)

    with result_col2:
        # Probability gauge chart I built with Plotly
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=probability[1] * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Approval Probability", "font": {"size": 18}},
            number={"suffix": "%", "font": {"size": 36}},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#2563eb"},
                   "steps": [{"range": [0, 40], "color": "#fee2e2"}, {"range": [40, 60], "color": "#fef3c7"}, {"range": [60, 100], "color": "#d1fae5"}],
                   "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 50}}
        ))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    # SHAP explanation — the key differentiator of this project
    # I use SHAP to show which factors pushed the decision in each direction
    st.markdown("### 🔍 Why Did the Model Make This Decision?")
    st.markdown("*SHAP shows which factors pushed the decision toward approval or denial:*")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)

    feature_names = ["Age", "Annual Income", "Credit Score", "Employment Years", "Loan Amount", "DTI Ratio", "Credit Lines", "Previous Default", "Loan Purpose", "Home Ownership"]

    shap_df = pd.DataFrame({
        "Feature": feature_names, "SHAP Value": shap_values[0],
        "Direction": ["Helps Approval" if v > 0 else "Hurts Approval" for v in shap_values[0]]
    }).sort_values("SHAP Value", key=abs, ascending=True)

    fig_shap = px.bar(shap_df, x="SHAP Value", y="Feature", orientation="h", color="Direction",
                      color_discrete_map={"Helps Approval": "#10b981", "Hurts Approval": "#ef4444"},
                      title="Feature Impact on Prediction")
    fig_shap.update_layout(height=400, xaxis_title="Impact on Approval (SHAP Value)", yaxis_title="",
                           legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
    st.plotly_chart(fig_shap, use_container_width=True)

    # Plain English explanation
    st.markdown("#### 📝 Plain English Explanation")
    top_positive = sorted([(f, v) for f, v in zip(feature_names, shap_values[0]) if v > 0], key=lambda x: x[1], reverse=True)
    top_negative = sorted([(f, v) for f, v in zip(feature_names, shap_values[0]) if v < 0], key=lambda x: x[1])
    if top_positive:
        st.success(f"🟢 Factors helping approval: {', '.join([f'**{f}**' for f, _ in top_positive[:3]])}")
    if top_negative:
        st.error(f"🔴 Factors hurting approval: {', '.join([f'**{f}**' for f, _ in top_negative[:3]])}")

st.markdown("---")
st.markdown("<div style='text-align:center;color:#9ca3af;padding:20px;'><p><strong>AI Loan Approval Predictor</strong> | Built with Python, XGBoost, SHAP & Streamlit</p><p>⚠️ Portfolio project for educational purposes only.</p><p>Built by <strong>Sanath</strong> | York University | 2026</p></div>", unsafe_allow_html=True)
