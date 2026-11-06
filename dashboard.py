import streamlit as st
import pandas as pd
import numpy as np
import requests

st.set_page_config(page_title="LK Telecom Retention AI Portal", layout="wide")

st.title("📡 Sri Lanka Telecom — Customer Churn & Explainable AI Portal")
st.markdown("Real-time subscriber risk scoring powered by **Segmented Gradient Boosting**, **TreeSHAP**, and **FastAPI** on Azure ML.")

# Load sample customers from Gold Test Set
@st.cache_data
def load_test_samples():
    df = pd.read_parquet("gold_test.parquet")
    return df.head(200)

df_samples = load_test_samples()

# Sidebar: Select a Customer from Holdout Set
st.sidebar.header("🔍 Subscriber Lookup")
contract_filter = st.sidebar.selectbox("Filter by Contract Type", ["All", "prepaid", "postpaid"])

if contract_filter != "All":
    filtered_df = df_samples[df_samples["contract_type"] == contract_filter]
else:
    filtered_df = df_samples

selected_cid = st.sidebar.selectbox("Select Customer ID", filtered_df["customer_id"].tolist())
cust_row = filtered_df[filtered_df["customer_id"] == selected_cid].iloc[0].to_dict()

st.sidebar.markdown("---")
st.sidebar.subheader("🛠️ What-If Simulator")
sim_days_recharge = st.sidebar.slider(
    "Days Since Last Recharge (Prepaid)", 0.0, 120.0,
    float(cust_row["days_since_last_recharge"]) if pd.notna(cust_row["days_since_last_recharge"]) else 0.0
)
sim_support = st.sidebar.selectbox(
    "Support Ticket Status",
    ["No_Complaint", "Resolved_Complaint", "Unresolved_Complaint"],
    index=["No_Complaint", "Resolved_Complaint", "Unresolved_Complaint"].index(cust_row["support_status"])
)

# Apply What-If overrides
payload = {k: (None if pd.isna(v) else v) for k, v in cust_row.items() if k not in ["churned", "complaint_text"]}
if payload["contract_type"] == "prepaid":
    payload["days_since_last_recharge"] = sim_days_recharge
    payload["is_prepaid_inactive_30d"] = 1 if sim_days_recharge > 30 else 0
payload["support_status"] = sim_support

# Top Summary Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Customer ID", payload["customer_id"])
col2.metric("Operator & Region", f"{payload['operator']} ({payload['region']})")
col3.metric("Contract & Plan", f"{payload['contract_type'].title()} — {payload['plan_name']}")
col4.metric("Monthly Plan Price", f"Rs. {payload['monthly_price_lkr']}")

st.markdown("---")

# Call FastAPI Endpoint
try:
    res = requests.post("http://localhost:8000/predict", json=payload, timeout=5)
    if res.status_code == 200:
        result = res.json()
        prob = result["churn_probability"] * 100
        thresh = result["optimal_threshold"] * 100
        flagged = result["is_flagged_at_risk"]

        left_col, right_col = st.columns([1, 1.3])

        with left_col:
            st.subheader("🎯 Real-Time Risk Assessment")
            if flagged:
                st.error(f"🚨 **HIGH CHURN RISK DETECTED: {prob:.1f}%** (Threshold: {thresh:.1f}%)")
            else:
                st.success(f"✅ **LOW CHURN RISK: {prob:.1f}%** (Threshold: {thresh:.1f}%)")

            st.markdown("#### 💡 Recommended Retention Playbook")
            st.info(result["recommended_retention_action"])

        with right_col:
            st.subheader("🧠 Top 3 TreeSHAP Root Causes")
            drivers = result["top_root_causes"]
            df_drivers = pd.DataFrame(drivers)
            for d in drivers:
                st.markdown(f"- **`{d['feature']}`** = `{d['value']}` *(SHAP Impact: `+{d['shap_impact']}`)*")
            st.bar_chart(df_drivers.set_index("feature")["shap_impact"])
    else:
        st.error(f"API Error: {res.text}")
except Exception as e:
    st.warning(f"Make sure FastAPI is running on port 8000! Error: {e}")