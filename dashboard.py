import streamlit as st
import pandas as pd
import numpy as np
import requests
import os

st.set_page_config(page_title="LK Telecom Churn MLOps Portal", layout="wide")

st.title("📡 Sri Lanka Telecom — End-to-End Churn MLOps & Explainable AI")
st.markdown(
    "Complete workflow from **Feature Engineering & Data Inspection** $\\rightarrow$ **6-Model Benchmarking** "
    "$\\rightarrow$ **Champion Model Selection (`6_HistGBM_Segmented_Ensemble`)** $\\rightarrow$ **Real-Time TreeSHAP Inference**."
)

tab1, tab2, tab3 = st.tabs([
    "🎯 1. Live Subscriber Risk & SHAP Simulator",
    "🏆 2. Multi-Model Evaluation & Selection",
    "🔬 3. Feature Engineering & Data Inspection"
])

# ==============================================================================
# TAB 1: LIVE SUBSCRIBER PREDICTION & TREESHAP SIMULATOR
# ==============================================================================
with tab1:
    @st.cache_data
    def load_test_samples():
        df = pd.read_parquet("gold_test.parquet")
        return df.head(200)

    df_samples = load_test_samples()

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

    payload = {k: (None if pd.isna(v) else v) for k, v in cust_row.items() if k not in ["churned", "complaint_text"]}
    if payload["contract_type"] == "prepaid":
        payload["days_since_last_recharge"] = sim_days_recharge
        payload["is_prepaid_inactive_30d"] = 1 if sim_days_recharge > 30 else 0
    payload["support_status"] = sim_support

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Customer ID", payload["customer_id"])
    col2.metric("Operator & Region", f"{payload['operator']} ({payload['region']})")
    col3.metric("Contract & Plan", f"{payload['contract_type'].title()} — {payload['plan_name']}")
    col4.metric("Monthly Plan Price", f"Rs. {payload['monthly_price_lkr']}")

    st.markdown("---")

    try:
        res = requests.post("http://localhost:8000/predict", json=payload, timeout=5)
        if res.status_code == 200:
            result = res.json()
            prob = result["churn_probability"] * 100
            thresh = result["optimal_threshold"] * 100
            flagged = result["is_flagged_at_risk"]

            left_col, right_col = st.columns([1, 1.3])

            with left_col:
                st.subheader("🎯 Champion Model Risk Assessment")
                if flagged:
                    st.error(f"🚨 **HIGH CHURN RISK DETECTED: {prob:.1f}%** (Threshold: {thresh:.1f}%)")
                else:
                    st.success(f"✅ **LOW CHURN RISK: {prob:.1f}%** (Threshold: {thresh:.1f}%)")

                st.markdown("#### 💡 Automated Retention Playbook")
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
        st.warning(f"Make sure FastAPI (`uvicorn app:app --port 8000`) is running! Error: {e}")

# ==============================================================================
# TAB 2: MULTI-MODEL BENCHMARKING & CHAMPION SELECTION
# ==============================================================================
with tab2:
    st.subheader("🏆 Multi-Model Benchmark Leaderboard (6 Candidate Algorithms)")
    if os.path.exists("model_comparison_results.csv"):
        bench_df = pd.read_csv("model_comparison_results.csv")
        best_row = bench_df.iloc[0]

        st.success(
            f"**Selected Champion Model:** `{best_row['model_name']}` | "
            f"**Validation PR-AUC:** `{best_row['pr_auc']}` | **ROC-AUC:** `{best_row['roc_auc']}` | "
            f"**Top-10% Decile Lift:** `{best_row['top_10pct_lift']}x`"
        )

        st.dataframe(bench_df, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### PR-AUC Comparison Across Models")
            st.bar_chart(bench_df.set_index("model_name")["pr_auc"])
        with c2:
            st.markdown("#### Top-10% Decile Lift (Business ROI)")
            st.bar_chart(bench_df.set_index("model_name")["top_10pct_lift"])
    else:
        st.info("Run `python step2_train.py` to generate `model_comparison_results.csv`.")

# ==============================================================================
# TAB 3: FEATURE ENGINEERING & DATA INSPECTION
# ==============================================================================
with tab3:
    st.subheader("🔬 Engineered Features & Churn Signal Inspection")
    st.markdown("""
    **Key Domain Features Engineered in `step1_data_pipeline.py`:**
    * **`recharge_velocity_ratio`**: Ratio of latest recharge amount to 3-month rolling average (detects sudden prepaid drop-off).
    * **`is_prepaid_inactive_30d`**: Binary flag for prepaid subscribers with $>30$ days since last reload.
    * **`data_utilization_ratio` & `is_data_exhausted`**: Captures subscribers hitting 100% data cap or underutilizing overpriced plans.
    * **`support_status`**: Consolidates complaint flags and resolution status into `No_Complaint`, `Resolved_Complaint`, and `Unresolved_Complaint`.
    """)

    if os.path.exists("feature_inspection_summary.csv"):
        feat_df = pd.read_csv("feature_inspection_summary.csv")
        col_a, col_b = st.columns([1.2, 1])
        with col_a:
            st.markdown("#### Numerical Feature Correlation & Missingness Audit")
            st.dataframe(feat_df, use_container_width=True)
        with col_b:
            st.markdown("#### Top 10 Features by Absolute Correlation with Churn")
            st.bar_chart(feat_df.head(10).set_index("feature")["abs_corr_with_churn"])
    else:
        st.info("Run `python step1_inspect_features.py` to generate `feature_inspection_summary.csv`.")

    if os.path.exists("shap_global_prepaid.png"):
        st.markdown("---")
        st.markdown("#### Global TreeSHAP Feature Attribution (Beeswarm Plot)")
        st.image("shap_global_prepaid.png", caption="Global TreeSHAP impact on Prepaid Churn")