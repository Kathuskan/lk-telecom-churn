import streamlit as st
import pandas as pd
import numpy as np
import requests
import os

st.set_page_config(page_title="LK Telecom Churn MLOps Portal", layout="wide")

st.title("📡 Sri Lanka Telecom — End-to-End Churn MLOps & Explainable AI")
st.markdown(
    "Complete workflow from **Medallion Feature Engineering** $\\rightarrow$ **9-Model Benchmarking** "
    "$\\rightarrow$ **Champion Selection (`8_XGBoost_Segmented_Ensemble`)** $\\rightarrow$ **Real-Time TreeSHAP Inference**."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 1. Live Subscriber Risk & SHAP Simulator",
    "🏆 2. 9-Model Benchmark & Threshold ROI",
    "🔬 3. Feature Engineering & Global TreeSHAP",
    "🗺️ 4. Sri Lanka Regional & Executive Analytics"
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
            champ_model = result.get("champion_model", "8_XGBoost_Segmented_Ensemble")

            left_col, right_col = st.columns([1, 1.3])

            with left_col:
                st.subheader(f"🎯 Risk Assessment ({champ_model})")
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
# TAB 2: 9-MODEL BENCHMARKING & THRESHOLD ROI
# ==============================================================================
with tab2:
    st.subheader("🏆 9-Model Benchmark Leaderboard")
    if os.path.exists("model_comparison_results.csv"):
        bench_df = pd.read_csv("model_comparison_results.csv")
        best_row = bench_df.iloc[0]

        st.success(
            f"**Selected Champion Model:** `{best_row['model_name']}` | "
            f"**Validation PR-AUC:** `{best_row['pr_auc']}` | **ROC-AUC:** `{best_row['roc_auc']}` | "
            f"**Top-10% Decile Lift:** `{best_row['top_10pct_lift']}x`"
        )
        st.dataframe(bench_df, use_container_width=True)

    if os.path.exists("assets/fig1_model_benchmark.png"):
        st.image("assets/fig1_model_benchmark.png", caption="9-Model Validation PR-AUC and Top-10% Decile Lift Comparison")

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        if os.path.exists("assets/fig2_threshold_optimization.png"):
            st.image("assets/fig2_threshold_optimization.png", caption="Precision-Recall Threshold Optimization Curve")
    with col_t2:
        if os.path.exists("assets/fig3_decile_lift_roi.png"):
            st.image("assets/fig3_decile_lift_roi.png", caption="Decile Lift & Cumulative Gain ROI Curve")

# ==============================================================================
# TAB 3: FEATURE ENGINEERING & GLOBAL TREESHAP
# ==============================================================================
with tab3:
    st.subheader("🔬 Global TreeSHAP Attribution (Prepaid vs. Postpaid Ensembles)")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if os.path.exists("shap_global_prepaid.png"):
            st.image("shap_global_prepaid.png", caption="Global TreeSHAP Beeswarm — Prepaid Subscribers")
    with col_s2:
        if os.path.exists("shap_global_postpaid.png"):
            st.image("shap_global_postpaid.png", caption="Global TreeSHAP Beeswarm — Postpaid Subscribers")

    if os.path.exists("shap_feature_importance.csv"):
        st.markdown("#### Global Mean |SHAP| Importance Table")
        st.dataframe(pd.read_csv("shap_feature_importance.csv").head(15), use_container_width=True)

# ==============================================================================
# TAB 4: SRI LANKA REGIONAL & EXECUTIVE ANALYTICS
# ==============================================================================
with tab4:
    st.subheader("🗺️ Domain Feature Impact & Sri Lanka 25-District Risk Profile")
    if os.path.exists("assets/fig4_domain_features.png"):
        st.image("assets/fig4_domain_features.png", caption="Engineered Domain Features vs. Churn Rate")
    if os.path.exists("assets/fig5_sri_lanka_districts.png"):
        st.image("assets/fig5_sri_lanka_districts.png", caption="Churn Rate & Revenue at Risk Across Sri Lanka's 25 Districts")