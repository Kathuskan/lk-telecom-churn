import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder

def recommend_retention_action(row, top_shap_features):
    """Maps mathematical SHAP root causes + customer state to a business retention playbook."""
    top_feat_names = [f[0] for f in top_shap_features]

    # Rule 1: Unresolved Support Complaint
    if row.get('support_status') == 'Unresolved_Complaint' or 'support_status' in top_feat_names or 'satisfaction_score' in top_feat_names:
        if row.get('support_status') == 'Unresolved_Complaint':
            return "URGENT: Escalate open support ticket to Tier-2 specialist within 4 hours + issue 5GB apology data bonus."

    # Rule 2: Prepaid Recharge Decay / Inactivity
    if row.get('contract_type') == 'prepaid' and (
        'days_since_last_recharge' in top_feat_names or 'recharge_velocity_ratio' in top_feat_names or 'recharge_trend' in top_feat_names
    ):
        return "PREPAID RETENTION: Send automated SMS reload offer ('Reload Rs. 200 today and get 100% bonus data for 7 days')."

    # Rule 3: Postpaid Bill Delinquency
    if row.get('contract_type') == 'postpaid' and ('overdue_days' in top_feat_names or 'late_fee_lkr' in top_feat_names):
        return "POSTPAID RETENTION: Offer late-fee waiver upon bill settlement + suggest flexible installment plan."

    # Rule 4: Data Exhaustion / Plan Mismatch
    if row.get('is_data_exhausted') == 1 or 'data_utilization_ratio' in top_feat_names:
        return "PLAN OPTIMIZATION: Customer is exhausting data allowance. Offer 20% discount to upgrade to next tier."

    return "GENERAL RETENTION: Enrol in Loyalty Bonus Tier and offer 2GB weekend data pack."

def main():
    print("1. Loading trained model bundle and Holdout Test Data...")
    bundle = joblib.load("churn_model_bundle.pkl")
    test_df = pd.read_parquet("gold_test.parquet")

    feature_cols = bundle["feature_cols"]
    cat_cols = bundle["cat_cols"]
    opt_thresh = bundle["optimal_threshold"]

    # Ordinal encode categorical columns to integers for SHAP compatibility
    print("2. Encoding categorical features to integers for SHAP...")
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    
    X_test = test_df[feature_cols].copy()
    X_test[cat_cols] = encoder.fit_transform(X_test[cat_cols].astype(str))

    model_pre = bundle["prepaid_model"]
    model_post = bundle["postpaid_model"]

    print("3. Scoring Holdout Test Set to find the Highest-Risk Customers...")
    pre_mask = (test_df['contract_type'] == 'prepaid').values
    post_mask = (test_df['contract_type'] == 'postpaid').values

    # Score using original test dataframe (models handle categories natively during scoring)
    X_test_raw = test_df[feature_cols]
    test_df['churn_prob'] = 0.0
    test_df.loc[pre_mask, 'churn_prob'] = model_pre.predict_proba(X_test_raw[pre_mask])[:, 1]
    test_df.loc[post_mask, 'churn_prob'] = model_post.predict_proba(X_test_raw[post_mask])[:, 1]
    test_df['flagged_churn'] = (test_df['churn_prob'] >= opt_thresh).astype(int)

    # 4. Initialize TreeSHAP Explainers using integer-encoded feature matrices
    print("4. Computing Exact TreeSHAP Values...")
    explainer_pre = shap.TreeExplainer(model_pre)
    explainer_post = shap.TreeExplainer(model_post)

    # Save a Global SHAP Summary Plot for the Prepaid Model (Top 500 sample)
    pre_sample = X_test[pre_mask].iloc[:500]
    shap_values_pre = explainer_pre.shap_values(pre_sample)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_pre, pre_sample, max_display=12, show=False)
    plt.title("Global TreeSHAP Feature Impact (Prepaid Churn Model)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("shap_global_prepaid.png", dpi=150)
    plt.close()
    print("   Saved global SHAP beeswarm chart -> 'shap_global_prepaid.png'")

    # 5. Generate Individual Actionable Retention Cards
    print("\n================ ACTIONABLE RETENTION AI CARDS ================")
    top_prepaid = test_df[pre_mask].sort_values('churn_prob', ascending=False).head(3)
    top_postpaid = test_df[post_mask].sort_values('churn_prob', ascending=False).head(2)

    for label, subset, explainer in [
        ("HIGH-RISK PREPAID", top_prepaid, explainer_pre),
        ("HIGH-RISK POSTPAID", top_postpaid, explainer_post)
    ]:
        for idx, row in subset.iterrows():
            # Get encoded single row for SHAP
            x_single_encoded = X_test.loc[[idx]]
            sv = explainer.shap_values(x_single_encoded)[0]

            # Pair feature names with their positive SHAP contribution
            feat_impacts = sorted(
                zip(feature_cols, sv, test_df.loc[idx, feature_cols].values),
                key=lambda x: x[1],
                reverse=True
            )
            top_3_drivers = feat_impacts[:3]
            action = recommend_retention_action(row, top_3_drivers)

            print(f"\n[{label}] Customer ID: {row['customer_id']} | Operator: {row['operator']} ({row['region']})")
            print(f"   Predicted Churn Risk: {row['churn_prob']*100:.1f}% (Threshold: {opt_thresh*100:.1f}%) | Actual Churned: {bool(row['churned'])}")
            print("   Top 3 SHAP Churn Drivers:")
            for fname, sval, fval in top_3_drivers:
                print(f"      - {fname} = {fval} (SHAP impact: +{sval:.4f})")
            print(f"   Recommended Action: -> {action}")

if __name__ == "__main__":
    main()