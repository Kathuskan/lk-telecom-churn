import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def clean_and_engineer_telecom_data(filepath: str) -> pd.DataFrame:
    print(f"1. Loading Bronze data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"   Raw shape: {df.shape}")

    # ==========================================
    # SILVER LAYER: DATA CLEANING & CANONICALIZATION
    # ==========================================
    print("2. Standardizing dirty categorical strings...")
    
    # Fix Operator casing ('dialog' -> 'Dialog', etc.)
    operator_map = {
        'dialog': 'Dialog',
        'slt-mobitel': 'SLT-Mobitel',
        'hutch': 'Hutch'
    }
    df['operator'] = df['operator'].astype(str).str.strip().str.lower().map(operator_map)

    # Fix Sri Lankan District casing ('GAMPAHA' -> 'Gampaha', 'NUWARA ELIYA' -> 'Nuwara Eliya')
    # Reduces 50 dirty region strings down to the exact 25 administrative districts
    df['region'] = df['region'].astype(str).str.strip().str.title()

    # Canonicalize bundled_apps ('WhatsApp|FB' -> 'FB|WhatsApp')
    def normalize_apps(val):
        if pd.isna(val):
            return "None"
        val_str = str(val).strip()
        if "|" in val_str:
            return "|".join(sorted([app.strip() for app in val_str.split("|")]))
        return val_str

    df['bundled_apps'] = df['bundled_apps'].apply(normalize_apps)

    # Deduplicate records now that casing inconsistencies are resolved
    before_dedup = len(df)
    df = df.drop_duplicates(subset=['customer_id'], keep='first'). reset_index(drop=True)
    print(f"   Removed {before_dedup - len(df)} duplicate rows. Clean shape: {df.shape}")

    # Drop zero-variance column ('is_active' is True for 100% of rows)
    if 'is_active' in df.columns:
        df = df.drop(columns=['is_active'])

    # ==========================================
    # GOLD LAYER: DOMAIN FEATURE ENGINEERING
    # ==========================================
    print("3. Engineering Telecom Domain Features...")
    snapshot_date = pd.to_datetime("2026-10-03")

    # 3.1 Temporal Features
    df['subscription_activated_at'] = pd.to_datetime(df['subscription_activated_at'])
    df['days_since_plan_activation'] = (snapshot_date - df['subscription_activated_at']).dt.days

    # 3.2 Usage & Allowance Ratios (Prevent division by zero with +1e-5)
    # Convert data_allowance_gb to MB (1 GB = 1024 MB) to match data_used_mb
    df['data_allowance_mb'] = df['data_allowance_gb'] * 1024.0
    df['data_utilization_ratio'] = df['data_used_mb'] / (df['data_allowance_mb'] + 1e-5)
    df['voice_utilization_ratio'] = df['voice_minutes_used'] / (df['voice_allowance_mins'] + 1.0)
    df['sms_utilization_ratio'] = df['sms_count'] / (df['sms_allowance'] + 1.0)

    # Flag customers who are severly under-utilizing (<10%) or over-utilizing (>100%) data
    df['is_data_underutilized'] = (df['data_utilization_ratio'] < 0.10).astype(int)
    df['is_data_exhausted'] = (df['data_utilization_ratio'] >= 0.99).astype(int)

    # 3.3 Prepaid Recharge Velocity & Decay Features
    # Ratio < 1.0 means their latest recharge is smaller than their 3-month average
    df['recharge_velocity_ratio'] = np.where(
        df['contract_type'] == 'prepaid',
        df['recharge_amount_lkr'] / (df['recharge_amount_3mo_avg_lkr'] + 1.0),
        np.nan
    )
    df['is_prepaid_inactive_30d'] = np.where(
        (df['contract_type'] == 'prepaid') & (df['days_since_last_recharge'] > 30), 1, 0
    )

    # 3.4 Postpaid Delinquency Features
    df['has_overdue_bill'] = np.where(
        (df['contract_type'] == 'postpaid') & (df['overdue_days'] > 0), 1, 0
    )
    df['bill_to_plan_price_ratio'] = np.where(
        df['contract_type'] == 'postpaid',
        df['bill_amount_lkr'] / (df['monthly_price_lkr'] + 1.0),
        np.nan
    )

    # 3.5 Support & Complaint Severity Features (The #1 Churn Driver!)
    # Create an explicit interaction feature for complaint resolution status
    df['support_status'] = np.select(
        [
            df['is_complaint'] == False,
            (df['is_complaint'] == True) & (df['complaint_resolved'] == True),
            (df['is_complaint'] == True) & (df['complaint_resolved'] == False)
        ],
        ['No_Complaint', 'Resolved_Complaint', 'Unresolved_Complaint'],
        default='No_Complaint'
    )

    # 3.6 Multi-Hot App Bundle Parsing
    for app in ['FB', 'WhatsApp', 'YouTube', 'TikTok', 'Netflix']:
        df[f'app_has_{app.lower()}'] = df['bundled_apps'].str.contains(app, case=False, na=False).astype(int)

    # ==========================================
    # STRUCTURAL MISSING VALUE HANDLING
    # ==========================================
    print("4. Handling Structural Missing Values safely...")
    
    # Categorical columns with structural NaNs
    df['device_type'] = df['device_type'].fillna('Unknown')
    df['device_brand'] = df['device_brand'].fillna('Unknown')
    df['device_financed'] = df['device_financed'].map({True: 'Yes', False: 'No'}).fillna('Unknown')
    df['recharge_trend'] = df['recharge_trend'].fillna('Not_Applicable_Postpaid')
    df['complaint_type'] = df['complaint_type'].fillna('No_Complaint')
    df['escalated'] = df['escalated'].map({True: 'Yes', False: 'No'}).fillna('No_Complaint')

    # Convert boolean columns to clean integers (0/1) for ML compatibility
    bool_cols = ['kyc_verified', 'auto_renew', 'roaming_flag', 'has_support_contact_30d', 'is_complaint', 'churned']
    for bcol in bool_cols:
        df[bcol] = df[bcol].astype(int)

    # Drop raw date strings and redundant columns before modeling
    drop_cols = ['signup_date', 'subscription_activated_at', 'last_recharge_date', 'last_bill_date', 'complaint_resolved']
    df_gold = df.drop(columns=drop_cols)

    print(f"   Final Gold dataset shape: {df_gold.shape}")
    return df_gold
def inspect_and_prove_features(train_df: pd.DataFrame):
    """Generates statistical proof of engineered numerical & categorical features on the training split."""
    print("\n6. Inspecting Features & Generating Statistical Proof...")
    drop_cols = ['customer_id', 'churned', 'complaint_text']
    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    num_cols = train_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = train_df[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()

    # 1. Numerical Feature Correlation & Missingness Audit
    engineered_features = {
        'days_since_plan_activation', 'data_allowance_mb', 'data_utilization_ratio',
        'voice_utilization_ratio', 'sms_utilization_ratio', 'is_data_underutilized',
        'is_data_exhausted', 'recharge_velocity_ratio', 'is_prepaid_inactive_30d',
        'has_overdue_bill', 'bill_to_plan_price_ratio', 'app_has_fb',
        'app_has_whatsapp', 'app_has_youtube', 'app_has_tiktok', 'app_has_netflix'
    }

    corr_rows = []
    for col in num_cols:
        corr_val = train_df[col].corr(train_df['churned'])
        missing_pct = train_df[col].isna().mean() * 100
        corr_rows.append({
            "feature": col,
            "origin": "Engineered (Gold)" if col in engineered_features else "Raw / Cleaned (Silver)",
            "abs_corr_with_churn": round(abs(float(corr_val)), 4) if pd.notna(corr_val) else 0.0,
            "raw_corr_with_churn": round(float(corr_val), 4) if pd.notna(corr_val) else 0.0,
            "missing_pct": round(float(missing_pct), 2)
        })

    corr_df = pd.DataFrame(corr_rows).sort_values(by="abs_corr_with_churn", ascending=False)
    corr_df.to_csv("feature_inspection_summary.csv", index=False)

    print("   --- Top 10 Numerical Features by Correlation with Churn ---")
    print(corr_df.head(10).to_string(index=False))

    # 2. Categorical Feature Impact Proof (Churn Rate Spread Across Categories)
    cat_proof_rows = []
    base_churn = train_df['churned'].mean() * 100
    for c in cat_cols:
        grp = train_df.groupby(c, observed=False)['churned'].agg(['count', 'mean'])
        max_cat = grp['mean'].idxmax()
        min_cat = grp['mean'].idxmin()
        max_rate = grp['mean'].max() * 100
        min_rate = grp['mean'].min() * 100
        cat_proof_rows.append({
            "categorical_feature": c,
            "origin": "Engineered (Gold)" if c == "support_status" else "Cleaned (Silver)",
            "unique_values": train_df[c].nunique(),
            "highest_risk_group": f"{max_cat} ({max_rate:.2f}%)",
            "lowest_risk_group": f"{min_cat} ({min_rate:.2f}%)",
            "risk_spread_pct": round(float(max_rate - min_rate), 2),
            "max_lift_over_base": round(float(max_rate / base_churn), 2)
        })

    cat_proof_df = pd.DataFrame(cat_proof_rows).sort_values(by="risk_spread_pct", ascending=False)
    cat_proof_df.to_csv("categorical_churn_proof.csv", index=False)

    print("\n   --- Top Categorical Features by Churn Risk Spread ---")
    print(cat_proof_df.head(8).to_string(index=False))
    print("   Saved 'feature_inspection_summary.csv' and 'categorical_churn_proof.csv'!")
    
    
if __name__ == "__main__":
    gold_df = clean_and_engineer_telecom_data("lk_telecom_synthetic_100k.csv")
    
    # Stratified Train/Validation/Test Split (80% Train, 10% Val, 10% Test)
    # Why Stratified? Because churn is only 5.97%! Stratified ensures all splits have exactly 5.97% churn.
    train_df, temp_df = train_test_split(
        gold_df, test_size=0.20, stratify=gold_df['churned'], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df['churned'], random_state=42
    )

    print(f"5. Saving Stratified Splits:")
    print(f"   Train: {train_df.shape} (Churn rate: {train_df['churned'].mean()*100:.2f}%)")
    print(f"   Val:   {val_df.shape} (Churn rate: {val_df['churned'].mean()*100:.2f}%)")
    print(f"   Test:  {test_df.shape} (Churn rate: {test_df['churned'].mean()*100:.2f}%)")

    train_df.to_parquet("gold_train.parquet", index=False)
    val_df.to_parquet("gold_val.parquet", index=False)
    test_df.to_parquet("gold_test.parquet", index=False)
    print("Done! Saved gold_train.parquet, gold_val.parquet, and gold_test.parquet.")