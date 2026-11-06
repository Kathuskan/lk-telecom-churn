from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import joblib
import pandas as pd
import shap
import numpy as np

# Initialize FastAPI app
app = FastAPI(
    title="LK Telecom Churn & Explainable AI Engine",
    description="Production REST API for real-time customer churn prediction and TreeSHAP root-cause attribution.",
    version="2.0"
)

# Load model bundle at startup
print("Loading champion model bundle into memory...")
bundle = joblib.load("churn_model_bundle.pkl")
champion_name = bundle.get("champion_model_name", "Segmented_Ensemble")
champion_engine = bundle.get("champion_engine", "histgbm")
model_pre = bundle["prepaid_model"]
model_post = bundle["postpaid_model"]
feature_cols = bundle["feature_cols"]
cat_cols = bundle["cat_cols"]
opt_thresh = bundle["optimal_threshold"]

# Extract exact category mappings from bundle or directly from trained scikit-learn models
# so single-row inference never scrambles category integer codes!
def extract_model_categories(model, cat_columns):
    if "cat_categories" in bundle and bundle["cat_categories"]:
        return bundle["cat_categories"]
    try:
        # Fallback: Access fitted OrdinalEncoder inside HistGradientBoosting's preprocessor
        ord_encoder = model._preprocessor.named_transformers_['encoder']
        return {col: list(cats) for col, cats in zip(cat_columns, ord_encoder.categories_)}
    except Exception:
        return {}

pre_categories = extract_model_categories(model_pre, cat_cols)
post_categories = extract_model_categories(model_post, cat_cols)

# Initialize TreeSHAP explainers for both Prepaid and Postpaid models
explainer_pre = shap.TreeExplainer(model_pre)
explainer_post = shap.TreeExplainer(model_post)

# Define request body structure using Pydantic v2 Optional types for structural NaNs
class CustomerPayload(BaseModel):
    customer_id: str
    contract_type: str  # 'prepaid' or 'postpaid'
    operator: str
    region: str
    tenure_days: int
    kyc_verified: int
    id_type: str
    age_bracket: str
    device_type: str
    device_brand: str
    device_financed: str
    sim_type: str
    sim_swap_count: int
    customer_segment: str
    loyalty_tier: str
    plan_name: str
    monthly_price_lkr: int
    data_allowance_gb: float
    voice_allowance_mins: int
    sms_allowance: int
    plan_category: str
    plan_duration_type: str
    bundled_apps: str
    plan_changes_count: int
    auto_renew: int
    days_since_last_recharge: Optional[float] = None
    recharge_amount_lkr: Optional[float] = None
    recharge_amount_3mo_avg_lkr: Optional[float] = None
    recharge_trend: Optional[str] = "Not_Applicable_Postpaid"
    bill_amount_lkr: Optional[float] = None
    payment_method: str
    balance_after_lkr: Optional[float] = None
    overdue_days: Optional[float] = None
    late_fee_lkr: Optional[float] = None
    data_used_mb: float
    data_used_since_activation_mb: float
    primary_network_type: str
    voice_minutes_used: float
    sms_count: int
    roaming_flag: int
    has_support_contact_30d: int
    is_complaint: int
    complaint_type: Optional[str] = "No_Complaint"
    resolution_time_hours: Optional[float] = None
    satisfaction_score: Optional[float] = None
    escalated: Optional[str] = "No_Complaint"
    # Engineered features
    days_since_plan_activation: int
    data_allowance_mb: float
    data_utilization_ratio: float
    voice_utilization_ratio: float
    sms_utilization_ratio: float
    is_data_underutilized: int
    is_data_exhausted: int
    recharge_velocity_ratio: Optional[float] = None
    is_prepaid_inactive_30d: int = 0
    has_overdue_bill: int = 0
    bill_to_plan_price_ratio: Optional[float] = None
    support_status: str
    app_has_fb: int = 0
    app_has_whatsapp: int = 0
    app_has_youtube: int = 0
    app_has_tiktok: int = 0
    app_has_netflix: int = 0

@app.get("/health")
def health_check():
    """Public health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": True,
        "champion_model": champion_name,
        "engine": champion_engine,
        "threshold": opt_thresh
    }

@app.post("/predict")
def predict_churn(customer: CustomerPayload):
    """Takes customer attributes, predicts churn probability, and returns TreeSHAP root causes."""
    try:
        # Convert Pydantic payload to DataFrame (replaces None with np.nan automatically)
        df_input = pd.DataFrame([customer.model_dump() if hasattr(customer, "model_dump") else customer.dict()])
        df_input = df_input.replace({None: np.nan})

        contract = customer.contract_type.lower()
        if contract == 'prepaid':
            model = model_pre
            explainer = explainer_pre
            cat_map = pre_categories
        elif contract == 'postpaid':
            model = model_post
            explainer = explainer_post
            cat_map = post_categories
        else:
            raise HTTPException(status_code=400, detail="Invalid contract_type. Must be 'prepaid' or 'postpaid'.")

        # 1. Apply the EXACT training category order so single-row inference matches training codes!
        for c in cat_cols:
            if c in df_input.columns:
                if c in cat_map:
                    df_input[c] = pd.Categorical(df_input[c], categories=cat_map[c])
                else:
                    df_input[c] = df_input[c].astype('category')

        prob = float(model.predict_proba(df_input[feature_cols])[:, 1][0])
        is_flagged = bool(prob >= opt_thresh)

        # 2. Dynamically prepare feature matrix for TreeSHAP (Supports both HistGBM and XGBoost/LightGBM)
        if hasattr(model, "_preprocess_X"):
            # Scikit-Learn HistGradientBoosting internal preprocessor
            X_for_shap = model._preprocess_X(df_input[feature_cols], reset=False)
        else:
            # XGBoost / LightGBM native categorical DataFrame support
            X_for_shap = df_input[feature_cols]

        shap_vals = explainer.shap_values(X_for_shap)
        # Handle cases where shap_values returns a list (binary classification in some versions) vs 2D array
        if isinstance(shap_vals, list):
            sv = shap_vals[1][0]
        elif len( np.shape(shap_vals) ) == 3:
            sv = shap_vals[0, :, 1]
        else:
            sv = shap_vals[0]

        raw_vals = df_input[feature_cols].iloc[0].to_dict()

        top_drivers = sorted(
            zip(feature_cols, sv),
            key=lambda x: x[1],
            reverse=True
        )[:3]

        drivers_formatted = [
            {"feature": f, "value": str(raw_vals[f]), "shap_impact": round(float(imp), 4)}
            for f, imp in top_drivers
        ]

        # 3. Map to Retention Playbook Action
        top_feat_names = [f for f, _ in top_drivers]
        if customer.support_status == 'Unresolved_Complaint':
            action = "URGENT: Escalate open support ticket to Tier-2 specialist within 4 hours + issue 5GB apology data bonus."
        elif contract == 'prepaid' and any(k in top_feat_names for k in ['days_since_last_recharge', 'recharge_velocity_ratio', 'recharge_trend', 'is_prepaid_inactive_30d']):
            action = "PREPAID RETENTION: Send automated SMS reload offer ('Reload Rs. 200 today and get 100% bonus data for 7 days')."
        elif contract == 'postpaid' and any(k in top_feat_names for k in ['overdue_days', 'late_fee_lkr', 'has_overdue_bill']):
            action = "POSTPAID RETENTION: Offer late-fee waiver upon bill settlement + suggest flexible installment plan."
        else:
            action = "GENERAL RETENTION: Enrol in Loyalty Bonus Tier and offer 2GB weekend data pack."

        return {
            "customer_id": customer.customer_id,
            "champion_model": champion_name,
            "churn_probability": round(prob, 4),
            "optimal_threshold": opt_thresh,
            "is_flagged_at_risk": is_flagged,
            "top_root_causes": drivers_formatted,
            "recommended_retention_action": action
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))