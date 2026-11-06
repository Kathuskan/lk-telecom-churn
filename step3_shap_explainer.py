import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

def compute_shap_matrix(model, df_subset, feature_cols, cat_cols, engine="xgboost"):
    """Computes exact TreeSHAP values and returns (shap_values, display_df) for plotting."""
    # Prepare display DataFrame where categories are converted to integer codes for color mapping in beeswarm
    df_display = df_subset[feature_cols].copy()
    for c in cat_cols:
        df_display[c] = df_display[c].cat.codes

    if engine == "xgboost":
        # Use XGBoost's native C++ TreeSHAP engine (supports enable_categorical=True natively!)
        dmat = xgb.DMatrix(df_subset[feature_cols], enable_categorical=True)
        # pred_contribs returns shape (n_samples, n_features + 1); last column is bias term
        contribs = model.get_booster().predict(dmat, pred_contribs=True)
        shap_vals = contribs[:, :-1]
    else:
        # Fallback for Scikit-Learn HistGradientBoosting
        explainer = shap.TreeExplainer(model)
        X_trans = model._preprocess_X(df_subset[feature_cols], reset=False)
        shap_vals = explainer.shap_values(X_trans)

    return shap_vals, df_display

def main():
    print("1. Loading Champion Model Bundle & Holdout Test Data...")
    bundle = joblib.load("churn_model_bundle.pkl")
    champion_name = bundle.get("champion_model_name", "8_XGBoost_Segmented_Ensemble")
    champion_engine = bundle.get("champion_engine", "xgboost")
    model_pre = bundle["prepaid_model"]
    model_post = bundle["postpaid_model"]
    feature_cols = bundle["feature_cols"]
    cat_cols = bundle["cat_cols"]
    cat_categories = bundle.get("cat_categories", {})
    opt_thresh = bundle["optimal_threshold"]

    print(f"   Active Champion Model: {champion_name} (Engine: {champion_engine})")

    test_df = pd.read_parquet("gold_test.parquet")
    for c in cat_cols:
        if c in cat_categories:
            test_df[c] = pd.Categorical(test_df[c], categories=cat_categories[c])
        else:
            test_df[c] = test_df[c].astype('category')

    # Sample up to 1,500 Prepaid and 1,000 Postpaid subscribers for crisp global beeswarm plots
    pre_test = test_df[test_df["contract_type"] == "prepaid"].sample(n=1500, random_state=42)
    post_test = test_df[test_df["contract_type"] == "postpaid"].sample(n=1000, random_state=42)

    # =========================================================================
    # 2. GENERATE GLOBAL TREESHAP BEESWARM PLOT (PREPAID)
    # =========================================================================
    print("2. Computing exact TreeSHAP values for Prepaid Holdout Sample (n=1,500)...")
    shap_vals_pre, df_disp_pre = compute_shap_matrix(
        model_pre, pre_test, feature_cols, cat_cols, champion_engine
    )

    plt.figure(figsize=(11, 7))
    shap.summary_plot(
        shap_vals_pre,
        df_disp_pre,
        feature_names=feature_cols,
        max_display=15,
        show=False
    )
    plt.title(f"Global TreeSHAP Root Causes — Prepaid Subscribers ({champion_name})", fontsize=13, pad=15)
    plt.tight_layout()
    plt.savefig("shap_global_prepaid.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("   Saved 'shap_global_prepaid.png'!")

    # =========================================================================
    # 3. GENERATE GLOBAL TREESHAP BEESWARM PLOT (POSTPAID)
    # =========================================================================
    print("3. Computing exact TreeSHAP values for Postpaid Holdout Sample (n=1,000)...")
    shap_vals_post, df_disp_post = compute_shap_matrix(
        model_post, post_test, feature_cols, cat_cols, champion_engine
    )

    plt.figure(figsize=(11, 7))
    shap.summary_plot(
        shap_vals_post,
        df_disp_post,
        feature_names=feature_cols,
        max_display=15,
        show=False
    )
    plt.title(f"Global TreeSHAP Root Causes — Postpaid Subscribers ({champion_name})", fontsize=13, pad=15)
    plt.tight_layout()
    plt.savefig("shap_global_postpaid.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("   Saved 'shap_global_postpaid.png'!")

    # =========================================================================
    # 4. PRINT GLOBAL FEATURE IMPORTANCE RANKING (MEAN |SHAP|)
    # =========================================================================
    mean_abs_shap = np.abs(shap_vals_pre).mean(axis=0)
    shap_imp_df = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap_prepaid": np.round(mean_abs_shap, 4),
        "mean_abs_shap_postpaid": np.round(np.abs(shap_vals_post).mean(axis=0), 4)
    }).sort_values(by="mean_abs_shap_prepaid", ascending=False)

    shap_imp_df.to_csv("shap_feature_importance.csv", index=False)
    print("\n================ TOP 10 GLOBAL TREESHAP DRIVERS (PREPAID VS POSTPAID) ================")
    print(shap_imp_df.head(10).to_string(index=False))

    # =========================================================================
    # 5. LOCAL SUBSCRIBER ROOT-CAUSE ATTRIBUTION & PLAYBOOK VERIFICATION
    # =========================================================================
    probs_pre = model_pre.predict_proba(pre_test[feature_cols])[:, 1]
    top_risk_idx = int(np.argmax(probs_pre))
    sample_cust = pre_test.iloc[top_risk_idx]
    sample_shap = shap_vals_pre[top_risk_idx]

    top_3 = sorted(zip(feature_cols, sample_shap), key=lambda x: x[1], reverse=True)[:3]
    print("\n================ HIGHEST-RISK SUBSCRIBER DIAGNOSTIC CARD ================")
    print(f"   Customer ID       : {sample_cust['customer_id']}")
    print(f"   Churn Probability : {probs_pre[top_risk_idx]*100:.2f}% (Optimal Threshold: {opt_thresh*100:.2f}%)")
    print("   Top 3 Root Causes :")
    for feat, val in top_3:
        print(f"      - {feat} = {sample_cust[feat]} (SHAP Impact: +{val:.4f})")

if __name__ == "__main__":
    main()