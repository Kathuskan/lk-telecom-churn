import pandas as pd
import numpy as np
import joblib
import mlflow
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, precision_recall_curve
)

def evaluate_model(y_true, y_probs, name):
    """Calculates ROC-AUC, PR-AUC, Default 0.50 F1, Optimal Threshold F1, and Top-10% Lift."""
    # 1. Default 0.50 threshold check
    default_preds = (y_probs >= 0.50).astype(int)
    f1_default = f1_score(y_true, default_preds, zero_division=0)
    rec_default = recall_score(y_true, default_preds, zero_division=0)

    # 2. Optimal Probability Threshold from Precision-Recall Curve
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]

    # 3. Overall Ranking Metrics
    roc_auc = roc_auc_score(y_true, y_probs)
    pr_auc = average_precision_score(y_true, y_probs)

    # 4. Top-10% Decile Lift (Business Metric for Retention Teams)
    df_eval = pd.DataFrame({'y_true': y_true.values, 'prob': y_probs}).sort_values('prob', ascending=False)
    top_10_pct_n = int(len(df_eval) * 0.10)
    top_10_churn_rate = df_eval.iloc[:top_10_pct_n]['y_true'].mean()
    baseline_churn_rate = df_eval['y_true'].mean()
    top_10_lift = top_10_churn_rate / baseline_churn_rate
    top_10_recall = df_eval.iloc[:top_10_pct_n]['y_true'].sum() / df_eval['y_true'].sum()

    print(f"\n================ {name} ================")
    print(f"ROC-AUC:                   {roc_auc:.4f}")
    print(f"PR-AUC (Avg Precision):    {pr_auc:.4f}")
    print(f"--- Default 0.50 Threshold ---")
    print(f"F1-Score (@ 0.50):         {f1_default:.4f} | Recall: {rec_default:.4f}")
    print(f"--- Tuned Optimal Threshold ({best_thresh:.4f}) ---")
    print(f"Best F1-Score:             {f1_scores[best_idx]:.4f}")
    print(f"Recall (@ Best Thresh):    {recalls[best_idx]:.4f}")
    print(f"Precision (@ Best Thresh): {precisions[best_idx]:.4f}")
    print(f"--- Business Retention Impact (Top 10% Riskiest Users) ---")
    print(f"Top-10% Lift:              {top_10_lift:.2f}x over baseline")
    print(f"Churners Captured in D1:   {top_10_recall*100:.1f}%")

    return {
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "f1_default_05": round(float(f1_default), 4),
        "optimal_threshold": round(float(best_thresh), 4),
        "best_f1": round(float(f1_scores[best_idx]), 4),
        "best_recall": round(float(recalls[best_idx]), 4),
        "best_precision": round(float(precisions[best_idx]), 4),
        "top_10pct_lift": round(float(top_10_lift), 2),
        "top_10pct_recall": round(float(top_10_recall), 4)
    }

def main():
    print("1. Loading Gold Parquet Splits...")
    train_df = pd.read_parquet("gold_train.parquet")
    val_df = pd.read_parquet("gold_val.parquet")
    test_df = pd.read_parquet("gold_test.parquet")

    # Separate features (X) and target (y)
    drop_cols = ['customer_id', 'churned', 'complaint_text']
    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    # Convert string columns to 'category' so HistGradientBoosting handles them natively
    cat_cols = train_df[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()
    for df in [train_df, val_df, test_df]:
        for c in cat_cols:
            df[c] = df[c].astype('category')

    X_train, y_train = train_df[feature_cols], train_df['churned']
    X_val, y_val = val_df[feature_cols], val_df['churned']
    X_test, y_test = test_df[feature_cols], test_df['churned']

    # Set up MLflow experiment tracking (Automatically logs to Azure ML Studio -> Jobs!)
    mlflow.set_experiment("lk_telecom_churn_step2")

    # ---------------------------------------------------------
    # EXPERIMENT 1: Single Unified Model
    # ---------------------------------------------------------
    with mlflow.start_run(run_name="Exp1_Unified_Model"):
        params = {
            "strategy": "Unified_Single_Model",
            "max_iter": 150,
            "learning_rate": 0.03,
            "max_depth": 4,
            "min_samples_leaf": 100,
            "l2_regularization": 10.0
        }
        mlflow.log_params(params)

        model_uni = HistGradientBoostingClassifier(
            categorical_features=cat_cols,
            max_iter=params["max_iter"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            l2_regularization=params["l2_regularization"],
            random_state=42
        )
        model_uni.fit(X_train, y_train)
        val_probs_uni = model_uni.predict_proba(X_val)[:, 1]
        
        metrics_uni = evaluate_model(y_val, val_probs_uni, "Experiment 1: Unified Model (Val)")
        mlflow.log_metrics(metrics_uni)

    # ---------------------------------------------------------
    # EXPERIMENT 2: Segmented Models (Prepaid vs Postpaid)
    # ---------------------------------------------------------
    with mlflow.start_run(run_name="Exp2_Segmented_Prepaid_Postpaid"):
        mlflow.log_param("strategy", "Segmented_Prepaid_Postpaid_Ensemble")

        # Filter training data by contract_type
        pre_train = train_df[train_df['contract_type'] == 'prepaid']
        post_train = train_df[train_df['contract_type'] == 'postpaid']

        model_pre = HistGradientBoostingClassifier(
            categorical_features=cat_cols, max_iter=150, learning_rate=0.03,
            max_depth=4, min_samples_leaf=100, l2_regularization=10.0, random_state=42
        )
        model_pre.fit(pre_train[feature_cols], pre_train['churned'])

        model_post = HistGradientBoostingClassifier(
            categorical_features=cat_cols, max_iter=150, learning_rate=0.03,
            max_depth=3, min_samples_leaf=100, l2_regularization=15.0, random_state=42
        )
        model_post.fit(post_train[feature_cols], post_train['churned'])

        # Route validation rows to the matching model
        val_probs_seg = np.zeros(len(val_df))
        pre_mask = (val_df['contract_type'] == 'prepaid').values
        post_mask = (val_df['contract_type'] == 'postpaid').values

        val_probs_seg[pre_mask] = model_pre.predict_proba(X_val[pre_mask])[:, 1]
        val_probs_seg[post_mask] = model_post.predict_proba(X_val[post_mask])[:, 1]

        metrics_seg = evaluate_model(y_val, val_probs_seg, "Experiment 2: Segmented Model (Val)")
        mlflow.log_metrics(metrics_seg)

        # Final Unbiased Evaluation on Holdout Test Set
        test_probs_seg = np.zeros(len(test_df))
        test_pre_mask = (test_df['contract_type'] == 'prepaid').values
        test_post_mask = (test_df['contract_type'] == 'postpaid').values
        test_probs_seg[test_pre_mask] = model_pre.predict_proba(X_test[test_pre_mask])[:, 1]
        test_probs_seg[test_post_mask] = model_post.predict_proba(X_test[test_post_mask])[:, 1]
        
        evaluate_model(y_test, test_probs_seg, "FINAL HOLDOUT TEST SET (Segmented Model)")

       # Save exact category order from training data so single-row API inference never scrambles codes!
        cat_categories = {c: train_df[c].cat.categories.tolist() for c in cat_cols}

        bundle = {
            "unified_model": model_uni,
            "prepaid_model": model_pre,
            "postpaid_model": model_post,
            "feature_cols": feature_cols,
            "cat_cols": cat_cols,
            "cat_categories": cat_categories,
            "optimal_threshold": metrics_seg["optimal_threshold"]
        }
        joblib.dump(bundle, "churn_model_bundle.pkl")
        mlflow.log_artifact("churn_model_bundle.pkl")
        print("\nSaved model bundle to 'churn_model_bundle.pkl'")

if __name__ == "__main__":
    main()