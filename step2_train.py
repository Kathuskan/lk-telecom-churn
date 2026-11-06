import pandas as pd
import numpy as np
import joblib
import mlflow
import time
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, precision_recall_curve
)

def evaluate_predictions(y_true, y_probs, model_name, split_name="Val", train_time_sec=0.0):
    """Evaluates probability predictions across default and tuned thresholds + Top-10% Lift."""
    # 1. Default 0.50 Threshold
    default_preds = (y_probs >= 0.50).astype(int)
    f1_default = f1_score(y_true, default_preds, zero_division=0)

    # 2. Precision-Recall Curve Optimal Threshold Tuning
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]

    # 3. Ranking Metrics
    roc_auc = roc_auc_score(y_true, y_probs)
    pr_auc = average_precision_score(y_true, y_probs)

    # 4. Top-10% Decile Lift
    df_eval = pd.DataFrame({'y_true': y_true.values, 'prob': y_probs}).sort_values('prob', ascending=False)
    top_10_n = int(len(df_eval) * 0.10)
    top_10_churn_rate = df_eval.iloc[:top_10_n]['y_true'].mean()
    base_churn_rate = df_eval['y_true'].mean()
    top_10_lift = top_10_churn_rate / base_churn_rate
    top_10_recall = df_eval.iloc[:top_10_n]['y_true'].sum() / df_eval['y_true'].sum()

    return {
        "model_name": model_name,
        "split": split_name,
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "f1_default_05": round(float(f1_default), 4),
        "optimal_threshold": round(float(best_thresh), 4),
        "best_f1": round(float(f1_scores[best_idx]), 4),
        "best_recall": round(float(recalls[best_idx]), 4),
        "best_precision": round(float(precisions[best_idx]), 4),
        "top_10pct_lift": round(float(top_10_lift), 2),
        "top_10pct_recall": round(float(top_10_recall), 4),
        "train_time_sec": round(float(train_time_sec), 2)
    }

def main():
    print("1. Loading Gold Parquet Splits...")
    train_df = pd.read_parquet("gold_train.parquet")
    val_df = pd.read_parquet("gold_val.parquet")
    test_df = pd.read_parquet("gold_test.parquet")

    drop_cols = ['customer_id', 'churned', 'complaint_text']
    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    num_cols = train_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = train_df[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()

    # Align exact category levels across Train, Val, and Test for XGBoost / LightGBM / HistGBM
    cat_categories = {}
    for c in cat_cols:
        train_df[c] = train_df[c].astype('category')
        cats = train_df[c].cat.categories.tolist()
        cat_categories[c] = cats
        val_df[c] = pd.Categorical(val_df[c], categories=cats)
        test_df[c] = pd.Categorical(test_df[c], categories=cats)

    X_train, y_train = train_df[feature_cols], train_df['churned']
    X_val, y_val = val_df[feature_cols], val_df['churned']
    X_test, y_test = test_df[feature_cols], test_df['churned']

    # Preprocessing pipelines for traditional sklearn models (LogReg, DT, RF, ET)
    linear_preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), num_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), cat_cols)
    ])

    tree_preprocessor = ColumnTransformer(transformers=[
        ('num', SimpleImputer(strategy='median'), num_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ]), cat_cols)
    ])

    # Define Unified Candidate Models
    candidate_models = {
        "1_Logistic_Regression": Pipeline([
            ('prep', linear_preprocessor),
            ('clf', LogisticRegression(max_iter=500, random_state=42))
        ]),
        "2_Decision_Tree": Pipeline([
            ('prep', tree_preprocessor),
            ('clf', DecisionTreeClassifier(max_depth=6, min_samples_leaf=100, random_state=42))
        ]),
        "3_Random_Forest": Pipeline([
            ('prep', tree_preprocessor),
            ('clf', RandomForestClassifier(n_estimators=120, max_depth=8, min_samples_leaf=50, n_jobs=-1, random_state=42))
        ]),
        "4_Extra_Trees": Pipeline([
            ('prep', tree_preprocessor),
            ('clf', ExtraTreesClassifier(n_estimators=120, max_depth=8, min_samples_leaf=50, n_jobs=-1, random_state=42))
        ]),
        "5_LightGBM_Unified": LGBMClassifier(
            n_estimators=150, learning_rate=0.03, max_depth=4, num_leaves=15,
            min_child_samples=100, reg_lambda=10.0, random_state=42, verbose=-1
        ),
        "6_XGBoost_Unified": XGBClassifier(
            n_estimators=150, learning_rate=0.03, max_depth=4, min_child_weight=100,
            reg_lambda=10.0, subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", enable_categorical=True, eval_metric="logloss", random_state=42
        ),
        "7_HistGBM_Unified": HistGradientBoostingClassifier(
            categorical_features=cat_cols, max_iter=150, learning_rate=0.03,
            max_depth=4, min_samples_leaf=100, l2_regularization=10.0, random_state=42
        )
    }

    mlflow.set_experiment("lk_telecom_churn_multi_model_benchmark")
    comparison_rows = []
    trained_objects = {}

    print("\n2. Training & Evaluating Unified Models on Validation Split...")
    for name, model in candidate_models.items():
        with mlflow.start_run(run_name=name):
            t0 = time.time()
            if any(k in name for k in ["HistGBM", "XGBoost", "LightGBM"]):
                model.fit(X_train, y_train)
                val_probs = model.predict_proba(X_val)[:, 1]
            else:
                X_tr_obj = X_train.copy()
                X_va_obj = X_val.copy()
                for c in cat_cols:
                    X_tr_obj[c] = X_tr_obj[c].astype(str)
                    X_va_obj[c] = X_va_obj[c].astype(str)
                model.fit(X_tr_obj, y_train)
                val_probs = model.predict_proba(X_va_obj)[:, 1]

            elapsed = time.time() - t0
            metrics = evaluate_predictions(y_val, val_probs, name, "Validation", elapsed)
            comparison_rows.append(metrics)
            trained_objects[name] = model

            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
            print(f"   [{name}] ROC-AUC: {metrics['roc_auc']:.4f} | PR-AUC: {metrics['pr_auc']:.4f} | Best F1: {metrics['best_f1']:.4f} | Top-10% Lift: {metrics['top_10pct_lift']}x ({elapsed:.1f}s)")

    # Filter Prepaid vs Postpaid Training Subsets for Segmented Ensembles
    pre_train = train_df[train_df['contract_type'] == 'prepaid']
    post_train = train_df[train_df['contract_type'] == 'postpaid']
    pre_mask = (val_df['contract_type'] == 'prepaid').values
    post_mask = (val_df['contract_type'] == 'postpaid').values

    # 8. Candidate 8: XGBoost Segmented Ensemble (Prepaid + Postpaid)
    xgb_seg_name = "8_XGBoost_Segmented_Ensemble"
    with mlflow.start_run(run_name=xgb_seg_name):
        t0 = time.time()
        xgb_pre = XGBClassifier(
            n_estimators=150, learning_rate=0.03, max_depth=4, min_child_weight=100,
            reg_lambda=10.0, subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", enable_categorical=True, eval_metric="logloss", random_state=42
        )
        xgb_pre.fit(pre_train[feature_cols], pre_train['churned'])

        xgb_post = XGBClassifier(
            n_estimators=150, learning_rate=0.03, max_depth=3, min_child_weight=100,
            reg_lambda=15.0, subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", enable_categorical=True, eval_metric="logloss", random_state=42
        )
        xgb_post.fit(post_train[feature_cols], post_train['churned'])
        elapsed = time.time() - t0

        val_probs_xgb_seg = np.zeros(len(val_df))
        val_probs_xgb_seg[pre_mask] = xgb_pre.predict_proba(X_val[pre_mask])[:, 1]
        val_probs_xgb_seg[post_mask] = xgb_post.predict_proba(X_val[post_mask])[:, 1]

        metrics_xgb_seg = evaluate_predictions(y_val, val_probs_xgb_seg, xgb_seg_name, "Validation", elapsed)
        comparison_rows.append(metrics_xgb_seg)
        mlflow.log_metrics({k: v for k, v in metrics_xgb_seg.items() if isinstance(v, (int, float))})
        print(f"   [{xgb_seg_name}] ROC-AUC: {metrics_xgb_seg['roc_auc']:.4f} | PR-AUC: {metrics_xgb_seg['pr_auc']:.4f} | Best F1: {metrics_xgb_seg['best_f1']:.4f} | Top-10% Lift: {metrics_xgb_seg['top_10pct_lift']}x ({elapsed:.1f}s)")

    # 9. Candidate 9: HistGradientBoosting Segmented Ensemble (Prepaid + Postpaid)
    hgb_seg_name = "9_HistGBM_Segmented_Ensemble"
    with mlflow.start_run(run_name=hgb_seg_name):
        t0 = time.time()
        hgb_pre = HistGradientBoostingClassifier(
            categorical_features=cat_cols, max_iter=150, learning_rate=0.03,
            max_depth=4, min_samples_leaf=100, l2_regularization=10.0, random_state=42
        )
        hgb_pre.fit(pre_train[feature_cols], pre_train['churned'])

        hgb_post = HistGradientBoostingClassifier(
            categorical_features=cat_cols, max_iter=150, learning_rate=0.03,
            max_depth=3, min_samples_leaf=100, l2_regularization=15.0, random_state=42
        )
        hgb_post.fit(post_train[feature_cols], post_train['churned'])
        elapsed = time.time() - t0

        val_probs_hgb_seg = np.zeros(len(val_df))
        val_probs_hgb_seg[pre_mask] = hgb_pre.predict_proba(X_val[pre_mask])[:, 1]
        val_probs_hgb_seg[post_mask] = hgb_post.predict_proba(X_val[post_mask])[:, 1]

        metrics_hgb_seg = evaluate_predictions(y_val, val_probs_hgb_seg, hgb_seg_name, "Validation", elapsed)
        comparison_rows.append(metrics_hgb_seg)
        mlflow.log_metrics({k: v for k, v in metrics_hgb_seg.items() if isinstance(v, (int, float))})
        print(f"   [{hgb_seg_name}] ROC-AUC: {metrics_hgb_seg['roc_auc']:.4f} | PR-AUC: {metrics_hgb_seg['pr_auc']:.4f} | Best F1: {metrics_hgb_seg['best_f1']:.4f} | Top-10% Lift: {metrics_hgb_seg['top_10pct_lift']}x ({elapsed:.1f}s)")

    # Compile Leaderboard and Select Best Champion Model by Validation PR-AUC & ROC-AUC
    results_df = pd.DataFrame(comparison_rows).sort_values(by=["pr_auc", "roc_auc"], ascending=False)
    print("\n================ 9-MODEL VALIDATION LEADERBOARD ================")
    print(results_df[["model_name", "roc_auc", "pr_auc", "f1_default_05", "optimal_threshold", "best_f1", "best_recall", "top_10pct_lift", "train_time_sec"]].to_string(index=False))

    best_model_row = results_df.iloc[0]
    best_name = best_model_row["model_name"]
    print(f"\n🏆 SELECTED CHAMPION MODEL: {best_name} (PR-AUC: {best_model_row['pr_auc']}, ROC-AUC: {best_model_row['roc_auc']})")

    # Pick the winning Prepaid/Postpaid models for production deployment
    if "XGBoost" in best_name:
        champ_pre, champ_post = xgb_pre, xgb_post
        champ_engine = "xgboost"
    else:
        champ_pre, champ_post = hgb_pre, hgb_post
        champ_engine = "histgbm"

    # Final Unbiased Evaluation on Holdout Test Set
    test_probs = np.zeros(len(test_df))
    test_pre_mask = (test_df['contract_type'] == 'prepaid').values
    test_post_mask = (test_df['contract_type'] == 'postpaid').values
    test_probs[test_pre_mask] = champ_pre.predict_proba(X_test[test_pre_mask])[:, 1]
    test_probs[test_post_mask] = champ_post.predict_proba(X_test[test_post_mask])[:, 1]

    test_metrics = evaluate_predictions(y_test, test_probs, f"{best_name} (Holdout Test)", "Holdout_Test", 0.0)
    print("\n================ CHAMPION MODEL HOLDOUT TEST METRICS ================")
    for k, v in test_metrics.items():
        print(f"   {k}: {v}")

    results_df.to_csv("model_comparison_results.csv", index=False)
    print("\nSaved benchmark table -> 'model_comparison_results.csv'")

    bundle = {
        "champion_model_name": best_name,
        "champion_engine": champ_engine,
        "unified_model": trained_objects["7_HistGBM_Unified"],
        "prepaid_model": champ_pre,
        "postpaid_model": champ_post,
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "cat_categories": cat_categories,
        "optimal_threshold": float(best_model_row["optimal_threshold"]),
        "leaderboard": results_df.to_dict(orient="records"),
        "holdout_test_metrics": test_metrics
    }
    joblib.dump(bundle, "churn_model_bundle.pkl")
    print("Saved champion model bundle -> 'churn_model_bundle.pkl'")

if __name__ == "__main__":
    main()