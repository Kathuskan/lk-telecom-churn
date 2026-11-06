import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc

# Set clean publication style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
sns.set_palette("deep")
os.makedirs("assets", exist_ok=True)

def main():
    print("1. Loading Gold Splits & Champion Model Bundle for Visualization Suite...")
    train_df = pd.read_parquet("gold_train.parquet")
    val_df = pd.read_parquet("gold_val.parquet")
    test_df = pd.read_parquet("gold_test.parquet")
    bundle = joblib.load("churn_model_bundle.pkl")

    feature_cols = bundle["feature_cols"]
    cat_cols = bundle["cat_cols"]
    cat_categories = bundle.get("cat_categories", {})
    model_pre = bundle["prepaid_model"]
    model_post = bundle["postpaid_model"]
    opt_thresh = bundle["optimal_threshold"]

    for df in [val_df, test_df]:
        for c in cat_cols:
            if c in cat_categories:
                df[c] = pd.Categorical(df[c], categories=cat_categories[c])
            else:
                df[c] = df[c].astype("category")

    # Compute Champion Model Probabilities on Validation Set
    val_probs = np.zeros(len(val_df))
    pre_mask = (val_df["contract_type"] == "prepaid").values
    post_mask = (val_df["contract_type"] == "postpaid").values
    val_probs[pre_mask] = model_pre.predict_proba(val_df.loc[pre_mask, feature_cols])[:, 1]
    val_probs[post_mask] = model_post.predict_proba(val_df.loc[post_mask, feature_cols])[:, 1]
    y_val = val_df["churned"].values

    # =========================================================================
    # CHART 1: 9-MODEL BENCHMARK LEADERBOARD (PR-AUC & TOP-10% DECILE LIFT)
    # =========================================================================
    print("2. Generating Chart 1: 9-Model Benchmark Comparison (assets/fig1_model_benchmark.png)...")
    if os.path.exists("model_comparison_results.csv"):
        bench_df = pd.read_csv("model_comparison_results.csv").sort_values("pr_auc", ascending=True)
        fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

        colors = ["#94a3b8" if "8_XGBoost" not in m else "#2563eb" for m in bench_df["model_name"]]

        # Subplot A: PR-AUC
        bars1 = axes[0].barh(bench_df["model_name"], bench_df["pr_auc"], color=colors, edgecolor="black", height=0.68)
        axes[0].axvline(0.0597, color="#dc2626", linestyle="--", linewidth=1.5, label="Random Baseline (0.0597)")
        axes[0].set_title("Validation PR-AUC (Precision-Recall Area)", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("PR-AUC Score")
        axes[0].set_xlim(0, max(bench_df["pr_auc"]) * 1.18)
        axes[0].legend(loc="lower right")
        for bar in bars1:
            w = bar.get_width()
            axes[0].text(w + 0.0015, bar.get_y() + bar.get_height()/2, f"{w:.4f}", va="center", fontsize=9, fontweight="bold")

        # Subplot B: Top-10% Decile Lift
        bars2 = axes[1].barh(bench_df["model_name"], bench_df["top_10pct_lift"], color=colors, edgecolor="black", height=0.68)
        axes[1].axvline(1.0, color="#dc2626", linestyle="--", linewidth=1.5, label="No-Skill Baseline (1.00x)")
        axes[1].set_title("Top-10% Decile Lift (Call-Center Campaign ROI)", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Lift Multiplier (x)")
        axes[1].set_xlim(0, max(bench_df["top_10pct_lift"]) * 1.18)
        axes[1].legend(loc="lower right")
        for bar in bars2:
            w = bar.get_width()
            axes[1].text(w + 0.04, bar.get_y() + bar.get_height()/2, f"{w:.2f}x", va="center", fontsize=9, fontweight="bold")

        plt.suptitle("9-Model Architecture Benchmark — Champion: 8_XGBoost_Segmented_Ensemble", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig("assets/fig1_model_benchmark.png", dpi=300, bbox_inches="tight")
        plt.close()

    # =========================================================================
    # CHART 2: PRECISION-RECALL THRESHOLD TUNING & F1 OPTIMIZATION CURVE
    # =========================================================================
    print("3. Generating Chart 2: Threshold Optimization Curve (assets/fig2_threshold_optimization.png)...")
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, precisions[:-1], label="Precision", color="#0284c7", linewidth=2)
    ax.plot(thresholds, recalls[:-1], label="Recall (Churn Capture Rate)", color="#16a34a", linewidth=2)
    ax.plot(thresholds, f1_scores, label="F1-Score (Harmonic Mean)", color="#ea580c", linewidth=2.5)

    # Highlight Optimal Threshold vs Default 0.50 Threshold
    ax.axvline(opt_thresh, color="#2563eb", linestyle="--", linewidth=2, label=f"Tuned Optimal Threshold ({opt_thresh:.4f})")
    ax.axvline(0.50, color="#dc2626", linestyle=":", linewidth=2, label="Default Threshold (0.50 -> F1 = 0.0000)")

    ax.set_title("Solving the 5.97% Imbalance Trap: Precision, Recall & F1 vs. Decision Threshold", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Churn Probability Threshold", fontsize=11)
    ax.set_ylabel("Metric Score", fontsize=11)
    ax.set_xlim(0.0, 0.55)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig("assets/fig2_threshold_optimization.png", dpi=300, bbox_inches="tight")
    plt.close()

    # =========================================================================
    # CHART 3: CUMULATIVE GAIN & DECILE LIFT CHART (BUSINESS ROI)
    # =========================================================================
    print("4. Generating Chart 3: Decile Lift & Cumulative Gain (assets/fig3_decile_lift_roi.png)...")
    df_gain = pd.DataFrame({"y_true": y_val, "prob": val_probs}).sort_values("prob", ascending=False).reset_index(drop=True)
    df_gain["decile"] = pd.qcut(df_gain.index, 10, labels=[f"D{i}" for i in range(1, 11)])
    base_rate = df_gain["y_true"].mean()

    decile_summary = df_gain.groupby("decile", observed=False)["y_true"].agg(["count", "mean", "sum"]).reset_index()
    decile_summary["lift"] = decile_summary["mean"] / base_rate
    decile_summary["cum_churn_pct"] = (decile_summary["sum"].cumsum() / df_gain["y_true"].sum()) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: Lift by Decile
    bar_colors = ["#1d4ed8" if i == 0 else "#3b82f6" if i < 3 else "#94a3b8" for i in range(10)]
    bars = axes[0].bar(decile_summary["decile"], decile_summary["lift"], color=bar_colors, edgecolor="black")
    axes[0].axhline(1.0, color="#dc2626", linestyle="--", linewidth=1.5, label="Random Calling (1.0x)")
    axes[0].set_title("Churn Lift Multiplier by Risk Decile (D1 = Top 10% Riskiest)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Subscriber Risk Decile")
    axes[0].set_ylabel("Lift over Base Churn Rate (x)")
    axes[0].legend()
    for bar in bars:
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2, h + 0.05, f"{h:.2f}x", ha="center", fontsize=9, fontweight="bold")

    # Right: Cumulative % of Churners Captured
    pop_pct = np.arange(0, 110, 10)
    cum_gain = np.concatenate([[0], decile_summary["cum_churn_pct"].values])
    axes[1].plot(pop_pct, cum_gain, marker="o", color="#16a34a", linewidth=2.5, label="Champion XGBoost Ensemble")
    axes[1].plot(pop_pct, pop_pct, linestyle="--", color="#dc2626", linewidth=1.5, label="Random Baseline")
    axes[1].set_title("Cumulative Gain Curve (% of Total Churners Captured)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("% of Subscribers Contacted (Sorted by Predicted Risk)")
    axes[1].set_ylabel("% of Actual Churners Captured")
    axes[1].legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("assets/fig3_decile_lift_roi.png", dpi=300, bbox_inches="tight")
    plt.close()

    # =========================================================================
    # CHART 4: DOMAIN FEATURE ENGINEERING IMPACT (4-PANEL SUBPLOT)
    # =========================================================================
    print("5. Generating Chart 4: Domain Feature Engineering Impact (assets/fig4_domain_features.png)...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    base_pct = train_df["churned"].mean() * 100

    # Panel 1: Support Status vs Churn Rate
    sup_df = train_df.groupby("support_status", observed=False)["churned"].mean().mul(100).reindex(
        ["No_Complaint", "Resolved_Complaint", "Unresolved_Complaint"]
    )
    b1 = axes[0, 0].bar(sup_df.index, sup_df.values, color=["#22c55e", "#f59e0b", "#ef4444"], edgecolor="black", width=0.55)
    axes[0, 0].axhline(base_pct, color="black", linestyle="--", label=f"Base Churn ({base_pct:.2f}%)")
    axes[0, 0].set_title("A. Churn Rate by Engineered Support Status", fontweight="bold")
    axes[0, 0].set_ylabel("Churn Rate (%)")
    axes[0, 0].legend()
    for b in b1:
        axes[0, 0].text(b.get_x() + b.get_width()/2, b.get_height() + 0.2, f"{b.get_height():.2f}%", ha="center", fontweight="bold")

    # Panel 2: Prepaid 30-Day Inactivity Flag vs Churn Rate
    pre_only = train_df[train_df["contract_type"] == "prepaid"]
    inact_df = pre_only.groupby("is_prepaid_inactive_30d", observed=False)["churned"].mean().mul(100)
    inact_labels = ["Active (<=30d Recharge)", "Inactive (>30d No Recharge)"]
    b2 = axes[0, 1].bar(inact_labels, inact_df.values, color=["#3b82f6", "#dc2626"], edgecolor="black", width=0.5)
    axes[0, 1].axhline(base_pct, color="black", linestyle="--", label=f"Base Churn ({base_pct:.2f}%)")
    axes[0, 1].set_title("B. Prepaid Churn Rate by 30-Day Recharge Inactivity", fontweight="bold")
    axes[0, 1].set_ylabel("Churn Rate (%)")
    axes[0, 1].legend()
    for b in b2:
        axes[0, 1].text(b.get_x() + b.get_width()/2, b.get_height() + 0.2, f"{b.get_height():.2f}%", ha="center", fontweight="bold")

    # Panel 3: Postpaid Overdue Bill Flag vs Churn Rate
    post_only = train_df[train_df["contract_type"] == "postpaid"]
    over_df = post_only.groupby("has_overdue_bill", observed=False)["churned"].mean().mul(100)
    over_labels = ["Paid On Time", "Has Overdue Bill (>0 Days)"]
    b3 = axes[1, 0].bar(over_labels, over_df.values, color=["#10b981", "#e11d48"], edgecolor="black", width=0.5)
    axes[1, 0].axhline(base_pct, color="black", linestyle="--", label=f"Base Churn ({base_pct:.2f}%)")
    axes[1, 0].set_title("C. Postpaid Churn Rate by Bill Delinquency Status", fontweight="bold")
    axes[1, 0].set_ylabel("Churn Rate (%)")
    axes[1, 0].legend()
    for b in b3:
        axes[1, 0].text(b.get_x() + b.get_width()/2, b.get_height() + 0.2, f"{b.get_height():.2f}%", ha="center", fontweight="bold")

    # Panel 4: Operator & Contract Type Churn Comparison
    op_df = train_df.groupby(["operator", "contract_type"], observed=False)["churned"].mean().mul(100).unstack()
    op_df.plot(kind="bar", ax=axes[1, 1], color=["#6366f1", "#f97316"], edgecolor="black", rot=0)
    axes[1, 1].axhline(base_pct, color="black", linestyle="--", label=f"Base Churn ({base_pct:.2f}%)")
    axes[1, 1].set_title("D. Churn Rate by Sri Lankan Operator & Contract Segment", fontweight="bold")
    axes[1, 1].set_ylabel("Churn Rate (%)")
    axes[1, 1].set_xlabel("Telecom Operator")
    axes[1, 1].legend(title="Contract Type")

    plt.suptitle("Domain Feature Engineering & Behavioral Churn Forensics (Gold Layer)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("assets/fig4_domain_features.png", dpi=300, bbox_inches="tight")
    plt.close()

    # =========================================================================
    # CHART 5: SRI LANKA 25 DISTRICTS REGIONAL CHURN & REVENUE AT RISK
    # =========================================================================
    print("6. Generating Chart 5: Sri Lanka 25 Districts Risk Profile (assets/fig5_sri_lanka_districts.png)...")
    reg_df = train_df.groupby("region", observed=False).agg(
        subscribers=("customer_id", "count"),
        churn_rate_pct=("churned", lambda x: x.mean() * 100),
        monthly_rev_at_risk_lkr=("monthly_price_lkr", lambda s: s[train_df.loc[s.index, "churned"] == 1].sum())
    ).sort_values("churn_rate_pct", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 8), sharey=True)
    axes[0].barh(reg_df.index, reg_df["churn_rate_pct"], color="#0284c7", edgecolor="black")
    axes[0].axvline(base_pct, color="#dc2626", linestyle="--", linewidth=1.5, label=f"National Avg ({base_pct:.2f}%)")
    axes[0].set_title("Churn Rate (%) Across Sri Lanka's 25 Districts", fontweight="bold")
    axes[0].set_xlabel("Churn Rate (%)")
    axes[0].legend(loc="lower right")

    axes[1].barh(reg_df.index, reg_df["monthly_rev_at_risk_lkr"] / 1000.0, color="#e11d48", edgecolor="black")
    axes[1].set_title("Monthly Churned Revenue at Risk (Thousands LKR)", fontweight="bold")
    axes[1].set_xlabel("Churned Revenue (LKR '000s)")

    plt.suptitle("Geospatial Telecom Churn Audit Across Sri Lanka's 25 Administrative Districts", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("assets/fig5_sri_lanka_districts.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Copy existing SHAP plots into assets/ for clean GitHub README links
    for shap_file in ["shap_global_prepaid.png", "shap_global_postpaid.png"]:
        if os.path.exists(shap_file):
            import shutil
            shutil.copy(shap_file, os.path.join("assets", shap_file))

    print("\nDone! All high-resolution visual charts saved in 'assets/'.")

if __name__ == "__main__":
    main()