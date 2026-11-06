"""Portfolio Value & Stress-Testing Simulation Script.

Simulates the business and financial impact of deploying the AI Credit Risk Engine
and Automated Restructuring Advisor across 2,594 out-of-time loan applicants.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.engine import RiskEngine
from src.schemas import ApplicantInput


def run_portfolio_simulation():
    print("=" * 70)
    print("=== RUNNING PORTFOLIO VALUE & STRESS-TESTING SIMULATION ===")
    print("=" * 70)

    # 1. Initialize Risk Engine
    engine = RiskEngine()
    print("[OK] Loaded RiskEngine and serialized underwriting artifacts.")

    # 2. Load and prepare out-of-time test cohort
    df_path = os.path.join(os.path.dirname(__file__), "..", "notebooks", "df_clean.csv")
    df = pd.read_csv(df_path)
    applicants = df[df["has_loan"] == "Yes"].copy()

    # Ground-truth default probability formulation (Path B standard)
    np.random.seed(42)
    dti = applicants["monthly_emi_usd"] / applicants["monthly_income_usd"]
    runway_inv = applicants["monthly_expenses_usd"] / (applicants["savings_usd"] + 1000)
    credit_penalty = (850 - applicants["credit_score"]) / 550
    leverage = applicants["loan_amount_usd"] / (applicants["monthly_income_usd"] * 12)
    life_shock_noise = np.random.normal(loc=0.0, scale=0.75, size=len(applicants))

    log_odds = (
        -2.8
        + 2.2 * np.clip(dti, 0, 2.0)
        + 1.8 * credit_penalty
        + 0.8 * np.clip(runway_inv, 0, 2.0)
        + 0.5 * np.clip(leverage, 0, 5.0)
        + life_shock_noise
    )
    applicants["default_prob_ground_truth"] = 1 / (1 + np.exp(-log_odds))
    applicants["actual_default"] = (
        np.random.uniform(0, 1, size=len(applicants)) < applicants["default_prob_ground_truth"]
    ).astype(int)

    # Split out-of-time test set
    applicants["record_date"] = pd.to_datetime(applicants["record_date"])
    split_date = applicants["record_date"].quantile(0.80)
    test_df = applicants[applicants["record_date"] > split_date].copy().reset_index(drop=True)
    n_test = len(test_df)
    print(f"[OK] Extracted Out-of-Time Test Cohort: {n_test:,} borrowers.\n")

    # 3. Baseline Batch Underwriting
    print("Processing baseline model scoring...")
    X_test_df = test_df[engine.feature_cols]
    y_prob = engine.pipeline.predict_proba(X_test_df)[:, 1]
    test_df["pred_prob"] = y_prob

    low_max = engine.thresholds.get("low_risk_max", 0.35)
    mod_max = engine.thresholds.get("moderate_risk_max", 0.65)

    def classify_status(p):
        if p <= low_max:
            return "Approved"
        elif p <= mod_max:
            return "Manual Review"
        else:
            return "Declined"

    test_df["decision"] = test_df["pred_prob"].apply(classify_status)

    # Baseline counts and dollar volumes
    baseline_approved = test_df[test_df["decision"] == "Approved"]
    baseline_review = test_df[test_df["decision"] == "Manual Review"]
    baseline_declined = test_df[test_df["decision"] == "Declined"]

    total_requested_capital = test_df["loan_amount_usd"].sum()
    baseline_approved_capital = baseline_approved["loan_amount_usd"].sum()
    baseline_declined_capital = baseline_declined["loan_amount_usd"].sum()

    print(f"--- BASELINE UNDERWRITING SUMMARY (No Restructuring) ---")
    print(f"Total Applicants Evaluated:  {n_test:,}")
    print(f"Total Capital Requested:     ${total_requested_capital:,.2f}")
    print(f"  - Auto-Approved:           {len(baseline_approved):,} ({len(baseline_approved)/n_test:.1%}) | ${baseline_approved_capital:,.2f}")
    print(f"  - Manual Review:           {len(baseline_review):,} ({len(baseline_review)/n_test:.1%})")
    print(f"  - Auto-Declined:           {len(baseline_declined):,} ({len(baseline_declined)/n_test:.1%}) | ${baseline_declined_capital:,.2f}")

    # 4. Simulate AI Risk Advisor Restructuring
    print("\nSimulating AI Advisor Restructuring on at-risk borrowers...")
    at_risk_df = test_df[test_df["decision"].isin(["Manual Review", "Declined"])].copy()

    rescued_count = 0
    rescued_capital = 0.0
    rescue_details = []

    for idx, row in at_risk_df.iterrows():
        app_dict = row[engine.feature_cols].to_dict()
        recs = engine.advisor.generate_recommendations(app_dict)

        # Check if any recommendation successfully converts applicant to Approved (P <= 0.35)
        approved_recs = [r for r in recs if r.projected_decision == "Approved"]
        if approved_recs:
            best_rec = min(approved_recs, key=lambda x: x.projected_default_prob)
            rescued_count += 1
            rescued_capital += best_rec.recommended_loan_amount
            rescue_details.append({
                "original_loan": row["loan_amount_usd"],
                "new_loan": best_rec.recommended_loan_amount,
                "strategy": best_rec.strategy,
                "old_prob": row["pred_prob"],
                "new_prob": best_rec.projected_default_prob
            })

    # 5. Financial & Business Impact Analysis
    new_approved_count = len(baseline_approved) + rescued_count
    new_approval_rate = new_approved_count / n_test
    incremental_growth_pct = (rescued_count / len(baseline_approved)) * 100

    print("\n" + "=" * 70)
    print(" [METRICS] FINANCIAL IMPACT & VALUE DELIVERED")
    print("=" * 70)
    print(f"Initial Approval Rate:                 {len(baseline_approved)/n_test:.1%} ({len(baseline_approved):,} borrowers)")
    print(f"Borrowers Rescued by AI Advisor:       {rescued_count:,} borrowers")
    print(f"New Effective Approval Rate:           {new_approval_rate:.1%} ({new_approved_count:,} borrowers)")
    print(f"Origination Volume Expansion:          +{incremental_growth_pct:.1f}% incremental customer acquisition")
    print(f"Safe Lending Capital Unlocked:         ${rescued_capital:,.2f}")
    print(f"Toxic Debt Blocked & Averted:          ${baseline_declined_capital - rescued_capital:,.2f}")

    # 6. Save Report Artifacts
    results_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts")
    report_json_path = os.path.join(results_dir, "portfolio_simulation_results.json")

    results_data = {
        "test_cohort_size": n_test,
        "total_requested_capital": float(total_requested_capital),
        "baseline": {
            "approved_count": int(len(baseline_approved)),
            "approved_rate": float(len(baseline_approved) / n_test),
            "approved_capital": float(baseline_approved_capital),
            "review_count": int(len(baseline_review)),
            "declined_count": int(len(baseline_declined)),
            "declined_capital": float(baseline_declined_capital)
        },
        "advisor_impact": {
            "rescued_borrowers": int(rescued_count),
            "rescued_capital_unlocked": float(rescued_capital),
            "new_total_approved": int(new_approved_count),
            "new_approval_rate": float(new_approval_rate),
            "origination_lift_pct": float(incremental_growth_pct)
        }
    }

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=4)
    print(f"\n[OK] Saved metrics JSON to: {report_json_path}")

    # 7. Generate Executive Visual Chart
    plt.figure(figsize=(10, 5))

    # Bar comparison
    categories = ["Standard Approval", "With AI Advisor Restructuring"]
    counts = [len(baseline_approved), new_approved_count]
    colors = ["#4285F4", "#34A853"]

    bars = plt.bar(categories, counts, color=colors, width=0.5)
    plt.title("Loan Approval Expansion: Baseline vs. AI Risk Advisor", fontsize=14, fontweight="bold", pad=15)
    plt.ylabel("Number of Approved Borrowers", fontsize=11)
    plt.ylim(0, max(counts) * 1.25)

    for bar, count in zip(bars, counts):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 50, f"{count:,} ({count/n_test:.1%})",
                 ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()

    chart_path = os.path.join(results_dir, "portfolio_impact_chart.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"[OK] Saved visual chart to: {chart_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_portfolio_simulation()
