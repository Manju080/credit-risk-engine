"""Actionable 'What-If' Restructuring Simulator and Advisory Engine."""
from typing import List, Dict, Any
import pandas as pd
from src.schemas import RestructuringRecommendation


def calculate_emi(principal: float, term_months: int, interest_rate_pct: float) -> float:
    """Calculate monthly Equated Monthly Installment (EMI)."""
    if term_months <= 0:
        return 0.0
    r = (interest_rate_pct / 100.0) / 12.0
    if r > 0:
        emi = principal * (r * ((1.0 + r) ** term_months)) / (((1.0 + r) ** term_months) - 1.0)
    else:
        emi = principal / term_months
    return round(emi, 2)


class RiskAdvisor:
    """Simulates loan restructuring scenarios to convert declined applicants into approved ones."""

    def __init__(self, pipeline, config: Dict[str, Any]):
        self.pipeline = pipeline
        self.config = config
        self.feature_cols = config.get("feature_cols", [])
        self.low_threshold = config.get("thresholds", {}).get("low_risk_max", 0.35)

    def generate_recommendations(self, applicant_dict: Dict[str, Any]) -> List[RestructuringRecommendation]:
        """Generate prescriptive restructuring paths if applicant exceeds safe risk threshold."""
        recommendations = []
        orig_p = float(applicant_dict["loan_amount_usd"])
        orig_term = int(applicant_dict["loan_term_months"])
        orig_rate = float(applicant_dict["loan_interest_rate_pct"])
        monthly_income = float(applicant_dict["monthly_income_usd"])

        # Strategy 1: Tenure Extension (keep principal, extend term to reduce EMI strain)
        extended_terms = [36, 48, 60, 72, 84]
        candidate_terms = [t for t in extended_terms if t > orig_term]
        for new_term in candidate_terms:
            cand = applicant_dict.copy()
            cand["loan_term_months"] = new_term
            df_cand = pd.DataFrame([cand])[self.feature_cols]
            prob = float(self.pipeline.predict_proba(df_cand)[0, 1])
            new_emi = calculate_emi(orig_p, new_term, orig_rate)

            if prob <= self.low_threshold or new_term == candidate_terms[-1]:
                dec = "Approved" if prob <= self.low_threshold else ("Manual Review" if prob <= 0.65 else "Declined")
                recommendations.append(
                    RestructuringRecommendation(
                        strategy="Tenure Extension",
                        description=f"Extend loan tenure from {orig_term} to {new_term} months to reduce monthly EMI payment strain.",
                        original_loan_amount=orig_p,
                        recommended_loan_amount=orig_p,
                        original_term_months=orig_term,
                        recommended_term_months=new_term,
                        estimated_monthly_emi=new_emi,
                        projected_default_prob=round(prob, 4),
                        projected_decision=dec
                    )
                )
                break

        # Strategy 2: Principal Right-Sizing (downsize principal to fit income)
        # Safe underwriting limit: loan amount roughly 3x to 4x annual income
        safe_principal_targets = [
            round(monthly_income * 36, 2),  # 3 yrs income
            round(monthly_income * 24, 2),  # 2 yrs income
            round(monthly_income * 12, 2)   # 1 yr income
        ]
        for target_p in safe_principal_targets:
            if target_p < orig_p:
                cand = applicant_dict.copy()
                cand["loan_amount_usd"] = target_p
                df_cand = pd.DataFrame([cand])[self.feature_cols]
                prob = float(self.pipeline.predict_proba(df_cand)[0, 1])
                new_emi = calculate_emi(target_p, orig_term, orig_rate)

                if prob <= self.low_threshold or target_p == safe_principal_targets[-1]:
                    dec = "Approved" if prob <= self.low_threshold else ("Manual Review" if prob <= 0.65 else "Declined")
                    recommendations.append(
                        RestructuringRecommendation(
                            strategy="Principal Right-Sizing",
                            description=f"Downsize loan amount from ${orig_p:,.0f} to ${target_p:,.0f} to align with debt-to-income limits.",
                            original_loan_amount=orig_p,
                            recommended_loan_amount=target_p,
                            original_term_months=orig_term,
                            recommended_term_months=orig_term,
                            estimated_monthly_emi=new_emi,
                            projected_default_prob=round(prob, 4),
                            projected_decision=dec
                        )
                    )
                    break

        # Strategy 3: Balanced Restructuring (combine moderate principal reduction + realistic tenure)
        balanced_term = 60 if orig_term < 60 else orig_term
        balanced_p = min(orig_p * 0.70, monthly_income * 36)
        if balanced_p < orig_p or balanced_term != orig_term:
            cand = applicant_dict.copy()
            cand["loan_amount_usd"] = balanced_p
            cand["loan_term_months"] = balanced_term
            df_cand = pd.DataFrame([cand])[self.feature_cols]
            prob = float(self.pipeline.predict_proba(df_cand)[0, 1])
            new_emi = calculate_emi(balanced_p, balanced_term, orig_rate)
            dec = "Approved" if prob <= self.low_threshold else ("Manual Review" if prob <= 0.65 else "Declined")

            recommendations.append(
                RestructuringRecommendation(
                    strategy="Balanced Restructuring",
                    description=f"Combine sensible tenure ({balanced_term} mos) with proportional principal (${balanced_p:,.0f}).",
                    original_loan_amount=orig_p,
                    recommended_loan_amount=balanced_p,
                    original_term_months=orig_term,
                    recommended_term_months=balanced_term,
                    estimated_monthly_emi=new_emi,
                    projected_default_prob=round(prob, 4),
                    projected_decision=dec
                )
            )

        return recommendations

