"""Core Credit Risk Inference and Scoring Engine."""
import os
import json
from typing import Dict, Any, List
import joblib
import numpy as np
import pandas as pd

from src.schemas import ApplicantInput, FeatureContribution, RiskAssessmentResponse
from src.advisor import RiskAdvisor, calculate_emi


class RiskEngine:
    """Production scoring engine for personal finance credit risk assessment."""

    _instance = None

    def __init__(self, artifacts_dir: str = None):
        if artifacts_dir is None:
            # Default to artifacts directory relative to project root
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            artifacts_dir = os.path.join(base_dir, "artifacts")

        self.artifacts_dir = artifacts_dir
        self._load_artifacts()
        self.advisor = RiskAdvisor(self.pipeline, self.config)

    def _load_artifacts(self):
        """Load serialized pipeline, SHAP explainer, and underwriting config."""
        pipeline_path = os.path.join(self.artifacts_dir, "risk_pipeline.joblib")
        explainer_path = os.path.join(self.artifacts_dir, "shap_explainer.joblib")
        config_path = os.path.join(self.artifacts_dir, "config.json")

        if not os.path.exists(pipeline_path):
            raise FileNotFoundError(f"Missing pipeline artifact at: {pipeline_path}")

        self.pipeline = joblib.load(pipeline_path)
        self.explainer_data = joblib.load(explainer_path)
        self.explainer = self.explainer_data["explainer"]
        self.feature_names_out = self.explainer_data["feature_names"]
        self.expected_value = self.explainer_data.get(
            "expected_value", 
            self.explainer_data.get("excepted_value", getattr(self.explainer, "expected_value", 0.0))
        )

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.feature_cols = self.config["feature_cols"]
        self.thresholds = self.config["thresholds"]

    def _format_interpretation(self, feat_name: str, val: Any, shap_val: float) -> str:
        """Translate raw SHAP attribution into plain-English financial insight."""
        direction = "Increases Risk" if shap_val > 0 else "Reduces Risk"
        clean_name = feat_name.replace("num__", "").replace("cat__", "")

        if "loan_amount" in clean_name:
            return f"Loan principal of ${float(val):,.0f} {direction.lower()} (+{shap_val:.2f} log-odds)."
        elif "loan_term" in clean_name:
            if float(val) <= 24:
                return f"Short tenure of {int(val)} months compresses payment into high monthly burden."
            else:
                return f"Tenure of {int(val)} months spreads payments comfortably, lowering default risk."
        elif "credit_score" in clean_name:
            if float(val) >= 700:
                return f"Strong credit score of {int(val)} demonstrates established repayment reliability."
            else:
                return f"Lower credit score of {int(val)} flags elevated historical credit risk."
        elif "monthly_income" in clean_name:
            return f"Monthly income of ${float(val):,.0f} acts as primary financial anchor."
        elif "loan_interest_rate" in clean_name:
            return f"Interest rate of {float(val):.2f}% affects overall debt carrying cost."
        elif "monthly_expenses" in clean_name:
            return f"Living expenses of ${float(val):,.0f}/mo impact monthly cash flow margin."
        elif "savings" in clean_name:
            return f"Liquid savings of ${float(val):,.0f} provide cash buffer against shocks."
        else:
            return f"{clean_name} = {val} ({direction.lower()})."

    def assess_applicant(self, applicant: ApplicantInput) -> RiskAssessmentResponse:
        """Score an applicant, compute SHAP attributions, and generate restructuring advice."""
        # Convert Pydantic model to dictionary and DataFrame
        applicant_dict = applicant.model_dump() if hasattr(applicant, "model_dump") else applicant.dict()
        df_input = pd.DataFrame([applicant_dict])[self.feature_cols]

        # 1. Model Inference
        prob = float(self.pipeline.predict_proba(df_input)[0, 1])

        # 2. Risk Tier & Decision
        low_max = self.thresholds.get("low_risk_max", 0.35)
        mod_max = self.thresholds.get("moderate_risk_max", 0.65)

        if prob <= low_max:
            risk_tier = "Low Risk"
            decision = "Approved"
        elif prob <= mod_max:
            risk_tier = "Moderate Risk"
            decision = "Manual Review"
        else:
            risk_tier = "High Risk"
            decision = "Declined"

        # Risk score on standard 300 - 850 credit-style scale (higher = safer)
        risk_score = int(850 - (prob * 550))

        # 3. Financial Metrics
        emi = calculate_emi(
            applicant.loan_amount_usd,
            applicant.loan_term_months,
            applicant.loan_interest_rate_pct
        )
        dti = round(emi / applicant.monthly_income_usd, 4)
        disposable = round(applicant.monthly_income_usd - applicant.monthly_expenses_usd - emi, 2)

        # 4. SHAP Local Attribution
        preprocessor = self.pipeline.named_steps["preprocessor"]
        X_trans = preprocessor.transform(df_input)
        shap_vals = self.explainer.shap_values(X_trans)[0]

        # Extract top 5 most impactful features by absolute SHAP value
        feat_order = np.argsort(np.abs(shap_vals))[::-1]
        top_indices = feat_order[:5]

        # Map back to raw transformed values
        raw_vals = X_trans.toarray()[0] if hasattr(X_trans, "toarray") else X_trans[0]

        key_drivers = []
        for idx in top_indices:
            feat_name = self.feature_names_out[idx]
            s_val = float(shap_vals[idx])
            val = raw_vals[idx]
            direction = "Increases Risk" if s_val > 0 else "Reduces Risk"
            interp = self._format_interpretation(feat_name, val, s_val)

            clean_feat = feat_name.replace("num__", "").replace("cat__", "")
            key_drivers.append(
                FeatureContribution(
                    feature=clean_feat,
                    feature_value=str(round(val, 2) if isinstance(val, (int, float)) else val),
                    shap_value=round(s_val, 4),
                    direction=direction,
                    interpretation=interp
                )
            )

        # 5. Restructuring Advice (if not auto-approved)
        restructuring = None
        if decision != "Approved":
            restructuring = self.advisor.generate_recommendations(applicant_dict)

        return RiskAssessmentResponse(
            default_probability=round(prob, 4),
            risk_score=risk_score,
            risk_tier=risk_tier,
            decision=decision,
            estimated_monthly_emi=emi,
            debt_to_income_ratio=dti,
            disposable_income_usd=disposable,
            key_drivers=key_drivers,
            restructuring_options=restructuring
        )
