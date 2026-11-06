"""Pydantic data schemas for personal finance risk assessment."""
from typing import List, Optional
from pydantic import BaseModel, Field


class ApplicantInput(BaseModel):
    """Raw applicant details submitted for loan underwriting assessment."""
    age: int = Field(..., ge=18, le=100, description="Applicant age in years")
    gender: str = Field(default="Male", description="Gender (Male, Female, Other)")
    education_level: str = Field(default="Bachelor", description="High School, Bachelor, Master, Doctorate")
    employment_status: str = Field(default="Employed", description="Employed, Self-Employed, Unemployed")
    job_title: str = Field(default="Analyst", description="Current occupation / job title")
    region: str = Field(default="North America", description="Geographic region")
    monthly_income_usd: float = Field(..., gt=0, description="Gross monthly income in USD")
    monthly_expenses_usd: float = Field(..., ge=0, description="Total monthly living expenses in USD")
    savings_usd: float = Field(..., ge=0, description="Current total savings and liquid assets in USD")
    credit_score: int = Field(..., ge=300, le=850, description="FICO or equivalent credit score (300-850)")
    loan_type: str = Field(default="Personal", description="Type of loan (Personal, Auto, Car, Home, Student)")
    loan_amount_usd: float = Field(..., gt=0, description="Requested loan principal in USD")
    loan_term_months: int = Field(..., ge=1, le=360, description="Loan repayment tenure in months")
    loan_interest_rate_pct: float = Field(..., ge=0.0, le=100.0, description="Annual interest rate in %")


class FeatureContribution(BaseModel):
    """Local SHAP feature attribution explaining individual risk push/pull."""
    feature: str
    feature_value: str
    shap_value: float
    direction: str  # "Increases Risk" or "Reduces Risk"
    interpretation: str


class RestructuringRecommendation(BaseModel):
    """Actionable loan restructuring options to lower default probability."""
    strategy: str
    description: str
    original_loan_amount: float
    recommended_loan_amount: float
    original_term_months: int
    recommended_term_months: int
    estimated_monthly_emi: float
    projected_default_prob: float
    projected_decision: str


class RiskAssessmentResponse(BaseModel):
    """Complete underwriting and risk advisory response."""
    default_probability: float
    risk_score: int  # 0 to 1000 scale
    risk_tier: str   # "Low Risk", "Moderate Risk", "High Risk"
    decision: str    # "Approved", "Manual Review", "Declined"
    estimated_monthly_emi: float
    debt_to_income_ratio: float
    disposable_income_usd: float
    key_drivers: List[FeatureContribution]
    restructuring_options: Optional[List[RestructuringRecommendation]] = None

