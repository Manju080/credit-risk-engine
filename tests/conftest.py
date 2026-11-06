"""Pytest fixtures for credit risk assessment tests."""
import pytest 
import sys
import os 

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.schemas import ApplicantInput
from src.engine import RiskEngine 

@pytest.fixture(scope="session")
def engine():
    """Load RiskEngie once for the entire test session"""
    return RiskEngine()

@pytest.fixture 
def toxic_applicant():
    """Profile of a borrower taking an impossible debt burden (should be Declined)."""
    return ApplicantInput(age=62,
        gender="Male",
        education_level="Master",
        employment_status="Employed",
        job_title="Student",
        region="North America",
        monthly_income_usd=4051.0,
        monthly_expenses_usd=2244.42,
        savings_usd=194901.41,
        credit_score=718,
        loan_type="Car",
        loan_amount_usd=407915.25,
        loan_term_months=12,
        loan_interest_rate_pct=22.21
    )

@pytest.fixture 
def prime_applicant():
    """Profile of low-risk, prime borrower (should be Approved)."""
    return ApplicantInput(
        age=38,
        gender="Female",
        education_level="Master",
        employment_status="Employed",
        job_title="Engineer",
        region="North America",
        monthly_income_usd=8500.0,
        monthly_expenses_usd=2500.0,
        savings_usd=60000.0,
        credit_score=790,
        loan_type="Personal",
        loan_amount_usd=20000.0,
        loan_term_months=48,
        loan_interest_rate_pct=7.5
    )