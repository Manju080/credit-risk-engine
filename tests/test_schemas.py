"""Tests for input validation schemas."""
import pytest 
from src.schemas import ApplicantInput
from pydantic import ValidationError

def test_valid_applicant_schema(prime_applicant):
    """Test tahat a well-formed applicant passes validation.""" 
    assert prime_applicant.credit_score == 790 
    assert prime_applicant.monthly_income_usd == 8500.0


@pytest.mark.parametrize("invalid_credit_score", [250, 900, -10])

def test_invalid_credit_score_raises_error(invalid_credit_score, prime_applicant):
    """Credit score outside [300,850] must fail validation"""
    data = prime_applicant.model_dump()
    data['credit_score'] = invalid_credit_score 

    with pytest.raises(ValidationError):
        ApplicantInput(**data)

@pytest.mark.parametrize("invalid_income", [0, -500.0])
def test_zero_or_negative_income_raises_error(invalid_income, prime_applicant):
    """Monthly income must be strictly greateer than zero"""
    data = prime_applicant.model_dump()
    data['monthly_income_usd'] = invalid_income

    with pytest.raises(ValidationError):
        ApplicantInput(**data)
