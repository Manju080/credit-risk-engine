"""Unit tests for the RiskEngine, SHAP attributions, and RiskAdvisor."""
import pytest


def test_toxic_applicant_is_declinded(engine, toxic_applicant):
    """An applicant with posssible DTI ($408k loan on $4k salary) must 
    be Declined."""

    result = engine.assess_applicant(toxic_applicant)

    assert result.decision == "Declined"
    assert result.risk_tier == "High Risk"
    assert result.default_probability >0.65
    assert result.debt_to_income_ratio >1.0

def test_prime_applicant_is_approved(engine, prime_applicant):
    """A prime borrower ($8.5k income, 790 credit score, $20k loan) 
    must be Approved."""
    result = engine.assess_applicant(prime_applicant)
    assert result.decision == "Approved"
    assert result.risk_tier == "Low Risk"
    assert result.default_probability <= 0.35
    assert result.restructuring_options is None  # Approved applicants don't need restructuring

def test_shap_local_attributions(engine, toxic_applicant):
    """Verify that SHAP outputs 5 key drivers with correct 
    directional risk"""

    result = engine.assess_applicant(toxic_applicant)
    assert len(result.key_drivers) ==5 
    driver_names = [d.feature for d in result.key_drivers]

    assert "loan_amount_usd" in driver_names
    for driver in result.key_drivers:
        assert driver.direction in ["Increases Risk", "Reduces Risk"]
        assert len(driver.interpretation) > 0

def test_advisor_restructures_toxic_applicant(engine, toxic_applicant):
    """The advisor must produce restructuring options that lower the 
    default probability."""

    result = engine.assess_applicant(toxic_applicant)

    assert result.restructuring_options is not None
    assert len(result.restructuring_options)>0

    best_opt = min(result.restructuring_options, key=lambda opt: opt.projected_default_prob)
    assert best_opt.projected_default_prob < result.default_probability
    assert best_opt.projected_decision in ["Approved", "Manual Review"]


def test_dti_knockout_rule(engine, prime_applicant):
    """An applicant with otherwise good credit but DTI > 0.55 must be Declined by policy knockout."""
    from src.schemas import ApplicantInput
    data = prime_applicant.model_dump()
    data["loan_amount_usd"] = 120000.0
    data["loan_term_months"] = 24  # Results in EMI > $5,000 -> DTI > 0.55
    high_dti_applicant = ApplicantInput(**data)

    result = engine.assess_applicant(high_dti_applicant)
    assert result.debt_to_income_ratio > 0.55
    assert result.decision == "Declined"
    assert result.risk_tier == "High Risk"

    