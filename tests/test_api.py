"""Integrateion tests for FastAPI endpoints."""
from fastapi.testclient import TestClient 
from api import app 

client = TestClient(app)

def test_root_endpoint():
    """Verify root endpoint returns welcome payload."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "docs_url" in data
    assert data["docs_url"] == "/docs"


def test_health_endpoint():
    """Verify health status, model version, and benchmarks."""
    response = client.get("/health")
    assert response.status_code ==200 

    data = response.json()
    assert data['status'] == 'healthy'
    assert 'model_version' in data 
    assert 'benchmarks' in data 
    assert 'thresholds' in data 

def test_asses_prime_applicant_api(prime_applicant):
    """Prime applicant payload sent via HTTP POST must 
    return Approved."""
    payload  = prime_applicant.model_dump()
    response = client.post("/api/v1/assess", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data['decision'] == 'Approved'
    assert data["risk_tier"] == "Low Risk"
    assert data["default_probability"] <= 0.35
    assert len(data["key_drivers"]) == 5

def test_assess_toxic_applicant_api(toxic_applicant):
    """Toxic applicant payload sent via HTTP POST must return 
    Declined and restructuring advice."""
    payload = toxic_applicant.model_dump()
    response = client.post("/api/v1/assess", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["decision"] == "Declined"
    assert data["risk_tier"] == "High Risk"
    assert data["restructuring_options"] is not None
    assert len(data["restructuring_options"]) > 0

def test_assess_invalid_payload_returns_422():
    """Malformed request (e.g. invalid credit score) must return HTTP 422 Unprocessable Entity."""
    bad_payload = {
        "age": 30,
        "monthly_income_usd": 5000.0,
        "credit_score": 999  # Invalid credit score (>850)
    }
    response = client.post("/api/v1/assess", json=bad_payload)
    assert response.status_code == 422