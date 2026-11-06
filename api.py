"""FastAPI Production REST Service for Credit Risk Assessment and Advisory."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.schemas import ApplicantInput, RiskAssessmentResponse
from src.engine import RiskEngine

app = FastAPI(
    title="Personal Finance Credit Risk & Advisory API",
    description="Underwriting risk assessment, local SHAP explainability, and automated restructuring engine.",
    version="1.0.0"
)

# Enable CORS for frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy singleton engine instance
_engine: RiskEngine = None


def get_engine() -> RiskEngine:
    """Retrieve or initialize the RiskEngine singleton."""
    global _engine
    if _engine is None:
        _engine = RiskEngine()
    return _engine


@app.get("/health", tags=["Monitoring"])
def health_check():
    """Health and model metadata endpoint."""
    engine = get_engine()
    return {
        "status": "healthy",
        "model_version": engine.config.get("model_version", "1.0.0"),
        "benchmarks": engine.config.get("benchmarks", {}),
        "thresholds": engine.config.get("thresholds", {})
    }


@app.post("/api/v1/assess", response_model=RiskAssessmentResponse, tags=["Underwriting"])
def assess_applicant(applicant: ApplicantInput):
    """Underwrite an applicant, predict default probability, and return key drivers + advice."""
    try:
        engine = get_engine()
        return engine.assess_applicant(applicant)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
