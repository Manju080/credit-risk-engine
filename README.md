# AI Credit Risk Underwriting & Financial Advisor Engine

An end-to-end, production-grade credit risk modeling system powered by XGBoost, local SHAP explainability, and an automated "What-If" loan restructuring engine.

---

## 🏗️ System Architecture

```text
Personal finance/
├── artifacts/
│   ├── risk_pipeline.joblib          # Trained Scikit-Learn Pipeline (Preprocessor + XGBoost)
│   ├── shap_explainer.joblib         # Serialized SHAP TreeExplainer & feature metadata
│   └── config.json                   # Underwriting thresholds and performance benchmarks
├── src/
│   ├── __init__.py
│   ├── schemas.py                    # Pydantic data validation contracts (ApplicantInput, Responses)
│   ├── engine.py                     # Fast inference engine & SHAP local attribution mapper
│   └── advisor.py                    # Prescriptive 'What-If' restructuring simulator
├── notebooks/
│   ├── EDA.ipynb                     # Exploratory Data Analysis & anomaly discovery
│   ├── model.ipynb                   # Initial modeling experiments
│   ├── model_raw.ipynb               # Raw feature benchmarking
│   └── risk_analysis.ipynb           # Path B: Probabilistic default modeling & SHAP analysis
├── app.py                            # Streamlit Interactive Underwriting & Advisor Dashboard
├── api.py                            # FastAPI High-Performance REST Service (/assess, /health)
├── requirements.txt                  # Full environment dependencies
└── README.md
```

---

## 🚀 Quickstart

### 1. Launch the Interactive Dashboard (Streamlit)
```bash
streamlit run app.py
```
* **Live Underwriting**: Enter any applicant profile or pick one of the built-in scenario presets.
* **Risk Gauge**: Real-time Default Probability (%) and underwriting decision (`Approved`, `Manual Review`, `Declined`).
* **SHAP Explainability**: Instant breakdown showing which financial factors are increasing vs. reducing risk.
* **Automated Restructuring**: If an applicant is declined, the advisor calculates the exact restructured tenure and principal needed to get them approved.

### 2. Launch the REST API (FastAPI)
```bash
uvicorn api:app --reload --port 8000
```
* Swagger UI Docs: `http://localhost:8000/docs`
* Health Check: `http://localhost:8000/health`
* Underwriting Endpoint: `POST /api/v1/assess`

---

## 📊 Underwriting Policy & Thresholds

| Risk Tier | Default Probability | Underwriting Action | Advisor Trigger |
| :--- | :---: | :---: | :--- |
| **Low Risk** | $\le 35\%$ | **Auto-Approved** | No restructuring required |
| **Moderate Risk** | $35\% - 65\%$ | **Manual Review** | Advisory restructuring available |
| **High Risk** | $> 65\%$ | **Declined** | Automatic restructuring initiated |

---

## 🎯 Model Benchmarks (Out-of-Time Test Set)
* **ROC-AUC**: `0.8979`
* **PR-AUC**: `0.9695`
* **Accuracy**: `82%` (Balanced 82% recall on both Defaults and Repaid borrowers)

---

## 🧪 Automated Testing Suite

Comprehensive unit and integration tests covering Pydantic validation schemas, inference thresholds, local SHAP attribution values, and REST API endpoints:

```bash
pytest -v
```

* **Test Coverage**:
  * `tests/test_schemas.py`: Boundary verification (credit scores [300, 850], negative incomes, field validations).
  * `tests/test_engine.py`: Single-borrower inference, SHAP driver verification, restructuring triggers.
  * `tests/test_api.py`: FastAPI endpoint responses, HTTP 200 health checks, HTTP 422 validation guards.
  * **Result**: `14 passed in 11.95s` (100% test pass rate).

---

## 🐳 Docker Containerization & Microservices

Run the complete multi-container production suite with zero environment friction:

```bash
# Build and run API and Streamlit dashboard in detached mode
docker-compose up -d --build

# View container status and health
docker ps
```

* **FastAPI Service**: Runs on `http://localhost:8000` (Docker healthcheck enabled via `/health`).
* **Streamlit Dashboard**: Runs on `http://localhost:8501`.

---

## 🔄 CI/CD Pipeline (GitHub Actions)

Continuous Integration is pre-configured via `.github/workflows/ci.yml`:
* Triggers automatically on `push` and `pull_request` to `main` and `master`.
* Sets up Python 3.11 with pip caching.
* Installs dependencies and runs the entire automated test suite (`pytest -v`) before deployment.

---

## 📈 Portfolio Value & Stress-Testing Simulation

Simulated across **2,594 out-of-time borrowers** ($642.18M total requested loan capital) in `scripts/portfolio_simulation.py`:

| Metric | Baseline Policy | With AI Restructuring Advisor | Business Impact |
| :--- | :---: | :---: | :--- |
| **Approved Borrowers** | 588 (22.7%) | **2,478 (95.5%)** | **+1,890 borrowers rescued** |
| **Origination Volume** | Baseline | **+321.4% Lift** | Massive customer acquisition expansion |
| **Safe Capital Unlocked** | $45.46M | **+$161.48M** | Originated at safe, restructured terms ($P_{default} \le 35\%$) |
| **Toxic Debt Averted** | — | **$338.49M Blocked** | High-risk unsustainable debt safely filtered out |

* Simulation script: `python scripts/portfolio_simulation.py`
* Saved metrics: `artifacts/portfolio_simulation_results.json`
* Executive visual chart: `artifacts/portfolio_impact_chart.png`


