# Use official lightweight Python runtime 
FROM python:3.11-slim 

# Prevent Python from writing .pyc files and buffer stdout/stderr 
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app 

# Install system dependencies (build-essential needed for lightgbm/xgboost if compiling)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies 
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt 

# Copy production source code and pre-trained model artifacts
COPY artifacts/ ./artifacts/
COPY src/ ./src/
COPY api.py app.py pytest.ini ./

EXPOSE 8000 8501 

ENV SERVICE_TYPE=api

CMD ["sh", "-c", "if [ \"$SERVICE_TYPE\" = \"dashboard\" ]; then exec streamlit run app.py --server.port ${PORT:-8501} --server.address 0.0.0.0 --server.enableCORS false --server.enableXsrfProtection false; else exec uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}; fi"]
