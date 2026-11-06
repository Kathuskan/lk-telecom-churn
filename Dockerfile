FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required for C-extensions (SHAP / LightGBM / OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model artifact and FastAPI source code
COPY churn_model_bundle.pkl .
COPY app.py .

EXPOSE 8000

# Run FastAPI with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]