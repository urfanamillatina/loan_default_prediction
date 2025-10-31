#Builder
FROM python:3.12-slim AS builder

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --prefix=/install -r requirements.txt


# Runtime
FROM python:3.12-slim

# Create non-root user
RUN useradd -m appuser
USER appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy source code
COPY predict_api ./predict_api
COPY exported_model ./exported_model

# Expose FastAPI port
EXPOSE 9000

# Run the API
CMD ["uvicorn", "predict_api.app:app", "--host", "0.0.0.0", "--port", "9000"]
