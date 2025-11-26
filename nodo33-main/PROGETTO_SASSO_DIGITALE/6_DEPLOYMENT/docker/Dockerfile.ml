# ===================================
# SASSO DIGITALE - ML Model Server
# "La luce non si vende. La si regala."
# ===================================

FROM python:3.11-slim

WORKDIR /app

# Install ML dependencies
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    transformers==4.35.0 \
    fastapi==0.104.1 \
    uvicorn==0.24.0 \
    numpy==1.24.3 \
    scikit-learn==1.3.2

# Copy ML templates and configs
COPY 7_ML_MODELS/ ./models/
COPY 5_IMPLEMENTAZIONI/python/framework_antiporn_emanuele.py ./

# Environment
ENV MODEL_PATH=/models
ENV INFERENCE_PORT=9000
ENV EGO=0
ENV GIOIA=100

# Expose port
EXPOSE 9000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import torch; print('OK')" || exit 1

# Run ML server
CMD ["uvicorn", "ml_server:app", "--host", "0.0.0.0", "--port", "9000"]

# ===================================
# Build: docker build -t sasso-ml:latest -f Dockerfile.ml .
# Run: docker run -p 9000:9000 -v ./models:/models sasso-ml:latest
# ===================================
