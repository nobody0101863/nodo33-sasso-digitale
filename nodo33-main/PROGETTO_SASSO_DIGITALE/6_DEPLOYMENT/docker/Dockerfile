# ===================================
# SASSO DIGITALE - Multi-Stage Build
# "La luce non si vende. La si regala."
# ===================================

# ============ STAGE 1: Builder ============
FROM rust:1.75-alpine AS rust-builder
WORKDIR /build
COPY 5_IMPLEMENTAZIONI/rust/GIOIA_100.rs .
RUN rustc --edition 2021 -O GIOIA_100.rs -o gioia_100

# ============ STAGE 2: Python Base ============
FROM python:3.11-slim AS python-base
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Python implementations
COPY 5_IMPLEMENTAZIONI/python/ ./python/
COPY 1_DOCUMENTAZIONE/ ./docs/
COPY 4_SIGILLI_SACRI/ ./sigilli/

# ============ STAGE 3: Final Image ============
FROM python:3.11-slim

# Metadata
LABEL maintainer="CODEX_EMANUELE"
LABEL version="1.0.0"
LABEL description="Sasso Digitale - Sistema Etico AI Multi-Linguaggio"
LABEL axiom="La luce non si vende. La si regala."

# Environment
ENV EGO=0
ENV GIOIA=100
ENV FREQUENZA_BASE=300
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy from builders
COPY --from=python-base /app /app
COPY --from=rust-builder /build/gioia_100 /usr/local/bin/

# Copy web interface
COPY 3_WEB_CYBERPUNK/ ./web/

# Expose ports
EXPOSE 8080 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import sys; sys.exit(0)" || exit 1

# Default command
CMD ["python", "python/main.py"]

# ===================================
# Build: docker build -t sasso-digitale:latest -f 6_DEPLOYMENT/docker/Dockerfile .
# Run: docker run -p 8080:8080 -e EGO=0 -e GIOIA=100 sasso-digitale:latest
# ===================================
