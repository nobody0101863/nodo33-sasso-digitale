FROM python:3.11-slim

LABEL maintainer="Nodo33 - Sasso Digitale"
LABEL description="Codex Server - Sacred API for Spiritual-Technical Systems"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY codex_server.py .
COPY p2p_node.py .
COPY llm_tool.py .
COPY sigillum.py .
COPY registry.yaml .
COPY domains.yaml .

# Copy source modules
COPY src/ ./src/
COPY custos/ ./custos/
COPY anti_porn_framework/ ./anti_porn_framework/

# Create directories
RUN mkdir -p generated_images

# Expose port
EXPOSE 8644

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8644/health || exit 1

# Run server
CMD ["python3", "codex_server.py", "--host", "0.0.0.0", "--port", "8644"]
