# FastAPI service — calls vLLM, parses outputs, returns structured predictions.
# Lightweight: no torch/vllm inside this image.
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps — only what's needed for FastAPI + openai + pandas
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy just packaging metadata first for better layer caching.
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

RUN pip install --upgrade pip && \
    pip install ".[serve]"

# Non-root user
RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app
USER app

EXPOSE 8001

# Defaults — override via environment at compose/run time.
ENV VLLM_BASE_URL=http://vllm:8000/v1 \
    MODEL_NAME=football-llm \
    MAX_OUTPUT_TOKENS=300 \
    TEMPERATURE=0.1 \
    TOP_P=0.9 \
    HOST=0.0.0.0 \
    PORT=8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request, sys; \
      r = urllib.request.urlopen('http://localhost:8001/health', timeout=3); \
      sys.exit(0 if r.status == 200 else 1)"

CMD ["uvicorn", "football_llm.serving.api:app", "--host", "0.0.0.0", "--port", "8001", "--log-level", "info"]
