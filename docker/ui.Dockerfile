# Gradio UI — thin wrapper around the FastAPI /predict endpoint.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

RUN pip install --upgrade pip && pip install ".[serve]"

RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app
USER app

EXPOSE 7860

ENV API_URL=http://api:8001 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

CMD ["python", "-m", "football_llm.serving.demo_ui"]
