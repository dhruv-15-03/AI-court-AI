# Production container for AI Court API (multi-stage)

FROM python:3.12-slim AS builder

# Build arg to bust Render's layer cache when needed
ARG CACHE_BUST=1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VENV_PATH=/opt/venv
RUN python -m venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"
WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir gunicorn

# Install the project itself
COPY pyproject.toml ./
COPY src ./src
COPY README.md ./
RUN pip install --no-cache-dir .

FROM python:3.12-slim AS runtime
ARG APP_VERSION=0.1.0
ARG GIT_COMMIT=dev
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    APP_VERSION=${APP_VERSION} \
    GIT_COMMIT=${GIT_COMMIT} \
    APP_ENV=production \
    MODEL_PATH=/app/models/legal_case_classifier.pkl

WORKDIR /app

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY . .

# Create non-root user and set ownership
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 CMD curl -fsS http://127.0.0.1:8000/api/health || exit 1

# Run with gunicorn for production
CMD ["gunicorn", "-c", "gunicorn.conf.py", "src.ai_court.api.server:app"]
