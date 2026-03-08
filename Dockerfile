# CineMatch — Multi-stage Dockerfile
# Supports both Streamlit frontend and FastAPI backend

# ── Stage 1: Builder ──
FROM python:3.11-slim AS builder

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Stage 2: Runtime ──
FROM python:3.11-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN useradd --create-home --shell /bin/bash appuser

# Copy code
COPY --chown=appuser:appuser config/ ./config/
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser api/ ./api/
COPY --chown=appuser:appuser frontend/ ./frontend/
COPY --chown=appuser:appuser analytics/ ./analytics/

# Copy ALL data (including MovieLens + Kaggle)
# Create data directories with write permissions for appuser
RUN mkdir -p data/raw/kaggle_bo data/processed data/cache && \
    chown -R appuser:appuser data/processed data/cache
COPY --chown=appuser:appuser data/raw/ ./data/raw/

# Create catboost_info directory with write permissions for CatBoost training
RUN mkdir -p /app/catboost_info && chown -R appuser:appuser /app/catboost_info

USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    ENVIRONMENT=production \
    STREAMLIT_SERVER_PORT=${PORT:-8501} \
    CATBOOST_TRAIN_DIR=/app/catboost_info

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || curl -f http://localhost:8000/health || exit 1

# 唯一的启动命令
CMD streamlit run frontend/app.py --server.port ${PORT:-8501} --server.address 0.0.0.0 --server.headless true --server.enableCORS false --server.enableXsrfProtection false