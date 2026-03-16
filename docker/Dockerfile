# =============================================================================
# docker/Dockerfile — Containerised prediction service
# =============================================================================
# Builds a minimal image for the FastAPI prediction service.
# The model is NOT baked in — it is loaded at runtime from a
# volume-mounted MLflow tracking store (mlruns/).
# =============================================================================

FROM python:3.12-slim

# System dependency required by scikit-learn (OpenMP).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security.
RUN useradd --create-home appuser

WORKDIR /app

# Copy dependency manifest first for layer caching.
COPY pyproject.toml .

# Copy source code.
COPY src/ src/

# Install the project and its pinned dependencies.
RUN pip install --no-cache-dir -e .

# Switch to non-root user.
USER appuser

# Default API port.
EXPOSE 8000

# Health check — start period allows time for model loading.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Start the prediction API via uvicorn.
CMD ["uvicorn", "src.deployment.app:app", "--host", "0.0.0.0", "--port", "8000"]
