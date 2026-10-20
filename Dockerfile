# Dockerfile
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Run as a non-root user (uid 1000 is also what Hugging Face Spaces expects)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR /app

# Copy requirements first for better caching.
# The CPU-only torch index keeps the image ~2GB smaller than the default CUDA build.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# Copy application code
COPY --chown=user . .

EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# One worker so the model is only loaded into memory once; threads handle concurrent requests
CMD gunicorn --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:${PORT} "app:create_app()"
