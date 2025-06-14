# Use Python 3.11 slim image as base
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies (minimal for API service)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN pip install uv

# Create app user and directory
RUN groupadd -r appuser && useradd -r -g appuser appuser
WORKDIR /app

# Copy dependency files
COPY pyproject.toml ./
COPY uv.lock* ./

# Install Python dependencies using uv
RUN uv sync --frozen --no-dev

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/

# Create temp directory for processing files
RUN mkdir -p /tmp/media-to-text && chown -R appuser:appuser /tmp/media-to-text

# Change ownership of app directory
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

# Run the application
CMD ["uv", "run", "uvicorn", "src.media_to_text.main:app", "--host", "0.0.0.0", "--port", "8000"]