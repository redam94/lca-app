# Multi-stage build for smaller image
FROM python:3.12-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /build

# Upgrade pip first
RUN pip install --no-cache-dir --upgrade pip

# Copy lca_analysis package (from parent directory context)
COPY lca_analysis/ lca_analysis/

# Install lca_analysis package
RUN pip install --no-cache-dir ./lca_analysis

# Copy backend project (from api/ subdirectory in parent context)
COPY api/pyproject.toml .
COPY api/src/ src/

# Install the backend package
RUN pip install --no-cache-dir .

# Production image
FROM python:3.12-slim

# Install runtime dependencies including g++ for PyTensor compilation
RUN apt-get update && apt-get install -y \
    curl \
    libgomp1 \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

# Set working directory
WORKDIR /app

# Create data directories
RUN mkdir -p /app/data /app/model_results && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command - use the installed entry point
CMD ["backend-api"]