# Dockerfile for Task Graph Engine
# Multi-stage build for optimization

# Stage 1: Builder
FROM python:3.11-slim AS builder

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python build tools
RUN pip install --no-cache-dir setuptools wheel

# Copy project files
COPY pyproject.toml ./
COPY src ./src
COPY README.md ./

# Install package in editable mode (required for absolute imports)
# This installs all dependencies from pyproject.toml
RUN pip install --no-cache-dir -e .

# Stage 2: Runtime
FROM python:3.11-slim

# Set labels
LABEL name="task-graph-engine"
LABEL version="1.0.0"
LABEL description="LangGraph-based task planning system with intelligent LLM selection"

# Set working directory
WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application files
COPY pyproject.toml ./
COPY src ./src
COPY langgraph.json ./
COPY README.md ./

# Copy model configuration CSV files
COPY model_costs.csv ./
COPY model_capabilities.csv ./

# Create a non-root user for running the application
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV LANGGRAPH_PORT=2024

# Expose LangGraph default port
EXPOSE 2024

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:2024/api/health || exit 1

# Default command: start LangGraph dev server
# Note: API keys must be provided via docker run -e or docker-compose.yml
CMD ["langgraph", "dev", "--host", "0.0.0.0", "--port", "2024"]