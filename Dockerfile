# Stage 1: Build Stage
FROM python:3.11-slim AS builder

# Set build-time environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies to a local directory
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Production Stage
FROM python:3.11-slim

# Set runtime environment variables
# HF Spaces requires port 7860
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

WORKDIR /app

# Create a non-root user for security
RUN groupadd -r streamlit && useradd -r -g streamlit streamlit

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Ensure data directories exist with proper permissions
RUN mkdir -p /app/data/uploads && \
    chown -R streamlit:streamlit /app

# Switch to the non-root user
USER streamlit

# HF Spaces expects port 7860
EXPOSE 7860

# Run the Streamlit application
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
