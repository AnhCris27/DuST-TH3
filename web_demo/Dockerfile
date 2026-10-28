# Multi-Modal Driving Risk AI (DUST) Deployment Container
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python requirements
COPY web_demo/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy models code
COPY models /app/models

# Ensure saved_models directory exists (even if weights are git-ignored)
RUN mkdir -p /app/saved_models

# Copy web application
COPY web_demo /app/web_demo

WORKDIR /app/web_demo

EXPOSE 8000 10000

ENV PYTHONUNBUFFERED=1

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
