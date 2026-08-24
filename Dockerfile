FROM python:3.11-slim

WORKDIR /app

# Install system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and static assets
COPY controlplane /app/controlplane
COPY config /app/config
COPY dashboard /app/dashboard

# Create data directory for SQLite WAL logs
RUN mkdir -p /app/data

EXPOSE 8000

ENV PYTHONUNBUFFERED=1
ENV CONTROLPLANE_POLICY_PATH=config/policies.yaml
ENV CONTROLPLANE_DB_PATH=data/audit_logs.db

CMD ["uvicorn", "controlplane.main:app", "--host", "0.0.0.0", "--port", "8000"]
