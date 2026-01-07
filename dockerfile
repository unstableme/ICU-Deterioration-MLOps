FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first (important)
RUN pip install --upgrade pip

# Install CPU-only PyTorch explicitly
RUN pip install \
    torch==2.1.1+cpu \
    torchvision==0.16.1+cpu \
    torchaudio==2.1.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install the rest (lightweight libs)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY dvc.yaml .
COPY params.yaml .
COPY .dvc/ ./.dvc/

# No CMD — Airflow overrides it
