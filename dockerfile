FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY dvc.yaml .
COPY params.yaml .
COPY .dvc/ ./.dvc/ 

# CMD [] #not needed as we are using airflows DockerOperator to run specific commands which overrides this 
