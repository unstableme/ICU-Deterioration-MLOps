FROM python:3.11-slim

WORKDIR /app


COPY backend/requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cpu


# Copying src folder which is used inside model-training code
COPY src ./src

COPY backend/app ./app

# Add /app to PYTHONPATH so Python can find 'src' and 'app' packages
ENV PYTHONPATH=/app:$PYTHONPATH

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]