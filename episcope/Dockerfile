# syntax=docker/dockerfile:1

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for unstructured and PDF processing
RUN apt-get update && apt-get install -y \
    build-essential \
    libpoppler-cpp-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ .

EXPOSE 8000

CMD ["uvicorn", "episcope.api:app", "--host", "0.0.0.0", "--port", "8000"]
