# syntax=docker/dockerfile:1

FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE.txt ./
COPY src ./src
RUN pip install --upgrade pip && \
    pip install ".[server,qdrant,mongo,grobid,faiss,providers]"

EXPOSE 8000

CMD ["sh", "-c", "uvicorn epilens.api:app --host ${EPILENS_API_HOST:-0.0.0.0} --port ${EPILENS_API_PORT:-8000}"]
