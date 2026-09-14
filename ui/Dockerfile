FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN pip install --upgrade pip

COPY pyproject.toml README.md LICENSE.txt ./
COPY src ./src

RUN pip install ".[ui]"

EXPOSE 8501

CMD ["epilens-ui", "--server.port", "8501", "--server.address", "0.0.0.0", "--server.headless", "true"]
