FROM python:3.10-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN pip install --upgrade pip && pip install streamlit requests pandas python-dotenv

COPY streamlit_app.py /app/streamlit_app.py

EXPOSE 8501

CMD ["streamlit", "run", "/app/streamlit_app.py", "--server.port", "8501", "--server.address", "0.0.0.0", "--server.headless", "true"]
