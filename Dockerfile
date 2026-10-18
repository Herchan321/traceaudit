FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Dépendances système (minimales)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Dépendances Python
COPY pyproject.toml ./
RUN pip install --upgrade pip && \
    pip install -e ".[rag,dev]" && \
    pip install fastapi uvicorn streamlit

# Code source
COPY src ./src
COPY config ./config
COPY .env ./

# Exposition des ports : API (8000) et Streamlit (8501)
EXPOSE 8000 8501

CMD ["uvicorn", "audittrace.api:app", "--host", "0.0.0.0", "--port", "8000"]
