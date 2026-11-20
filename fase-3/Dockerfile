# Imagen de la API (Fase 3). No entrena durante el build: datos y modelos se montan al ejecutar.
FROM python:3.12.7-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODELS_DIR=models \
    MODEL_VERSION=v1 \
    DATA_PATH=data/train.csv \
    TRAIN_SEED=42

WORKDIR /app

# Dependencias primero (aprovecha la caché de capas de Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código: lógica de la Fase 2 (src/), API, cliente, registro de ejemplo y pruebas
COPY src ./src
COPY api.py cliente.py ejemplo_registro.json pytest.ini ./
COPY tests ./tests

# Hash del commit para metadata.json de las versiones creadas con /train
ARG GIT_COMMIT=desconocido
ENV GIT_COMMIT=${GIT_COMMIT}

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]