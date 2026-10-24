# Imagen para ejecutar train.py y predict.py (no entrena durante el build).
FROM python:3.12.7-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencias primero (aprovecha la caché de capas de Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código de la aplicación. Datos, modelos y salidas se montan como volúmenes al ejecutar.
COPY src ./src
COPY train.py predict.py pytest.ini ./
COPY tests ./tests

# Hash del commit para metadata.json (docker build --build-arg GIT_COMMIT=$(git rev-parse HEAD) ...)
ARG GIT_COMMIT=desconocido
ENV GIT_COMMIT=${GIT_COMMIT}

# Sin ENTRYPOINT: el comando se indica en `docker run` (python train.py ... / python predict.py ...).
CMD ["python", "train.py", "--help"]
