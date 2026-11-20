# Fase 3 — API REST

**Integrantes:** Carol Juliana Henao Causil y Carolina Arboleda Guzmán
**Entrega 3 — etiqueta de Git:** `entrega-3`

API con FastAPI que expone el modelo de la Fase 2 (clasificación de `Accident_severity`). Reutiliza sin cambios el código de `fase-2/src` (copiado en `fase-3/src` para que la carpeta y la imagen Docker sean autocontenidas): mismo esquema de entrada, mismo preprocesamiento guardado en el Pipeline y la misma función `train_model`.

## Estructura
```
fase-3/
  api.py                 # rutas /health, /predict, /train
  cliente.py             # cliente de prueba (envía ejemplo_registro.json)
  ejemplo_registro.json  # fila real de data/ejemplo_prediccion.csv (Num 952)
  src/                   # lógica de la Fase 2
  tests/                 # pytest con el TestClient de FastAPI
  data/                  # train.csv y ejemplo_prediccion.csv (montaje de Docker)
  models/ v1/ v2/        # versiones guardadas (montaje de Docker)
  Dockerfile  requirements.txt  pytest.ini
```

## Rutas
| Método | Ruta | Respuesta |
|---|---|---|
| GET | `/health` | 200 · `estado`, versión configurada y versiones disponibles |
| POST | `/predict` | 200 · `prediccion` (etiqueta) y `version`. 422 si falta una variable o el tipo es incompatible. 503 si la versión configurada no existe |
| POST | `/train` | 201 · `version` nueva (v3, v4…) con modelo, métricas y metadatos en `models/<version>/`; no borra las anteriores |

Documentación interactiva: http://localhost:8000/docs

**Entradas:** las 30 variables predictoras de la Fase 2 (sin `Num`, `Casualty_severity` ni `Accident_severity`). Las categorías se envían como texto, `Number_of_vehicles_involved` y `Number_of_casualties` como enteros y `Time` como `HH:MM:SS`. Todas son obligatorias, pero aceptan `null`, que se trata con la regla de la Fase 2: categórica → `"Desconocido"`; numérica u hora → mediana del entrenamiento. Las claves adicionales se ignoran.

## Configuración y selección de versión
Variables de entorno: `MODEL_VERSION` (versión que usa `/predict`, def. `v1`), `MODELS_DIR` (def. `models`), `DATA_PATH` (def. `data/train.csv`), `TRAIN_SEED` (def. `42`).

Para predecir con otra versión disponible (ver `versiones_disponibles` en `/health`):
- Docker: cambiar `-e MODEL_VERSION=v1` por `-e MODEL_VERSION=v2` (o `v3`).
- Local (CMD):
  ```bat
  set MODEL_VERSION=v2
  uvicorn api:app --port 8000
  ```

`/train` usa la configuración de `v1` (max_depth=7, 400 árboles, semilla 42, CV de 5 folds). La versión nueva **no se activa sola**: se activa reiniciando la API con `MODEL_VERSION=<nueva>`.

## Ejecución con Docker (desde `fase-3/`, CMD de Windows)
```bat
for /f %i in ('git rev-parse HEAD') do set GIT_COMMIT=%i
docker build -t modelos-fase3:1.0 --build-arg GIT_COMMIT=%GIT_COMMIT% .
docker run --rm -p 8000:8000 -v "%cd%/data:/app/data:ro" -v "%cd%/models:/app/models" -e MODEL_VERSION=v1 modelos-fase3:1.0
```
- Puerto: `8000` (contenedor) → `8000` (anfitrión).
- Datos: `fase-3/data` montado en `/app/data` (solo lectura).
- Modelos: `fase-3/models` montado en `/app/models` (lectura y escritura: allí `/train` guarda las versiones nuevas, que persisten en el anfitrión).

## Ejecución local (sin Docker)
```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 8000
```

## Cliente de prueba
Con la API en marcha, en otra terminal desde `fase-3/`:
```bat
python cliente.py
python cliente.py --quitar Light_conditions
```
Con curl: `curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "@ejemplo_registro.json"`

## Evidencia de ejecución
Ejecución en Docker Desktop (Windows, CMD) con la imagen `modelos-fase3:1.0` (base `python:3.12.7-slim`), construida desde el commit `ed3e17328e0b4f7d27cf45a103bb9d55b54405b0`. El build terminó sin errores y no entrena durante la construcción.

### 1. `/health`
```
GET /health -> 200
{
  "estado": "ok",
  "version_configurada": "v1",
  "modelo_disponible": true,
  "versiones_disponibles": ["v1", "v2"]
}
```

### 2. `/predict` con un registro válido (`python cliente.py`)
Solicitud: contenido de `ejemplo_registro.json` (fila `Num 952` de `ejemplo_prediccion.csv`, con `Defect_of_vehicle` y `Work_of_casuality` en `null`).

Respuesta:
```
POST /predict -> 200
{
  "prediccion": "Fatal injury",
  "version": "v1",
  "avisos": []
}
```
Coincide con `predict.py` de la Fase 2 para la misma fila. Con `v2` la misma fila da `Slight Injury`: `v1` favorece el recall de la clase fatal (ver Fase 2).

### 3. `/predict` sin una variable obligatoria (`python cliente.py --quitar Light_conditions`)
```
POST /predict -> 422
{
  "error": "Entrada inválida",
  "detalle": ["Falta la variable obligatoria 'Light_conditions'."]
}
```

### 4. `/train` (comprobación manual)
```bat
curl -X POST http://localhost:8000/train
```
```json
{"version":"v3","carpeta":"models/v3","f1_macro_prueba":0.3993,"f1_macro_modelo_base":0.3087,"version_activa_para_predict":"v1","nota":"La nueva versión no se activa sola: use MODEL_VERSION para predecir con ella."}
```
Registro del servidor durante el entrenamiento:
```
[1/6] Leyendo datos: data/train.csv
[2/6] Separando entrenamiento y prueba (antes de ajustar cualquier transformación)
      train=6568  test=1642  grupos compartidos=0
[3/6] max_depth fijo: 7
[4/6] Entrenando modelo y modelo base
[5/6] Evaluando sobre prueba (misma partición y métrica para ambos)
[6/6] Guardando versión
INFO:     172.17.0.1:36640 - "POST /train HTTP/1.1" 201 Created
```
La nueva versión quedó en `fase-3/models/v3/` (`modelo.joblib`, `metrics.json`, `metadata.json`) en el anfitrión, gracias al montaje de `models/`; `v1` y `v2` siguen intactas. `v3` reproduce a `v1` (F1-macro 0,3993, misma configuración y semilla). Por `.gitignore`, solo `v1` y `v2` se suben a Git.

### 5. Pruebas en el contenedor
```bat
docker run --rm modelos-fase3:1.0 python -m pytest
```
```
platform linux -- Python 3.12.7, pytest-9.1.1, pluggy-1.7.0
collected 5 items
tests/test_api.py .....                                                  [100%]
============================== 5 passed in 1.77s ===============================
```

## Pruebas
```bat
python -m pytest
```
Ejecutar desde `fase-3/` en local; en el contenedor se usa el comando de la sección anterior. Las pruebas entrenan un modelo pequeño con datos sintéticos en una carpeta temporal (no dependen de `models/`):

| Caso | Prueba |
|---|---|
| `/health` responde 200 | `test_health_responde_200` |
| Entrada válida → predicción y versión | `test_predict_entrada_valida_devuelve_prediccion_y_version` |
| Falta un campo obligatorio → 422 con su nombre | `test_predict_sin_variable_obligatoria_se_rechaza` |
| Tipo incompatible → 422 (adicional) | `test_predict_tipo_incompatible_se_rechaza` |
| Versión inexistente → 503 (adicional) | `test_predict_modelo_no_disponible_devuelve_503` |