# Fase 2 — Scripts, versionamiento y Docker: severidad de accidentes de tránsito

**Modelos y Simulación de Sistemas I · Proyecto Integrador 2026-II · Docente: Andrés Parra**
**Integrantes:** Carol Juliana Henao Causil y Carolina Arboleda Guzmán

**Entrega 2 — commit/etiqueta de Git:** `entrega-2` → commit `d6fb4c3a4ee7f2e4ebf8cbc950c7e6293531abc0`

## 1. Problema y relación con la Fase 1

Clasificación multiclase de la severidad de un accidente (`Slight Injury` = herida leve, `Serious Injury` = herida grave, `Fatal injury` = fatal) con datos de Addis Abeba. Se conservan el problema, los datos y el enfoque de la Fase 1 (Random Forest con `class_weight="balanced"`, métrica **F1-macro**, baseline `DummyClassifier(most_frequent)`). La carpeta `fase-1/` se conserva sin cambios en el notebook; el notebook se convirtió en una aplicación con `train.py` y `predict.py`.

## 2. Correcciones respecto a la Fase 1

| Retroalimentación | Qué se hizo |
|---|---|
| Faltaba el CSV y la fuente exacta | `data/train.csv` está en el repositorio (8.210 filas, SHA-256 abajo) y se documenta la fuente (sección 4). En la Fase 1 el archivo existía en tu carpeta pero **no estaba rastreado por Git**. |
| `Hour` se creaba fuera del pipeline | La hora se calcula dentro del `Pipeline` (`TimeToHour`). El modelo guardado recibe la columna `Time` original (`HH:MM:SS`). |
| Rutas/nombres/estructura inconsistentes; sin README en `fase-1` | Estructura única descrita aquí; se agregó `fase-1/README.md` con el nombre real del notebook y las métricas reales. |
| Versiones de librerías | `requirements.txt` con versiones fijas y versión de Python (sección 3); también se guardan en `metadata.json`. |
| Unidad de análisis y separación | Se investigó (sección 5.1) y la separación ahora **mantiene juntas** las filas que parecen ser del mismo accidente. |
| ¿Cuándo se predice? | Se define el momento de uso y se mide la dependencia de `Cause_of_accident` (sección 5.2). |
| Resultados inconsistentes entre notebook y README | `metrics.json` es la única fuente; las cifras de este README salen de él. |
| Selección de `max_depth` | `--tune-depth` la elige por validación cruzada **solo con entrenamiento** (sección 8). |
| Clase fatal | Se guarda reporte por clase y matriz de confusión; análisis en la sección 8. |

Pendiente de la Fase 1 que no afecta a esta entrega: usar proporciones por grupo (no solo conteos) en el EDA del notebook.

## 3. Entorno

- **Python 3.12** (probado con 3.12.3 local y `python:3.12.7-slim` en el `Dockerfile`).
- Dependencias con versión explícita en `requirements.txt` (numpy, pandas, scikit-learn, joblib, pytest).

Bash (Linux/macOS/Git Bash), desde `fase-2/`:
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
PowerShell (Windows), desde `fase-2\`:
```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 4. Datos

- **Fuente original:** *Road Traffic Accident Dataset of Addis Ababa City*, Tarikwa Tesfa Bedane (2020), Mendeley Data, doi:10.17632/xytv86278f.1 (también en Kaggle: `saurabhshahane/road-traffic-accidents`; el original tiene 12.316 filas).
- **Archivo usado:** `data/train.csv`, 8.210 filas × 33 columnas, incluido en el repositorio. Enlace exacto de donde se descargó: el archivo se entrega ya incluido en `data/train.csv`; la fuente original es la de arriba (Kaggle `saurabhshahane/road-traffic-accidents`, Mendeley doi:10.17632/xytv86278f.1).
- **SHA-256:** `c0ddb80c64bada8a6b833077d5f4a861114d709fd008a7aa70134fbd3353e7c5` (se registra también en cada `metadata.json`).
- No se requieren credenciales ni descargas durante la ejecución.

**Esquema del CSV de predicción** (`data/ejemplo_prediccion.csv`, 10 filas reales, sin variable objetivo): las 30 columnas predictoras `Time, Day_of_week, Age_band_of_driver, Sex_of_driver, Educational_level, Vehicle_driver_relation, Driving_experience, Type_of_vehicle, Owner_of_vehicle, Service_year_of_vehicle, Defect_of_vehicle, Area_accident_occured, Lanes_or_Medians, Road_allignment, Types_of_Junction, Road_surface_type, Road_surface_conditions, Light_conditions, Weather_conditions, Type_of_collision, Number_of_vehicles_involved, Number_of_casualties, Vehicle_movement, Casualty_class, Sex_of_casualty, Age_band_of_casualty, Work_of_casuality, Fitness_of_casuality, Pedestrian_movement, Cause_of_accident`. Opcionalmente `Num` (identificador que se copia a la salida). `Time` es texto `HH:MM:SS`; `Number_of_vehicles_involved` y `Number_of_casualties` son numéricas; el resto son texto.

Reglas de entrada de `predict.py`:
- El orden de las columnas del CSV no importa.
- Una **celda vacía** no es un error: se imputa como en el entrenamiento (categóricas → `Desconocido`; numéricas y hora → mediana del entrenamiento).
- Una **columna ausente**, un valor no numérico en una columna numérica o una hora que no sea `HH:MM:SS` → error claro y código de salida 2, sin crear el archivo de salida.
- **Columnas adicionales**: se ignoran con un aviso.
- **Categorías no vistas** en el entrenamiento: no fallan; se codifican como todo ceros (`OneHotEncoder(handle_unknown="ignore")`) y se avisa.

## 5. Decisiones metodológicas

### 5.1 Unidad de análisis y separación sin fuga
Cada fila es una víctima (*casualty*) y no hay identificador de accidente. Para no suponer independencia sin evidencia se buscaron filas del mismo contexto de accidente (misma hora y día, zona, tipo de vía, condiciones, tipo de colisión y número de vehículos/víctimas; 13 columnas, ver `GROUP_COLUMNS` en `src/config.py`):

- 8.210 filas forman **8.011 grupos**; 197 grupos tienen más de una fila (396 filas, máx. 3 por grupo).
- En **196 de esos 197 grupos la severidad coincide**. Como la severidad es del accidente, esto es consistente con que sean víctimas del mismo accidente. No es una prueba definitiva (no hay identificador oficial).

Por eso `train.py` separa con `StratifiedGroupKFold` (un fold de prueba ≈ 20 %) usando esos grupos: ningún grupo queda repartido entre entrenamiento y prueba (`grupos_compartidos_train_test: 0` en `metadata.json`). Con `--no-groups` se reproduce la separación por fila de la Fase 1. El efecto práctico fue pequeño (sección 8), pero la separación ahora es defendible.

**Fuga evitada:** el objetivo no es predictor; se excluyen `Num` (identificador) y `Casualty_severity` (casi equivale al objetivo); la separación ocurre antes de ajustar imputación y codificación, que viven dentro del `Pipeline` y se ajustan solo con entrenamiento; `max_depth` se elige solo con validación cruzada del entrenamiento.

### 5.2 ¿Cuándo se hace la predicción?
Se define como **al momento del reporte policial** del accidente (con la información del reporte), para clasificar la severidad registrada, no para despachar la primera atención. Algunas variables, como `Cause_of_accident`, podrían no estar disponibles en una priorización inicial. Para medir cuánto dependemos de ella se entrenó una variante sin esa columna (`--drop-columns Num,Casualty_severity,Cause_of_accident`): F1-macro **0,3921** frente a 0,3993 de `v1` (prueba). La dependencia es baja; aun así, un uso en triaje inicial requeriría reentrenar con las variables disponibles en ese momento (el mismo script lo permite).

## 6. Uso

Todos los comandos se ejecutan desde `fase-2/` y deben escribirse en **una sola línea** (el `\` o `` ` `` es para continuar línea).

**Ayuda**
```bash
python train.py --help
python predict.py --help
```

**Entrenar una versión**
```bash
python train.py --data data/train.csv --target Accident_severity --output-dir models --version v1 --seed 42
python train.py --data data/train.csv --target Accident_severity --output-dir models --version v2 --seed 42 --tune-depth 3,5,7,10,15
```
Si la versión ya existe, el script se detiene con un error (no sobrescribe). Si se omite `--version`, usa fecha y hora.

**Predecir** (el modelo se puede indicar como archivo o como carpeta de versión)
```bash
python predict.py --model models/v1/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/predicciones.csv
python predict.py --model models/v2 --input data/ejemplo_prediccion.csv --output outputs/predicciones_v2.csv --proba
```
Salida: `fila` (1, 2, 3… en el orden de entrada), `Num` (si existe), `prediccion` y, con `--proba`, `proba_<clase>`. Significado: `Slight Injury` = leve, `Serious Injury` = grave, `Fatal injury` = fatal.

**Pruebas**
```bash
python -m pytest
```

## 7. Versiones, métricas y resultados

```
models/
  v1/  modelo.joblib  metrics.json  metadata.json     # max_depth=7 (configuración de la Fase 1)
  v2/  modelo.joblib  metrics.json  metadata.json     # max_depth elegido por CV (resultó 10)
outputs/                                              # predicciones generadas
```
`modelo.joblib` guarda el `Pipeline` completo (hora → imputación → One-Hot → Random Forest) y el esquema de columnas. `metadata.json` incluye versión, fecha, algoritmo y parámetros, variables y tipos, semilla, datos (fuente, SHA-256, nº de filas), separación, métrica, versiones de Python y librerías, y hash del commit. Reproducibilidad: el Random Forest usa `random_state`; con el mismo entorno y datos los resultados se repiten, y entre sistemas pueden aparecer diferencias numéricas pequeñas.

## 8. Resultados (conjunto de prueba, 1.642 filas, 17 fatales)

| | F1-macro | Accuracy | CV F1-macro (5 folds, entrenamiento) |
|---|---|---|---|
| Modelo base (más frecuente) | 0,3087 | 0,8624 | — |
| **`v1`** (max_depth=7) | **0,3993** | 0,7326 | 0,4092 ± 0,0223 |
| `v2` (max_depth=10 por CV) | 0,3833 | 0,8222 | 0,4543 ± 0,0483 |

- Ambos modelos superan al base en F1-macro (el base solo gana en accuracy porque predice siempre la clase leve, 86 %, y por eso la métrica principal es F1-macro).
- **Selección de `max_depth`:** en la Fase 1 el 7 se fijó tras una experimentación informal descrita en el notebook, sin registro de que fuera solo con entrenamiento. Aquí se compara por CV con el entrenamiento: profundidad 3 → 0,324; 5 → 0,364; 7 → 0,409; **10 → 0,454**; 15 → 0,343. La CV eligió 10, pero en prueba `v2` queda por debajo de `v1`. La variación entre folds (± 0,048) y los 17 casos fatales hacen que esa diferencia no sea concluyente, y no se vuelve a elegir mirando la prueba (eso sería fuga).
- **Clase fatal (`v1`):** precisión 0,07 y recall 0,35 (6 de 17 detectados, 70 falsas alarmas). En `v2`: precisión 0,14, recall 0,06 (1 de 17). Con tan pocos casos las cifras de esta clase son muy inestables; el modelo sirve como apoyo, no para decidir sobre casos fatales.
- **Comparación con la Fase 1:** con `--no-groups` (separación por fila) este código da F1-macro 0,4015 y accuracy 0,7247, cerca de los 0,4082 / 0,7296 del notebook. No son idénticas por la versión de scikit-learn y el orden de columnas. Con la separación por grupos baja a 0,3993; el cambio es pequeño frente a la variabilidad.
- **Limitaciones:** desbalance fuerte, 17 fatales en prueba, sin identificador oficial de accidente, 35 % de filas sin datos de víctima (`na`), y modelo estimado con un solo conjunto de prueba.

## 9. Docker

Se construye la imagen sin entrenar. Los datos se montan desde el anfitrión y `models/` y `outputs/` persisten fuera del contenedor.

**Bash** (desde `fase-2/`):
```bash
docker build -t modelos-fase2:1.0 --build-arg GIT_COMMIT=$(git rev-parse HEAD) .
mkdir -p models outputs
docker run --rm -v "$PWD/data:/app/data:ro" -v "$PWD/models:/app/models" modelos-fase2:1.0 python train.py --data data/train.csv --target Accident_severity --output-dir models --version v-docker --seed 42
docker run --rm -v "$PWD/data:/app/data:ro" -v "$PWD/models:/app/models:ro" -v "$PWD/outputs:/app/outputs" modelos-fase2:1.0 python predict.py --model models/v-docker/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/predicciones_docker.csv
```
**PowerShell** (desde `fase-2\`):
```powershell
docker build -t modelos-fase2:1.0 --build-arg GIT_COMMIT=$(git rev-parse HEAD) .
mkdir models, outputs -Force
docker run --rm -v "${PWD}/data:/app/data:ro" -v "${PWD}/models:/app/models" modelos-fase2:1.0 python train.py --data data/train.csv --target Accident_severity --output-dir models --version v-docker --seed 42
docker run --rm -v "${PWD}/data:/app/data:ro" -v "${PWD}/models:/app/models:ro" -v "${PWD}/outputs:/app/outputs" modelos-fase2:1.0 python predict.py --model models/v-docker/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/predicciones_docker.csv
```
La versión `v-docker` queda en `models/` del anfitrión (ignorada por Git; solo `v1` y `v2` se versionan). Terminal usada para esta guía: **CMD de Windows** (el que se usó para la evidencia):
```
for /f %i in ('git rev-parse HEAD') do set GIT_COMMIT=%i
docker build -t modelos-fase2:1.0 --build-arg GIT_COMMIT=%GIT_COMMIT% .
mkdir models outputs
docker run --rm -v "%cd%/data:/app/data:ro" -v "%cd%/models:/app/models" modelos-fase2:1.0 python train.py --data data/train.csv --target Accident_severity --output-dir models --version v-docker --seed 42
docker run --rm -v "%cd%/data:/app/data:ro" -v "%cd%/models:/app/models:ro" -v "%cd%/outputs:/app/outputs" modelos-fase2:1.0 python predict.py --model models/v-docker/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/predicciones_docker.csv
```. Las pruebas también corren en el contenedor: `docker run --rm modelos-fase2:1.0 python -m pytest`.

## 10. Pruebas (`tests/test_aplicacion.py`, 12 pruebas)

Usan datos sintéticos con el esquema real y `tmp_path` (sin red, no tocan `models/`; ~6 s).

| Comportamiento | Prueba |
|---|---|
| Entrenamiento y almacenamiento | `test_entrenamiento_genera_artefacto_metricas_y_metadatos` |
| Carga y predicción (igual al modelo en memoria, una por fila, orden conservado) | `test_modelo_guardado_predice_igual_que_el_modelo_en_memoria`, `test_probabilidades_opcionales_suman_uno` |
| Valores faltantes | `test_nulos_en_variables_que_los_admiten_se_imputan` |
| Entrada inválida (error claro, código ≠ 0, sin archivo de salida) | `test_columna_requerida_ausente_da_error_claro`, `test_cli_con_columna_ausente_...`, `test_modelo_inexistente_y_tipo_incompatible` |
| Versionamiento (v2 conserva v1; id repetido no sobrescribe) | `test_segunda_version_conserva_la_primera_y_id_repetido_no_sobrescribe` |
| Extras | orden de columnas, categoría no vista, transformador de hora, `--help` |

## 11. Evidencia de ejecución

Resultado de `python -m pytest` (entorno limpio con `requirements.txt`): `12 passed`.

`train.py` (v1): `Modelo F1-macro 0.3993 | Accuracy 0.7326`, `Modelo base F1-macro 0.3087`, archivos en `models/v1/`.
`predict.py` (v1): `Predicciones generadas: 10 -> outputs/predicciones.csv`.
Error esperado (columna ausente), código de salida 2 y sin archivo:
```
Error: Faltan columnas obligatorias en el CSV de entrada: ['Light_conditions']. Columnas requeridas: [...]
```
Docker (CMD de Windows, Docker 29.6.2):
- `docker build -t modelos-fase2:1.0 ...` terminó sin errores (imagen `modelos-fase2:1.0`, base `python:3.12.7-slim`); no se entrena durante el build.
- `docker run --rm modelos-fase2:1.0 python -m pytest` → `12 passed` (Python 3.12.7).
- Entrenamiento en el contenedor (`--version v-docker`): F1-macro 0.3993 y modelo base 0.3087, igual que `v1`.
- Predicción en el contenedor: `Predicciones generadas: 10 -> outputs/predicciones_docker.csv`, mismas predicciones que la ejecución local.
- Persistencia: con el contenedor ya eliminado (`--rm`), `models/v-docker/` (`modelo.joblib`, `metrics.json`, `metadata.json`) y `outputs/predicciones_docker.csv` existen en el equipo anfitrión; `metadata.json` guarda el hash del commit pasado con `--build-arg`.
## 12. Estructura

```
fase-2/
  train.py  predict.py  requirements.txt  Dockerfile  .dockerignore  .gitignore  pytest.ini  README.md
  src/   config.py  data.py  preprocessing.py  modeling.py  training.py  prediction.py  storage.py  errors.py
  tests/ conftest.py  test_aplicacion.py
  data/  train.csv  ejemplo_prediccion.csv
  models/ v1/  v2/        outputs/
```
`train.py` y `predict.py` solo leen argumentos y llaman a `src/` (`train_model`, `predict_file`), de modo que esa lógica se puede reutilizar en `/train` y `/predict` de la API de la fase siguiente.

## 13. Fuentes
- Bedane, T. T. (2020). *Road Traffic Accident Dataset of Addis Ababa City*. Mendeley Data, doi:10.17632/xytv86278f.1.
- Documentación de scikit-learn (`Pipeline`, `ColumnTransformer`, `StratifiedGroupKFold`, `RandomForestClassifier`).
- Código de la Fase 1 (`fase-1/ProyectoModelos.ipynb`) adaptado a módulos.
