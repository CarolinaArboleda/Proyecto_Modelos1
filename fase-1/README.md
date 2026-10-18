# Fase 1 — Modelo predictivo: severidad de accidentes de tránsito

**Modelos y Simulación de Sistemas I · Proyecto Integrador 2026-II**
**Integrantes:** Carol Juliana Henao Causil y Carolina Arboleda Guzmán

## Problema y objetivo
Clasificar la severidad de un accidente de tránsito (`Slight Injury`, `Serious Injury`, `Fatal injury`) a partir de datos del conductor, el vehículo, la vía, las condiciones ambientales y la víctima. Clasificación multiclase; variable objetivo: `Accident_severity`.

## Datos
*Road Traffic Accident Dataset of Addis Ababa City* (Bedane, 2020, Mendeley Data, doi:10.17632/xytv86278f.1; también en Kaggle: `saurabhshahane/road-traffic-accidents`). Archivo usado: `train.csv` (8.210 filas, 33 columnas). Desde la Fase 2 el archivo está en el repositorio en `fase-2/data/train.csv` (SHA-256 `c0ddb80c64bada8a6b833077d5f4a861114d709fd008a7aa70134fbd3353e7c5`).

## Contenido de esta carpeta
- `ProyectoModelos.ipynb`: notebook completo (EDA, preparación, baseline, Random Forest, evaluación, almacenamiento).
- `modelo.joblib`: `Pipeline` entrenado (imputación + One-Hot + Random Forest). Guardado con scikit-learn 1.9.1.
- `README.md`: este archivo.

## Algoritmo y métrica
Random Forest (`n_estimators=400`, `max_depth=7`, `class_weight="balanced"`, semilla 42). Métrica principal: **F1-macro** (por el fuerte desbalance: ~86 % leves, ~12,7 % graves, ~1 % fatales).

## Resultados (conjunto de prueba, 20 %, estratificado; salidas guardadas en el notebook)
| | F1-macro | Accuracy |
|---|---|---|
| Modelo base (clase más frecuente) | 0,3087 | 0,8624 |
| Random Forest | 0,4082 | 0,7296 |

Validación cruzada (5 folds, entrenamiento): F1-macro 0,4187 ± 0,0196. Clase fatal: precisión 0,07, recall 0,35 (17 casos).

## Cómo ejecutar el notebook
1. Python 3.12 y las librerías de `fase-2/requirements.txt` más `matplotlib` y `seaborn` (el notebook las usa para los gráficos).
2. El notebook ya lee el CSV desde ../fase-2/data/train.csv
3. Abra `fase-1/ProyectoModelos.ipynb` y ejecute todas las celdas en orden.

Nota: el notebook crea la variable `Hour` fuera del pipeline, así que `modelo.joblib` espera esa columna ya calculada y sin `Time`. La aplicación de la Fase 2 (`fase-2/`) ya integra ese paso dentro del pipeline.
