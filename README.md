# Severidad de accidentes de tránsito

**Modelos y Simulación de Sistemas I · Proyecto Integrador 2026-II · Docente: Andrés Parra**

**Integrantes:** Carol Juliana Henao Causil y Carolina Arboleda Guzmán

**Entrega 2 — commit/etiqueta de Git:** `entrega-2` → commit `d6fb4c3a4ee7f2e4ebf8cbc950c7e6293531abc0`

**Entrega 3 — commit/etiqueta de Git:** `entrega-3` → commit `c6dce953166c034888356a1d891b4eda48ad185a`

**Entrega 4 — etiqueta de Git:** `entrega-4` → commit `325ce1d649bfef90cc682a5079ef8f5e4b9c0b77`

Clasificación multiclase de la severidad de un accidente (`Slight Injury`, `Serious Injury`, `Fatal injury`) con el *Road Traffic Accident Dataset of Addis Ababa City* (Bedane, 2020, Mendeley Data, doi:10.17632/xytv86278f.1). Modelo: Random Forest con `class_weight="balanced"` en un Pipeline de scikit-learn; métrica principal F1-macro.

## Fases
| Fase | Carpeta | Contenido |
|---|---|---|
| 1 — Modelo predictivo | [`fase-1/`](fase-1/README.md) | Notebook con EDA, preprocesamiento, modelo y evaluación |
| 2 — Scripts, versionamiento y Docker | [`fase-2/`](fase-2/README.md) | `train.py`, `predict.py`, versiones `v1`/`v2`, Dockerfile y pruebas |
| 3 — API REST | [`fase-3/`](fase-3/README.md) | API FastAPI con `/health`, `/predict` y `/train`, cliente y pruebas |
| 4 — Monitoreo | [`fase-4/`](fase-4/README.md) | Registro de solicitudes, script de métricas, reporte y política de reentrenamiento |

Cada fase conserva las anteriores sin cambios. Las instrucciones completas están en el README de cada carpeta.

## Flujo de trabajo
Cada integrante aporta mediante una rama `feature/...` y un Pull Request revisado hacia `develop`. Cada entrega se integra en `main` y se identifica con una etiqueta.
