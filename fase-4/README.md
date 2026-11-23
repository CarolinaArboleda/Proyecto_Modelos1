# Fase 4 — Monitoreo

**Integrantes:** Carol Juliana Henao Causil y Carolina Arboleda Guzmán
**Entrega 4 — etiqueta de Git:** `entrega-4`

Se conserva `fase-3/` y se agrega `fase-4/` con la misma API más un registro de cada solicitud a `/predict` y un script que calcula dos métricas de operación.

## Archivos nuevos respecto a la Fase 3
| Archivo | Función |
|---|---|
| `registro.py` | Agrega una fila CSV por cada llamada a `/predict` (incluidas las rechazadas) |
| `api.py` | Llama al registro en `/predict` y en el manejador de errores de validación |
| `monitoreo.py` | Lee el registro y muestra total de solicitudes y % de fallidas |
| `muestra_registro.csv` | Registro usado para el reporte |
| `ejemplo_registro_2.json` | Segunda fila real (`Num 75`) para el ensayo |
| `tests/test_monitoreo.py` | Pruebas del registro y del script |

## Registro
- Ubicación: `fase-4/logs/predicciones.csv` (configurable con `LOG_PATH`). En Docker, `logs/` se monta en `/app/logs`, así que el archivo queda en el anfitrión y se conserva entre ejecuciones (se abre en modo agregar).
- Columnas: `fecha_hora`, `estado_http`, `version`, `prediccion`.
- Predicción válida → `200`, versión que respondió y etiqueta. Entrada rechazada por validación (campo ausente o tipo inválido) → `422` sin versión ni predicción (se registra en el manejador de errores de FastAPI, antes de usar el modelo). Modelo no disponible → `503` sin versión. Cada intento genera una sola fila; `/health` y `/train` no se registran.

## Ejecución (desde `fase-4/`, PowerShell)
```powershell
docker build -t modelos-fase4:1.0 --build-arg GIT_COMMIT=$(git rev-parse HEAD) .
docker run --rm -p 8000:8000 -v "${PWD}/data:/app/data:ro" -v "${PWD}/models:/app/models" -v "${PWD}/logs:/app/logs" -e MODEL_VERSION=v1 modelos-fase4:1.0
```
Métricas (en otra terminal):
```powershell
python monitoreo.py
```
o dentro de Docker: `docker run --rm -v "${PWD}/logs:/app/logs" modelos-fase4:1.0 python monitoreo.py`

Pruebas: `python -m pytest` → `8 passed` (5 de la Fase 3 + 3 de monitoreo).

## Reporte

**Solicitudes enviadas** (sobre un registro vacío, con la API en `v1`):
1. `python cliente.py` → registro `ejemplo_registro.json` (fila `Num 952`) → `200`, `Fatal injury`.
2. `python cliente.py --archivo ejemplo_registro_2.json` (fila `Num 75`) → `200`, `Serious Injury`.
3. `python cliente.py --quitar Light_conditions` → `422`, "Falta la variable obligatoria 'Light_conditions'".

**Registro obtenido** (`muestra_registro.csv`):
```text
fecha_hora,estado_http,version,prediccion
2026-11-22T20:22:45-05:00,200,v1,Fatal injury
2026-11-22T20:22:54-05:00,200,v1,Serious Injury
2026-11-22T20:22:58-05:00,422,,
```

**Indicadores** (`python monitoreo.py`):
| Indicador | Valor |
|---|---|
| Total de solicitudes | 3 |
| Porcentaje de solicitudes fallidas | 33,3 % (1 de 3) |

**Interpretación:** la API procesó dos registros con `v1` y rechazó el tercero porque faltaba `Light_conditions`; la tasa de fallos fue 1/3 (33,3 %). El rechazo era el resultado esperado de la prueba de validación, así que el porcentaje describe el funcionamiento del servicio y no la calidad del modelo, que requeriría conocer la severidad real de esos accidentes.

## Versiones
| Versión | Ubicación | Configuración | F1-macro (prueba) | Estado |
|---|---|---|---|---|
| `v1` | `fase-4/models/v1/` | max_depth=7 | 0,3993 | **Activa** (la usa la API) |
| `v2` | `fase-4/models/v2/` | max_depth=10 (elegido por CV) | 0,3833 | Conservada como alternativa |

La versión activa se elige con `MODEL_VERSION` (por defecto `v1`) y se confirma en `/health` (`version_configurada`) y en la columna `version` del registro. Para cambiar a `v2`, reinicien el contenedor con `-e MODEL_VERSION=v2`; para volver, usen `-e MODEL_VERSION=v1`. No se borra ninguna versión. Las que crea `/train` (`v3`, …) quedan en `models/` del anfitrión y se activan de la misma forma.

## Política de reentrenamiento
Cada seis meses, o antes si se reúnen 500 accidentes nuevos con la severidad confirmada en el reporte policial, el equipo agregará esos casos a los datos y ejecutará `POST /train` para crear una versión candidata. La candidata y la versión activa se compararán con el F1-macro (métrica de la Fase 2) sobre el mismo conjunto reservado de casos nuevos, que no se usará para entrenar. La candidata se activará con `MODEL_VERSION` solo si mejora el F1-macro sin bajar el recall de `Fatal injury`; si no, se mantiene la versión activa. Un aumento de solicitudes fallidas (422) no lleva a reentrenar: se corrige con quien envía los datos.