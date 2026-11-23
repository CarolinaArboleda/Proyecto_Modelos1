"""Registro de solicitudes a /predict (Fase 4): una fila CSV por llamada, incluidas las rechazadas.

Columnas: fecha_hora, estado_http, version, prediccion. En un error la predicción queda vacía
y, si no se llegó a usar un modelo, también la versión. El archivo se abre en modo agregar,
así que el historial se conserva entre ejecuciones (en Docker, en la carpeta montada logs/).

Ruta configurable con la variable de entorno LOG_PATH (def.: logs/predicciones.csv).
"""
import csv
import os
import threading
from datetime import datetime
from pathlib import Path

COLUMNAS = ["fecha_hora", "estado_http", "version", "prediccion"]
_lock = threading.Lock()   # FastAPI atiende solicitudes en varios hilos


def ruta_registro():
    return Path(os.environ.get("LOG_PATH", "logs/predicciones.csv"))


def registrar(estado_http, version=None, prediccion=None):
    """Agrega una fila al registro; escribe el encabezado si el archivo es nuevo o está vacío."""
    ruta = ruta_registro()
    fila = {
        "fecha_hora": datetime.now().astimezone().isoformat(timespec="seconds"),
        "estado_http": int(estado_http),
        "version": version or "",
        "prediccion": "" if prediccion is None else prediccion,
    }
    with _lock:
        ruta.parent.mkdir(parents=True, exist_ok=True)
        nuevo = not ruta.exists() or ruta.stat().st_size == 0
        with open(ruta, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=COLUMNAS)
            if nuevo:
                writer.writeheader()
            writer.writerow(fila)