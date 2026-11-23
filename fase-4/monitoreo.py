#!/usr/bin/env python
"""Métricas de operación de la API a partir del registro de /predict (Fase 4).

Muestra el total de solicitudes y el porcentaje de solicitudes fallidas (HTTP 4xx o 5xx).
Si el registro no existe o no tiene filas: total 0 y "sin solicitudes".

Ejemplos (desde fase-4):
    python monitoreo.py
    python monitoreo.py --registro logs/predicciones.csv
"""
import argparse
import csv
import os
import sys
from collections import Counter
from pathlib import Path


def leer_registro(ruta):
    ruta = Path(ruta)
    if not ruta.exists() or ruta.stat().st_size == 0:
        return []
    with open(ruta, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def calcular_metricas(filas):
    total = len(filas)
    fallidas = sum(1 for f in filas if 400 <= int(f["estado_http"]) <= 599)
    porcentaje = (fallidas / total * 100) if total else None
    versiones = Counter(f["version"] for f in filas if f["estado_http"] == "200")
    return {"total": total, "fallidas": fallidas, "porcentaje_fallos": porcentaje,
            "predicciones_por_version": dict(versiones)}


def formatear(m, ruta):
    if m["porcentaje_fallos"] is None:
        pct = "sin solicitudes"
    else:
        pct = f"{m['porcentaje_fallos']:.1f} %".replace(".", ",")
    lineas = [
        f"Registro: {ruta}",
        f"Total de solicitudes: {m['total']}",
        f"Solicitudes fallidas (HTTP 4xx/5xx): {m['fallidas']}",
        f"Porcentaje de solicitudes fallidas: {pct}",
    ]
    if m["predicciones_por_version"]:
        detalle = ", ".join(f"{v}: {n}" for v, n in sorted(m["predicciones_por_version"].items()))
        lineas.append(f"Predicciones exitosas por versión: {detalle}")
    return "\n".join(lineas)


def main(argv=None):
    p = argparse.ArgumentParser(description="Total de solicitudes y porcentaje de fallos de /predict.")
    p.add_argument("--registro", default=os.environ.get("LOG_PATH", "logs/predicciones.csv"),
                   help="CSV del registro (def.: LOG_PATH o logs/predicciones.csv).")
    args = p.parse_args(argv)
    try:
        filas = leer_registro(args.registro)
        metricas = calcular_metricas(filas)
    except (KeyError, ValueError, csv.Error) as exc:
        print(f"Error: el registro {args.registro} no tiene el formato esperado ({exc}).",
              file=sys.stderr)
        return 2
    print(formatear(metricas, args.registro))
    return 0


if __name__ == "__main__":
    sys.exit(main())