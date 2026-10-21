#!/usr/bin/env python
"""Genera predicciones con una versión guardada (no reentrena).

Ejemplo (desde fase-2, en una sola línea):
    python predict.py --model models/v1/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/predicciones.csv
"""
import argparse
import sys

from src.errors import AppError
from src.prediction import predict_file


def build_parser():
    p = argparse.ArgumentParser(
        description="Predice la severidad de accidentes para las filas de un CSV (sin variable objetivo).")
    p.add_argument("--model", required=True,
                   help="Ruta de modelo.joblib o de la carpeta de la versión (ej. models/v1).")
    p.add_argument("--input", required=True, help="CSV con las variables de entrada.")
    p.add_argument("--output", required=True, help="CSV de salida con la columna 'prediccion'.")
    p.add_argument("--id-col", default=None,
                   help="Columna del CSV de entrada que se copia a la salida como identificador "
                        "(por defecto 'Num' si existe).")
    p.add_argument("--proba", action="store_true", help="Agrega una columna de probabilidad por clase.")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        res = predict_file(args.model, args.input, args.output, id_column=args.id_col,
                           with_proba=args.proba)
    except AppError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"Error inesperado ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1
    for aviso in res["avisos"]:
        print(f"Aviso: {aviso}", file=sys.stderr)
    print(f"Versión del modelo: {res['version']}")
    print(f"Predicciones generadas: {res['n_filas']} -> {res['salida']}")
    print(f"Distribución: {res['conteo']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
