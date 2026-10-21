#!/usr/bin/env python
"""Entrena el modelo y guarda una versión nueva en models/<version>/.

Ejemplo (desde fase-2, en una sola línea):
    python train.py --data data/train.csv --target Accident_severity --output-dir models --version v1 --seed 42
"""
import argparse
import sys

from src.config import DEFAULT_DROP_COLUMNS, DEFAULT_TARGET
from src.errors import AppError
from src.modeling import parse_depth
from src.training import format_summary, train_model


def build_parser():
    p = argparse.ArgumentParser(
        description="Entrena el clasificador de severidad de accidentes y guarda una versión "
                    "(modelo + preprocesamiento + métricas + metadatos).")
    p.add_argument("--data", required=True, help="CSV con datos etiquetados.")
    p.add_argument("--target", default=DEFAULT_TARGET, help="Variable objetivo (def.: %(default)s).")
    p.add_argument("--output-dir", default="models", help="Carpeta de versiones (def.: %(default)s).")
    p.add_argument("--version", default=None,
                   help="Identificador único de la versión (ej. v1). Si se omite se usa fecha y hora. "
                        "Si ya existe, el script se detiene sin sobrescribir.")
    p.add_argument("--seed", type=int, default=42, help="Semilla aleatoria (def.: %(default)s).")
    p.add_argument("--test-size", type=float, default=0.2, help="Fracción de prueba (def.: %(default)s).")
    p.add_argument("--no-groups", action="store_true",
                   help="Separar por fila (como en la Fase 1) en vez de mantener juntas las filas "
                        "que parecen ser del mismo accidente.")
    p.add_argument("--max-depth", default="7", help="max_depth del Random Forest o 'none' (def.: %(default)s).")
    p.add_argument("--tune-depth", default=None,
                   help="Lista separada por comas (ej. 3,5,7,10,none): elige max_depth por validación "
                        "cruzada usando solo el entrenamiento; reemplaza a --max-depth.")
    p.add_argument("--n-estimators", type=int, default=400, help="Número de árboles (def.: %(default)s).")
    p.add_argument("--cv-folds", type=int, default=5,
                   help="Folds de validación cruzada sobre entrenamiento; 0 la desactiva (def.: %(default)s).")
    p.add_argument("--drop-columns", default=",".join(DEFAULT_DROP_COLUMNS),
                   help="Columnas que no entran al modelo, separadas por comas (def.: %(default)s).")
    p.add_argument("--data-source", default=None, help="Texto con la fuente de los datos (va a metadata.json).")
    p.add_argument("--n-jobs", type=int, default=-1, help="Procesos paralelos (def.: %(default)s).")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        tune = [parse_depth(x) for x in args.tune_depth.split(",")] if args.tune_depth else None
        drop = [c.strip() for c in args.drop_columns.split(",") if c.strip()]
        result = train_model(
            data_path=args.data, target=args.target, output_dir=args.output_dir,
            version=args.version, seed=args.seed, test_size=args.test_size,
            use_groups=not args.no_groups, max_depth=parse_depth(args.max_depth),
            tune_depths=tune, n_estimators=args.n_estimators, cv_folds=args.cv_folds,
            drop_columns=drop, data_source=args.data_source, n_jobs=args.n_jobs)
    except AppError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - mensaje claro en lugar de una traza larga
        print(f"Error inesperado ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1
    print(format_summary(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
