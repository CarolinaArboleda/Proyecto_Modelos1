"""Lectura y validación de datos. Funciones puras: no se ejecuta nada al importar."""
import hashlib
from pathlib import Path

import pandas as pd

from .config import NUMERIC_COLUMNS, TIME_COLUMN
from .errors import AppError


def sha256_file(path, chunk_size=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path, description="archivo CSV"):
    """Lee un CSV como texto (los vacíos quedan como NaN). Falla con AppError claro."""
    p = Path(path)
    if not p.exists():
        raise AppError(f"No se encontró el {description}: {p}")
    if not p.is_file():
        raise AppError(f"La ruta del {description} no es un archivo: {p}")
    try:
        df = pd.read_csv(p, dtype=str, encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        raise AppError(f"El {description} está vacío: {p}")
    except (pd.errors.ParserError, UnicodeDecodeError, OSError) as exc:
        raise AppError(f"No se pudo leer el {description} ({p}): {exc}")
    df.columns = [str(c).strip() for c in df.columns]
    if df.empty:
        raise AppError(f"El {description} no tiene filas de datos: {p}")
    return df


def build_schema(df, target, drop_columns):
    """Define qué columnas entran al modelo y de qué tipo es cada una."""
    if target not in df.columns:
        raise AppError(
            f"La variable objetivo '{target}' no existe en el archivo de datos. "
            f"Columnas disponibles: {list(df.columns)}"
        )
    drop = [c for c in drop_columns if c in df.columns]
    features = [c for c in df.columns if c != target and c not in drop]
    if not features:
        raise AppError("No quedan variables predictoras después de excluir columnas.")
    numeric = [c for c in NUMERIC_COLUMNS if c in features]
    time_column = TIME_COLUMN if TIME_COLUMN in features else None
    categorical = [c for c in features if c not in numeric and c != time_column]
    return {
        "target": target,
        "drop_columns": drop,
        "feature_columns": features,
        "numeric_columns": numeric,
        "time_column": time_column,
        "categorical_columns": categorical,
    }


def prepare_features(df, schema):
    """Valida columnas/tipos y devuelve (X, columnas_extra).

    - Reordena las columnas según el esquema del entrenamiento (el orden del CSV no importa).
    - Una celda nula NO es una columna ausente: los nulos los trata el preprocesamiento guardado.
    - Las columnas adicionales se ignoran (se informan en `columnas_extra`).
    """
    required = schema["feature_columns"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise AppError(
            "Faltan columnas obligatorias en el CSV de entrada: "
            f"{missing}. Columnas requeridas: {required}"
        )
    extra = [c for c in df.columns if c not in required]
    X = df[required].copy()

    for col in schema["numeric_columns"]:
        converted = pd.to_numeric(X[col], errors="coerce")
        bad = X[col].notna() & converted.isna()
        if bad.any():
            ejemplos = X.loc[bad, col].unique()[:3].tolist()
            filas = [int(i) + 1 for i in X.index[bad][:5]]
            raise AppError(
                f"La columna numérica '{col}' contiene valores no numéricos "
                f"(ej.: {ejemplos}; filas de datos {filas})."
            )
        X[col] = converted.astype(float)

    for col in schema["categorical_columns"]:
        X[col] = X[col].astype(object)
    if schema["time_column"]:
        X[schema["time_column"]] = X[schema["time_column"]].astype(object)
    return X, extra
