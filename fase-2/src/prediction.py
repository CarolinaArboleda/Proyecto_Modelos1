"""Predicción con una versión guardada. No reentrena ni reajusta transformaciones."""
import os
from pathlib import Path

import pandas as pd

from .data import prepare_features, read_csv
from .errors import AppError
from .storage import load_bundle

DEFAULT_ID_COLUMN = "Num"


def predict_dataframe(bundle, df, with_proba=False, id_column=None):
    """Una predicción por fila, en el mismo orden. Devuelve (DataFrame, avisos)."""
    if len(df) == 0:
        raise AppError("El archivo de entrada no tiene filas.")
    X, extra = prepare_features(df, bundle["schema"])
    avisos = []
    extra = [c for c in extra if c != id_column]   # el identificador no es una "columna rara"
    if extra:
        avisos.append(f"Columnas adicionales ignoradas: {extra}")
    unseen = {}
    for col, known in bundle["known_categories"].items():
        nuevas = set(X[col].dropna().unique()) - set(known)
        if nuevas:
            unseen[col] = sorted(nuevas)[:3]
    if unseen:
        avisos.append("Categorías no vistas en el entrenamiento (se codifican como todo ceros): "
                      f"{unseen}")
    pipe = bundle["pipeline"]
    try:
        pred = pipe.predict(X)
        proba = pipe.predict_proba(X) if with_proba else None
    except ValueError as exc:
        raise AppError(f"No se pudo procesar la entrada: {exc}")
    out = pd.DataFrame({"fila": range(1, len(df) + 1), "prediccion": pred})
    if proba is not None:
        for i, cls in enumerate(pipe.classes_):
            out[f"proba_{cls}"] = proba[:, i].round(6)
    return out, avisos


def predict_file(model, input_path, output_path, id_column=None, with_proba=False):
    bundle = load_bundle(model)
    df = read_csv(input_path, "archivo de entrada")
    if Path(output_path).resolve() == Path(input_path).resolve():
        raise AppError("La ruta de salida no puede ser la misma que la de entrada.")
    if Path(output_path).is_dir():
        raise AppError(f"La ruta de salida es una carpeta, no un archivo: {output_path}")
    if id_column is not None and id_column not in df.columns:
        raise AppError(f"La columna identificadora '{id_column}' no existe en el archivo de entrada.")
    id_column = id_column or (DEFAULT_ID_COLUMN if DEFAULT_ID_COLUMN in df.columns else None)

    out, avisos = predict_dataframe(bundle, df, with_proba=with_proba, id_column=id_column)
    if id_column:
        out.insert(1, id_column, df[id_column].to_numpy())

    # Se escribe en un temporal y se reemplaza al final: si algo falla no queda un archivo "válido".
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(f".{out_path.name}.tmp-{os.getpid()}")
    try:
        out.to_csv(tmp, index=False, encoding="utf-8")
        os.replace(tmp, out_path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return {"n_filas": len(out), "version": bundle["version"], "avisos": avisos,
            "conteo": out["prediccion"].value_counts().to_dict(), "salida": str(out_path)}
