"""Preprocesamiento reutilizable: se ajusta SOLO con entrenamiento (dentro del Pipeline)."""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import HOUR_COLUMN, MISSING_CATEGORY


class TimeToHour(BaseEstimator, TransformerMixin):
    """Reemplaza la columna de texto HH:MM:SS por la hora numérica (Hour).

    No aprende nada de los datos. Formar parte del Pipeline hace que el modelo
    guardado reciba la columna `Time` original (corrección de la Fase 1, donde
    Hour se creaba fuera del pipeline).
    """

    def __init__(self, time_column="Time", hour_column=HOUR_COLUMN):
        self.time_column = time_column
        self.hour_column = hour_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        raw = X[self.time_column]
        parsed = pd.to_datetime(raw, format="%H:%M:%S", errors="coerce")
        invalid = raw.notna() & parsed.isna()
        if invalid.any():
            ejemplos = raw[invalid].unique()[:3].tolist()
            raise ValueError(
                f"La columna '{self.time_column}' debe tener formato HH:MM:SS; "
                f"valores inválidos, ej.: {ejemplos}"
            )
        X[self.hour_column] = parsed.dt.hour.astype(float)
        return X.drop(columns=[self.time_column])


def build_pipeline(schema, classifier):
    """Pipeline completo: hora -> imputación/codificación -> clasificador."""
    numeric = list(schema["numeric_columns"])
    steps = []
    if schema["time_column"]:
        steps.append(("time", TimeToHour(schema["time_column"], HOUR_COLUMN)))
        numeric.append(HOUR_COLUMN)

    transformers = [
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=MISSING_CATEGORY)),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), list(schema["categorical_columns"])),
    ]
    if numeric:
        transformers.append(
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric)
        )
    steps.append(("preprocessor", ColumnTransformer(transformers)))
    steps.append(("classifier", classifier))
    return Pipeline(steps)
