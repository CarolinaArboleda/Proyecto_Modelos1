"""Separación de datos, entrenamiento y evaluación (sin lectura/escritura de archivos)."""
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import (StratifiedGroupKFold, StratifiedKFold,
                                     cross_val_score, train_test_split)

from .config import LABEL_ORDER
from .errors import AppError
from .preprocessing import build_pipeline


def make_groups(df, columns):
    """Id de grupo por combinación de columnas de contexto del accidente (o None si faltan)."""
    if any(c not in df.columns for c in columns):
        return None
    return df.groupby(list(columns), dropna=False, sort=False).ngroup().to_numpy()


def split_data(y, groups, test_size, seed):
    """Devuelve índices (train, test). Estratificado; con grupos, ningún grupo se reparte."""
    idx = np.arange(len(y))
    try:
        if groups is None:
            tr, te = train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)
        else:
            k = max(2, round(1 / test_size))
            splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
            tr, te = next(splitter.split(idx, y, groups))
    except ValueError as exc:
        raise AppError(f"No fue posible separar entrenamiento y prueba: {exc}")
    return np.sort(tr), np.sort(te)


def make_cv(groups, folds, seed):
    if groups is None:
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)


def make_classifier(seed, n_estimators=400, max_depth=7, n_jobs=-1):
    return RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, class_weight="balanced",
        random_state=seed, n_jobs=n_jobs,
    )


def parse_depth(text):
    """'7' -> 7, 'none' -> None."""
    t = str(text).strip().lower()
    if t in ("none", "null", "sin"):
        return None
    try:
        value = int(t)
    except ValueError:
        raise AppError(f"Profundidad inválida: '{text}' (use un entero positivo o 'none').")
    if value < 1:
        raise AppError(f"Profundidad inválida: {value} (debe ser >= 1).")
    return value


def cv_f1_macro(schema, X_train, y_train, groups_train, depth, folds, seed, n_estimators, n_jobs):
    pipe = build_pipeline(schema, make_classifier(seed, n_estimators, depth, n_jobs))
    cv = make_cv(groups_train, folds, seed)
    try:
        scores = cross_val_score(pipe, X_train, y_train, groups=groups_train, cv=cv,
                                 scoring="f1_macro", n_jobs=1)
    except ValueError as exc:
        raise AppError(f"Falló la validación cruzada: {exc}")
    return scores


def tune_max_depth(schema, X_train, y_train, groups_train, candidates, folds, seed,
                   n_estimators, n_jobs):
    """Elige max_depth por validación cruzada SOLO con el entrenamiento (F1-macro).

    En empate gana la profundidad menor (modelo más simple); None cuenta como infinito.
    """
    results = []
    for depth in candidates:
        s = cv_f1_macro(schema, X_train, y_train, groups_train, depth, folds, seed,
                        n_estimators, n_jobs)
        results.append({"max_depth": depth, "cv_f1_macro_mean": float(s.mean()),
                        "cv_f1_macro_std": float(s.std()),
                        "cv_f1_macro_folds": [float(v) for v in s]})
    best = max(results, key=lambda r: (round(r["cv_f1_macro_mean"], 6),
                                       -(r["max_depth"] if r["max_depth"] is not None else 10**9)))
    return best["max_depth"], results


def baseline_pipeline(schema, seed):
    return build_pipeline(schema, DummyClassifier(strategy="most_frequent", random_state=seed))


def ordered_labels(y):
    present = set(y)
    return [l for l in LABEL_ORDER if l in present] + sorted(present - set(LABEL_ORDER))


def evaluate(pipe, X, y, labels):
    pred = pipe.predict(X)
    report = classification_report(y, pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, pred, labels=labels)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "f1_macro": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "per_class": {l: {"precision": float(report[l]["precision"]),
                          "recall": float(report[l]["recall"]),
                          "f1": float(report[l]["f1-score"]),
                          "support": int(report[l]["support"])} for l in labels},
        "confusion_matrix": {"labels": labels, "rows": "real", "columns": "predicho",
                             "matrix": cm.tolist()},
    }
