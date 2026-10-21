"""Flujo de entrenamiento completo, reutilizable (por train.py y, más adelante, por la API)."""
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from .config import (DEFAULT_DATA_SOURCE, DEFAULT_DROP_COLUMNS, DEFAULT_TARGET, GROUP_COLUMNS,
                     LABEL_MEANING, MISSING_CATEGORY)
from .data import build_schema, prepare_features, read_csv, sha256_file
from .errors import AppError
from .modeling import (baseline_pipeline, cv_f1_macro, evaluate, make_classifier, make_groups,
                       ordered_labels, split_data, tune_max_depth)
from .preprocessing import build_pipeline
from .storage import ensure_version_free, get_git_info, save_version, validate_version_name


def default_version():
    return datetime.now(timezone.utc).strftime("v%Y%m%d-%H%M%S")


def _type_description(schema):
    types = {c: "categórica (texto)" for c in schema["categorical_columns"]}
    types.update({c: "numérica" for c in schema["numeric_columns"]})
    if schema["time_column"]:
        types[schema["time_column"]] = "texto HH:MM:SS (se convierte en la hora, Hour)"
    return {c: types[c] for c in schema["feature_columns"]}


def train_model(*, data_path, target=DEFAULT_TARGET, output_dir="models", version=None, seed=42,
                test_size=0.2, use_groups=True, max_depth=7, tune_depths=None, n_estimators=400,
                cv_folds=5, drop_columns=None, data_source=None, n_jobs=-1, log=print):
    """Entrena, evalúa (modelo y modelo base) y guarda una versión nueva en output_dir/version."""
    drop_columns = DEFAULT_DROP_COLUMNS if drop_columns is None else list(drop_columns)
    if not 0 < test_size < 1:
        raise AppError("--test-size debe estar entre 0 y 1 (exclusivo).")
    if cv_folds not in (0,) and cv_folds < 2:
        raise AppError("--cv-folds debe ser 0 (sin validación cruzada) o >= 2.")
    version = version or default_version()
    validate_version_name(version)
    ensure_version_free(output_dir, version)  # falla temprano, antes de entrenar

    log(f"[1/6] Leyendo datos: {data_path}")
    df = read_csv(data_path, "archivo de datos")
    schema = build_schema(df, target, drop_columns)
    X, _ = prepare_features(df, schema)
    if df[target].isna().any():
        raise AppError(f"La variable objetivo '{target}' tiene valores vacíos; "
                       "corrija los datos (no se eliminan filas en silencio).")
    y = df[target].astype(object)
    labels = ordered_labels(y)
    if len(labels) < 2:
        raise AppError("La variable objetivo debe tener al menos dos clases.")

    groups = make_groups(df, GROUP_COLUMNS) if use_groups else None
    if use_groups and groups is None:
        log("      Aviso: faltan columnas de contexto; se usa separación estratificada por fila.")
        use_groups = False

    log("[2/6] Separando entrenamiento y prueba (antes de ajustar cualquier transformación)")
    tr, te = split_data(y.to_numpy(), groups, test_size, seed)
    X_train, X_test = X.iloc[tr], X.iloc[te]
    y_train, y_test = y.iloc[tr], y.iloc[te]
    groups_train = groups[tr] if groups is not None else None
    shared = int(len(set(groups[tr]) & set(groups[te]))) if groups is not None else None
    log(f"      train={len(tr)}  test={len(te)}  "
        f"{'grupos compartidos=' + str(shared) if groups is not None else 'sin grupos'}")

    tuning = None
    if tune_depths:
        log(f"[3/6] Eligiendo max_depth por CV ({cv_folds} folds) solo con entrenamiento: {tune_depths}")
        if cv_folds < 2:
            raise AppError("Para --tune-depth use --cv-folds >= 2.")
        max_depth, tuning = tune_max_depth(schema, X_train, y_train, groups_train, tune_depths,
                                           cv_folds, seed, n_estimators, n_jobs)
        log(f"      max_depth elegido: {max_depth}")
    else:
        log(f"[3/6] max_depth fijo: {max_depth}")

    log("[4/6] Entrenando modelo y modelo base")
    model = build_pipeline(schema, make_classifier(seed, n_estimators, max_depth, n_jobs))
    model.fit(X_train, y_train)
    base = baseline_pipeline(schema, seed)
    base.fit(X_train, y_train)

    log("[5/6] Evaluando sobre prueba (misma partición y métrica para ambos)")
    model_metrics = evaluate(model, X_test, y_test, labels)
    base_metrics = evaluate(base, X_test, y_test, labels)
    cv_info = None
    if cv_folds >= 2:
        if tuning is not None:
            chosen = next(r for r in tuning if r["max_depth"] == max_depth)
            cv_info = {"folds": cv_folds, "f1_macro_mean": chosen["cv_f1_macro_mean"],
                       "f1_macro_std": chosen["cv_f1_macro_std"],
                       "f1_macro_folds": chosen["cv_f1_macro_folds"]}
        else:
            s = cv_f1_macro(schema, X_train, y_train, groups_train, max_depth, cv_folds, seed,
                            n_estimators, n_jobs)
            cv_info = {"folds": cv_folds, "f1_macro_mean": float(s.mean()),
                       "f1_macro_std": float(s.std()), "f1_macro_folds": [float(v) for v in s]}

    metrics = {
        "version": version,
        "metrica_principal": "f1_macro",
        "conjunto_evaluacion": "prueba (no usado para ajustar ni para elegir hiperparámetros)",
        "modelo": model_metrics,
        "modelo_base": {"descripcion": "DummyClassifier(most_frequent)", **base_metrics},
        "mejora_f1_macro_vs_base": model_metrics["f1_macro"] - base_metrics["f1_macro"],
        "validacion_cruzada_entrenamiento": cv_info,
        "seleccion_max_depth": tuning,
    }

    git = get_git_info()
    clf = model.named_steps["classifier"]
    metadata = {
        "version": version,
        "fecha_creacion_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tipo_problema": "clasificación multiclase",
        "variable_objetivo": target,
        "significado_etiquetas": {l: LABEL_MEANING.get(l, l) for l in labels},
        "algoritmo": "RandomForestClassifier (scikit-learn) dentro de un Pipeline",
        "variables_entrada": _type_description(schema),
        "columnas_excluidas": schema["drop_columns"],
        "parametros_modelo": {"n_estimators": clf.n_estimators, "max_depth": clf.max_depth,
                              "class_weight": clf.class_weight, "random_state": seed},
        "semilla": seed,
        "preprocesamiento": {
            "hora": "Time (HH:MM:SS) -> Hour numérica, dentro del pipeline",
            "categoricas_nulas": f"constante '{MISSING_CATEGORY}'",
            "numericas_nulas": "mediana del entrenamiento",
            "codificacion": "OneHotEncoder(handle_unknown='ignore'): categoría no vista = todo ceros",
            "escalamiento": "no aplica (Random Forest)",
        },
        "datos": {
            "fuente": data_source or DEFAULT_DATA_SOURCE,
            "archivo": Path(data_path).name,
            "sha256": sha256_file(data_path),
            "n_observaciones": int(len(df)),
            "n_columnas": int(df.shape[1]),
        },
        "separacion": {
            "estrategia": ("StratifiedGroupKFold (1 fold de prueba) con grupos por contexto del accidente"
                           if groups is not None else "train_test_split estratificado por fila"),
            "columnas_de_grupo": GROUP_COLUMNS if groups is not None else None,
            "n_grupos": int(len(set(groups))) if groups is not None else None,
            "grupos_compartidos_train_test": shared,
            "test_size_solicitado": test_size,
            "n_entrenamiento": int(len(tr)),
            "n_prueba": int(len(te)),
            "clases_entrenamiento": {k: int(v) for k, v in y_train.value_counts().items()},
            "clases_prueba": {k: int(v) for k, v in y_test.value_counts().items()},
        },
        "metrica_evaluacion": "f1_macro (principal); accuracy y reporte por clase como referencia",
        "entorno": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__, "pandas": pd.__version__,
            "numpy": np.__version__, "joblib": joblib.__version__,
            "archivo_dependencias": "requirements.txt",
        },
        "codigo": {"commit_git": git["commit"], "cambios_sin_commit": git["cambios_sin_commit"]},
    }

    bundle = {
        "pipeline": model,
        "schema": schema,
        "classes": [str(c) for c in model.classes_],
        "known_categories": {c: sorted(X_train[c].dropna().unique().tolist())
                             for c in schema["categorical_columns"]},
        "version": version,
    }

    log("[6/6] Guardando versión")
    paths = save_version(output_dir, version, bundle, metrics, metadata)
    return {"version": version, "paths": paths, "metrics": metrics, "metadata": metadata,
            "bundle": bundle}


def format_summary(result):
    m, b = result["metrics"]["modelo"], result["metrics"]["modelo_base"]
    fatal = m["per_class"].get("Fatal injury")
    lines = [
        "",
        f"Versión entrenada: {result['version']}",
        f"  Modelo       -> F1-macro: {m['f1_macro']:.4f} | Accuracy: {m['accuracy']:.4f}",
        f"  Modelo base  -> F1-macro: {b['f1_macro']:.4f} | Accuracy: {b['accuracy']:.4f}",
    ]
    cv = result["metrics"]["validacion_cruzada_entrenamiento"]
    if cv:
        lines.append(f"  CV (entrenamiento, {cv['folds']} folds) -> F1-macro: "
                     f"{cv['f1_macro_mean']:.4f} (+/- {cv['f1_macro_std']:.4f})")
    if fatal:
        lines.append(f"  Clase fatal  -> precisión {fatal['precision']:.2f} | recall "
                     f"{fatal['recall']:.2f} | casos en prueba {fatal['support']}")
    lines += ["Archivos generados:"] + [f"  {k}: {v}" for k, v in result["paths"].items()]
    return "\n".join(lines)
