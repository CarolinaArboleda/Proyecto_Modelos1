"""Pruebas básicas: se ejecutan con `python -m pytest` desde fase-2 (sin red, con tmp_path)."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import METADATA_FILE, METRICS_FILE, MODEL_FILE
from src.data import prepare_features, read_csv
from src.errors import AppError
from src.prediction import predict_dataframe, predict_file
from src.preprocessing import TimeToHour
from src.storage import load_bundle
from src.training import train_model

from .conftest import make_synthetic_df

ROOT = Path(__file__).resolve().parents[1]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _new_data(tmp_path, n=25, **kwargs):
    df = make_synthetic_df(n=n, seed=99, with_target=False)
    path = tmp_path / "nuevos.csv"
    df.to_csv(path, index=False)
    return df, path


# 1. Entrenamiento y almacenamiento -------------------------------------------------
def test_entrenamiento_genera_artefacto_metricas_y_metadatos(train_args):
    res = train_model(version="v1", **train_args)
    vdir = Path(train_args["output_dir"]) / "v1"
    for name in (MODEL_FILE, METRICS_FILE, METADATA_FILE):
        assert (vdir / name).is_file()

    bundle = load_bundle(vdir)                       # artefacto cargable
    assert bundle["version"] == "v1"
    metrics = json.loads((vdir / METRICS_FILE).read_text(encoding="utf-8"))
    assert 0.0 <= metrics["modelo"]["f1_macro"] <= 1.0
    assert "f1_macro" in metrics["modelo_base"]       # el modelo base también se evalúa
    meta = json.loads((vdir / METADATA_FILE).read_text(encoding="utf-8"))
    assert meta["version"] == "v1" and meta["semilla"] == 7
    assert meta["datos"]["n_observaciones"] == 300 and len(meta["datos"]["sha256"]) == 64
    assert "Time" in meta["variables_entrada"] and "Casualty_severity" not in meta["variables_entrada"]
    assert meta["separacion"]["grupos_compartidos_train_test"] == 0
    assert res["metadata"]["entorno"]["python"]


# 2. Carga y predicción -------------------------------------------------------------
def test_modelo_guardado_predice_igual_que_el_modelo_en_memoria(train_args, tmp_path):
    res = train_model(version="v1", **train_args)
    df, new_csv = _new_data(tmp_path)

    out_csv = tmp_path / "salida" / "pred.csv"
    summary = predict_file(Path(train_args["output_dir"]) / "v1" / MODEL_FILE, new_csv, out_csv)

    out = pd.read_csv(out_csv)
    assert len(out) == len(df)                                        # una predicción por fila
    assert out["Num"].tolist() == list(range(1, len(df) + 1))         # orden conservado
    X, _ = prepare_features(read_csv(new_csv), res["bundle"]["schema"])
    esperado = res["bundle"]["pipeline"].predict(X)
    assert out["prediccion"].tolist() == esperado.tolist()
    assert set(out["prediccion"]) <= {"Slight Injury", "Serious Injury", "Fatal injury"}
    assert summary["n_filas"] == len(df)


def test_probabilidades_opcionales_suman_uno(train_args, tmp_path):
    train_model(version="v1", **train_args)
    _, new_csv = _new_data(tmp_path, n=10)
    out_csv = tmp_path / "p.csv"
    predict_file(Path(train_args["output_dir"]) / "v1", new_csv, out_csv, with_proba=True)
    out = pd.read_csv(out_csv)
    proba = out[[c for c in out.columns if c.startswith("proba_")]]
    assert proba.shape[1] == 3
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-4)


# 3. Valores faltantes --------------------------------------------------------------
def test_nulos_en_variables_que_los_admiten_se_imputan(train_args, tmp_path):
    train_model(version="v1", **train_args)
    bundle = load_bundle(Path(train_args["output_dir"]) / "v1")
    df = make_synthetic_df(n=6, seed=5, with_target=False)
    df.loc[0, ["Defect_of_vehicle", "Service_year_of_vehicle", "Work_of_casuality"]] = np.nan
    df.loc[1, "Number_of_casualties"] = np.nan        # numérica nula -> mediana del entrenamiento
    df.loc[2, "Time"] = np.nan                        # hora nula -> mediana del entrenamiento

    out, _ = predict_dataframe(bundle, df)
    assert len(out) == 6 and out["prediccion"].notna().all()   # ninguna fila se pierde

    # El nulo equivale exactamente a la categoría "Desconocido" aprendida en la imputación.
    explicit = df.copy()
    explicit.loc[0, ["Defect_of_vehicle", "Service_year_of_vehicle", "Work_of_casuality"]] = "Desconocido"
    out_explicit, _ = predict_dataframe(bundle, explicit)
    assert out["prediccion"].tolist() == out_explicit["prediccion"].tolist()


# 4. Entrada inválida ---------------------------------------------------------------
def test_columna_requerida_ausente_da_error_claro(train_args, tmp_path):
    train_model(version="v1", **train_args)
    bundle = load_bundle(Path(train_args["output_dir"]) / "v1")
    df = make_synthetic_df(n=5, seed=3, with_target=False).drop(columns=["Light_conditions"])
    with pytest.raises(AppError, match="Light_conditions"):
        predict_dataframe(bundle, df)


def test_cli_con_columna_ausente_sale_con_codigo_distinto_de_cero_y_sin_archivo(train_args, tmp_path):
    train_model(version="v1", **train_args)
    df = make_synthetic_df(n=5, seed=3, with_target=False).drop(columns=["Weather_conditions"])
    bad_csv, out_csv = tmp_path / "malo.csv", tmp_path / "out" / "pred.csv"
    df.to_csv(bad_csv, index=False)
    proc = subprocess.run(
        [sys.executable, "predict.py", "--model", str(Path(train_args["output_dir"]) / "v1" / MODEL_FILE),
         "--input", str(bad_csv), "--output", str(out_csv)],
        cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "Weather_conditions" in proc.stderr and "Traceback" not in proc.stderr
    assert not out_csv.exists()                       # no aparenta resultados válidos


def test_modelo_inexistente_y_tipo_incompatible(train_args, tmp_path):
    with pytest.raises(AppError, match="No se encontró el modelo"):
        load_bundle(tmp_path / "no_existe")
    train_model(version="v1", **train_args)
    bundle = load_bundle(Path(train_args["output_dir"]) / "v1")
    df = make_synthetic_df(n=4, seed=3, with_target=False)
    df.loc[1, "Number_of_vehicles_involved"] = "muchos"
    with pytest.raises(AppError, match="Number_of_vehicles_involved"):
        predict_dataframe(bundle, df)
    df = make_synthetic_df(n=4, seed=3, with_target=False)
    df.loc[2, "Time"] = "mediodia"
    with pytest.raises(AppError, match="HH:MM:SS"):
        predict_dataframe(bundle, df)


# 5. Versionamiento -----------------------------------------------------------------
def test_segunda_version_conserva_la_primera_y_id_repetido_no_sobrescribe(train_args):
    out_dir = Path(train_args["output_dir"])
    train_model(version="v1", **train_args)
    hashes_v1 = {n: _sha(out_dir / "v1" / n) for n in (MODEL_FILE, METRICS_FILE, METADATA_FILE)}

    train_model(version="v2", **{**train_args, "seed": 8, "max_depth": 3})
    assert load_bundle(out_dir / "v1")["version"] == "v1"      # ambas se cargan
    assert load_bundle(out_dir / "v2")["version"] == "v2"
    assert {n: _sha(out_dir / "v1" / n) for n in hashes_v1} == hashes_v1   # v1 intacta

    with pytest.raises(AppError, match="ya existe"):
        train_model(version="v1", **train_args)
    assert {n: _sha(out_dir / "v1" / n) for n in hashes_v1} == hashes_v1   # sigue intacta
    assert sorted(p.name for p in out_dir.iterdir()) == ["v1", "v2"]       # sin residuos


# Pruebas adicionales ---------------------------------------------------------------
def test_el_orden_de_columnas_del_csv_no_importa_y_las_extra_se_ignoran(train_args):
    train_model(version="v1", **train_args)
    bundle = load_bundle(Path(train_args["output_dir"]) / "v1")
    df = make_synthetic_df(n=12, seed=11, with_target=False)
    base, _ = predict_dataframe(bundle, df)
    shuffled = df[list(reversed(df.columns))].assign(columna_rara="x")
    out, avisos = predict_dataframe(bundle, shuffled)
    assert out["prediccion"].tolist() == base["prediccion"].tolist()
    assert any("columna_rara" in a for a in avisos)


def test_categoria_no_vista_no_rompe_y_se_avisa(train_args):
    train_model(version="v1", **train_args)
    bundle = load_bundle(Path(train_args["output_dir"]) / "v1")
    df = make_synthetic_df(n=5, seed=2, with_target=False)
    df.loc[0, "Weather_conditions"] = "Tornado"
    out, avisos = predict_dataframe(bundle, df)
    assert len(out) == 5 and any("Tornado" in a for a in avisos)


def test_transformador_de_hora():
    X = pd.DataFrame({"Time": ["17:02:00", "00:15:00", None], "otra": [1, 2, 3]})
    out = TimeToHour().fit_transform(X)
    assert out["Hour"].iloc[:2].tolist() == [17.0, 0.0] and np.isnan(out["Hour"].iloc[2])
    assert "Time" not in out.columns


def test_ayuda_de_los_scripts():
    for script in ("train.py", "predict.py"):
        proc = subprocess.run([sys.executable, script, "--help"], cwd=ROOT, capture_output=True, text=True)
        assert proc.returncode == 0 and "--" in proc.stdout
