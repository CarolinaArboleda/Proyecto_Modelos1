"""Pruebas de la API: se ejecutan con `python -m pytest` desde fase-3 (sin red ni modelos reales)."""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api import app

ROOT = Path(__file__).resolve().parents[1]
REGISTRO_VALIDO = json.loads((ROOT / "ejemplo_registro.json").read_text(encoding="utf-8"))
ETIQUETAS = {"Slight Injury", "Serious Injury", "Fatal injury"}


@pytest.fixture
def client():
    return TestClient(app)


# 1. /health responde 200 ------------------------------------------------------------
def test_health_responde_200(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["estado"] == "ok"


# 2. Entrada válida -> predicción y versión ------------------------------------------
def test_predict_entrada_valida_devuelve_prediccion_y_version(client, modelos_prueba):
    r = client.post("/predict", json=REGISTRO_VALIDO)
    assert r.status_code == 200
    body = r.json()
    assert {"prediccion", "version"} <= set(body)
    assert body["prediccion"] in ETIQUETAS
    assert body["version"] == "v1"


# 3. Falta una variable obligatoria -> rechazo con mensaje que la nombra -------------
def test_predict_sin_variable_obligatoria_se_rechaza(client, modelos_prueba):
    registro = {k: v for k, v in REGISTRO_VALIDO.items() if k != "Light_conditions"}
    r = client.post("/predict", json=registro)
    assert r.status_code == 422
    assert "Light_conditions" in " ".join(r.json()["detalle"])


# Pruebas adicionales -----------------------------------------------------------------
def test_predict_tipo_incompatible_se_rechaza(client, modelos_prueba):
    r = client.post("/predict", json={**REGISTRO_VALIDO, "Number_of_casualties": "muchos"})
    assert r.status_code == 422
    assert "Number_of_casualties" in " ".join(r.json()["detalle"])


def test_predict_modelo_no_disponible_devuelve_503(client, modelos_prueba, monkeypatch):
    monkeypatch.setenv("MODEL_VERSION", "v99")
    r = client.post("/predict", json=REGISTRO_VALIDO)
    assert r.status_code == 503
    assert "v1" in r.json()["detail"]["versiones_disponibles"]