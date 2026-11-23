"""Pruebas del registro y del script de monitoreo (Fase 4)."""
import csv
import json
from pathlib import Path

from fastapi.testclient import TestClient

import monitoreo
from api import app

ROOT = Path(__file__).resolve().parents[1]
REGISTRO_VALIDO = json.loads((ROOT / "ejemplo_registro.json").read_text(encoding="utf-8"))


def _filas(ruta):
    with open(ruta, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_dos_validas_y_una_invalida_dan_3_solicitudes_y_33_3_por_ciento(
        modelos_prueba, registro_temporal, capsys):
    client = TestClient(app)
    incompleto = {k: v for k, v in REGISTRO_VALIDO.items() if k != "Light_conditions"}
    assert client.post("/predict", json=REGISTRO_VALIDO).status_code == 200
    assert client.post("/predict", json=REGISTRO_VALIDO).status_code == 200
    assert client.post("/predict", json=incompleto).status_code == 422
    client.get("/health")                                   # no se registra

    filas = _filas(registro_temporal)
    assert list(filas[0]) == ["fecha_hora", "estado_http", "version", "prediccion"]
    assert [f["estado_http"] for f in filas] == ["200", "200", "422"]   # una fila por intento
    assert filas[0]["version"] == "v1" and filas[0]["prediccion"]
    assert filas[2]["version"] == "" and filas[2]["prediccion"] == ""

    assert monitoreo.main(["--registro", str(registro_temporal)]) == 0
    salida = capsys.readouterr().out
    assert "Total de solicitudes: 3" in salida
    assert "33,3 %" in salida


def test_modelo_no_disponible_se_registra_sin_version(modelos_prueba, registro_temporal,
                                                     monkeypatch):
    monkeypatch.setenv("MODEL_VERSION", "v99")
    assert TestClient(app).post("/predict", json=REGISTRO_VALIDO).status_code == 503
    filas = _filas(registro_temporal)
    assert len(filas) == 1 and filas[0]["estado_http"] == "503" and filas[0]["version"] == ""


def test_registro_vacio_o_inexistente_muestra_sin_solicitudes(tmp_path, capsys):
    assert monitoreo.main(["--registro", str(tmp_path / "no_existe.csv")]) == 0
    salida = capsys.readouterr().out
    assert "Total de solicitudes: 0" in salida and "sin solicitudes" in salida