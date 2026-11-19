"""API REST de la Fase 3: consulta y entrenamiento del modelo de severidad de accidentes.

Reutiliza la lógica de la Fase 2 (carpeta src/): el mismo esquema de entrada, el mismo
preprocesamiento guardado dentro del Pipeline y las mismas funciones de entrenamiento.

Iniciar (desde fase-3):
    uvicorn api:app --host 0.0.0.0 --port 8000

Configuración por variables de entorno:
    MODELS_DIR     carpeta con las versiones guardadas          (def.: models)
    MODEL_VERSION  versión que usa /predict                     (def.: v1)
    DATA_PATH      CSV etiquetado que usa /train                (def.: data/train.csv)
    TRAIN_SEED     semilla de /train                            (def.: 42)
"""
import os
import re
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from src.config import MODEL_FILE
from src.errors import AppError
from src.prediction import predict_dataframe
from src.storage import load_bundle

app = FastAPI(
    title="API - Severidad de accidentes de tránsito",
    description="Fase 3: /health, /predict y /train sobre el modelo de la Fase 2.",
    version="3.0",
)


# --------------------------------------------------------------------------------------
# Configuración (se lee en cada solicitud para poder cambiarla sin tocar el código)
# --------------------------------------------------------------------------------------
def config():
    return {
        "models_dir": Path(os.environ.get("MODELS_DIR", "models")),
        "model_version": os.environ.get("MODEL_VERSION", "v1"),
        "data_path": Path(os.environ.get("DATA_PATH", "data/train.csv")),
        "seed": int(os.environ.get("TRAIN_SEED", "42")),
    }


def available_versions(models_dir):
    """Versiones guardadas: subcarpetas de models_dir que contienen modelo.joblib."""
    d = Path(models_dir)
    if not d.is_dir():
        return []
    return sorted(p.name for p in d.iterdir() if (p / MODEL_FILE).is_file())


# Las versiones no se sobrescriben nunca, así que se pueden guardar en memoria.
_bundle_cache = {}


def get_bundle(models_dir, version):
    key = str(Path(models_dir) / version)
    if key not in _bundle_cache:
        _bundle_cache[key] = load_bundle(Path(models_dir) / version)
    return _bundle_cache[key]


# --------------------------------------------------------------------------------------
# Esquema de entrada: las 30 variables predictoras de la Fase 2 (sin Num, sin
# Casualty_severity y sin la variable objetivo). Todas son obligatorias, pero aceptan
# null: el nulo se trata con la regla guardada en la Fase 2 (categórica -> "Desconocido",
# numérica u hora -> mediana del entrenamiento).
# --------------------------------------------------------------------------------------
EJEMPLO = {
    "Time": "6:00:00", "Day_of_week": "Tuesday", "Age_band_of_driver": "18-30",
    "Sex_of_driver": "Male", "Educational_level": "Junior high school",
    "Vehicle_driver_relation": "Employee", "Driving_experience": "Above 10yr",
    "Type_of_vehicle": "Lorry (11?40Q)", "Owner_of_vehicle": "Owner",
    "Service_year_of_vehicle": "Unknown", "Defect_of_vehicle": None,
    "Area_accident_occured": "Residential areas", "Lanes_or_Medians": "other",
    "Road_allignment": "Tangent road with flat terrain", "Types_of_Junction": "No junction",
    "Road_surface_type": "Asphalt roads", "Road_surface_conditions": "Dry",
    "Light_conditions": "Darkness - lights lit", "Weather_conditions": "Normal",
    "Type_of_collision": "Collision with roadside objects",
    "Number_of_vehicles_involved": 2, "Number_of_casualties": 1,
    "Vehicle_movement": "Going straight", "Casualty_class": "Driver or rider",
    "Sex_of_casualty": "Male", "Age_band_of_casualty": "18-30", "Work_of_casuality": None,
    "Fitness_of_casuality": "Normal", "Pedestrian_movement": "Not a Pedestrian",
    "Cause_of_accident": "Changing lane to the right",
}


class RegistroAccidente(BaseModel):
    """Un registro con las variables predictoras (texto para categorías, número para conteos)."""
    model_config = ConfigDict(extra="ignore", json_schema_extra={"examples": [EJEMPLO]})

    Time: Optional[str] = Field(..., pattern=r"^\d{1,2}:\d{2}:\d{2}$",
                                description="Hora HH:MM:SS (ej. 17:02:00)")
    Day_of_week: Optional[str]
    Age_band_of_driver: Optional[str]
    Sex_of_driver: Optional[str]
    Educational_level: Optional[str]
    Vehicle_driver_relation: Optional[str]
    Driving_experience: Optional[str]
    Type_of_vehicle: Optional[str]
    Owner_of_vehicle: Optional[str]
    Service_year_of_vehicle: Optional[str]
    Defect_of_vehicle: Optional[str]
    Area_accident_occured: Optional[str]
    Lanes_or_Medians: Optional[str]
    Road_allignment: Optional[str]
    Types_of_Junction: Optional[str]
    Road_surface_type: Optional[str]
    Road_surface_conditions: Optional[str]
    Light_conditions: Optional[str]
    Weather_conditions: Optional[str]
    Type_of_collision: Optional[str]
    Number_of_vehicles_involved: Optional[int] = Field(..., ge=0)
    Number_of_casualties: Optional[int] = Field(..., ge=0)
    Vehicle_movement: Optional[str]
    Casualty_class: Optional[str]
    Sex_of_casualty: Optional[str]
    Age_band_of_casualty: Optional[str]
    Work_of_casuality: Optional[str]
    Fitness_of_casuality: Optional[str]
    Pedestrian_movement: Optional[str]
    Cause_of_accident: Optional[str]


# --------------------------------------------------------------------------------------
# Errores de validación: 422 con un mensaje que nombra la variable con problema
# --------------------------------------------------------------------------------------
def _describe(err):
    loc = [str(x) for x in err.get("loc", []) if x != "body"]
    campo = ".".join(loc) if loc else "cuerpo de la solicitud"
    tipo = err.get("type", "")
    if tipo == "missing":
        return f"Falta la variable obligatoria '{campo}'."
    if tipo == "json_invalid":
        return "El cuerpo no es un JSON válido."
    if tipo == "model_attributes_type":
        return "El cuerpo debe ser un objeto JSON con las variables del registro."
    return f"Valor inválido en '{campo}': {err.get('msg')} (recibido: {err.get('input')!r})."


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    errores = [_describe(e) for e in exc.errors()]
    return JSONResponse(
        status_code=422,
        content=jsonable_encoder({"error": "Entrada inválida", "detalle": errores}),
    )


# --------------------------------------------------------------------------------------
# Rutas
# --------------------------------------------------------------------------------------
@app.get("/health")
def health():
    cfg = config()
    versiones = available_versions(cfg["models_dir"])
    return {
        "estado": "ok",
        "version_configurada": cfg["model_version"],
        "modelo_disponible": cfg["model_version"] in versiones,
        "versiones_disponibles": versiones,
    }


@app.post("/predict")
def predict(registro: RegistroAccidente):
    cfg = config()
    version = cfg["model_version"]
    try:
        bundle = get_bundle(cfg["models_dir"], version)
    except AppError as exc:
        raise HTTPException(status_code=503, detail={
            "error": f"El modelo configurado ('{version}') no está disponible.",
            "causa": str(exc),
            "versiones_disponibles": available_versions(cfg["models_dir"]),
            "solucion": "Inicie la API con MODEL_VERSION=<una versión disponible>.",
        })

    # Tabla de una fila con las columnas del esquema; null -> NaN (regla de la Fase 2).
    df = pd.DataFrame([registro.model_dump()], dtype=object)
    df = df.where(df.notna(), np.nan)
    try:
        out, avisos = predict_dataframe(bundle, df)
    except AppError as exc:
        raise HTTPException(status_code=422, detail={"error": "Entrada inválida",
                                                     "detalle": [str(exc)]})
    return {"prediccion": str(out["prediccion"].iloc[0]), "version": bundle["version"],
            "avisos": avisos}