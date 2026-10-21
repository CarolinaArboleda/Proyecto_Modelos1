"""Constantes del proyecto. Nada aquí lee archivos ni entrena modelos."""

DEFAULT_TARGET = "Accident_severity"

# Identificador de fila y variable de la víctima que casi equivale al objetivo
# (riesgo de fuga de información; ver README, sección de fuga).
DEFAULT_DROP_COLUMNS = ["Num", "Casualty_severity"]

TIME_COLUMN = "Time"   # texto HH:MM:SS -> se convierte en la variable numérica Hour
HOUR_COLUMN = "Hour"
NUMERIC_COLUMNS = ["Number_of_vehicles_involved", "Number_of_casualties"]

# Columnas que describen el *contexto del accidente* (no al conductor ni a la
# víctima). Filas que coinciden en todas ellas se tratan como posibles víctimas
# del mismo accidente y se mantienen juntas al separar train/test.
GROUP_COLUMNS = [
    "Time", "Day_of_week", "Area_accident_occured", "Lanes_or_Medians",
    "Road_allignment", "Types_of_Junction", "Road_surface_type",
    "Road_surface_conditions", "Light_conditions", "Weather_conditions",
    "Type_of_collision", "Number_of_vehicles_involved", "Number_of_casualties",
]

LABEL_ORDER = ["Slight Injury", "Serious Injury", "Fatal injury"]
LABEL_MEANING = {
    "Slight Injury": "herida leve",
    "Serious Injury": "herida grave",
    "Fatal injury": "accidente fatal",
}

MODEL_FILE = "modelo.joblib"
METRICS_FILE = "metrics.json"
METADATA_FILE = "metadata.json"

MISSING_CATEGORY = "Desconocido"   # imputación de categóricas faltantes

DEFAULT_DATA_SOURCE = (
    "Road Traffic Accident Dataset of Addis Ababa City (Bedane, 2020, Mendeley Data, "
    "doi:10.17632/xytv86278f.1; versión en Kaggle: saurabhshahane/road-traffic-accidents). "
    "Archivo usado: train.csv (8.210 filas), ver README."
)
