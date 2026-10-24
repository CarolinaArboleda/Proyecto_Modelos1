"""Datos sintéticos pequeños con el esquema real del proyecto (sin descargas ni archivos reales)."""
import numpy as np
import pandas as pd
import pytest

CATEGORICAL = {
    "Day_of_week": ["Monday", "Tuesday", "Friday", "Sunday"],
    "Age_band_of_driver": ["18-30", "31-50", "Over 51", "Unknown"],
    "Sex_of_driver": ["Male", "Female", "Unknown"],
    "Educational_level": ["Above high school", "Junior high school", "Elementary school"],
    "Vehicle_driver_relation": ["Employee", "Owner", "Other"],
    "Driving_experience": ["1-2yr", "5-10yr", "Above 10yr", "Below 1yr"],
    "Type_of_vehicle": ["Automobile", "Lorry (41?100Q)", "Taxi", "Motorcycle"],
    "Owner_of_vehicle": ["Owner", "Governmental", "Organization"],
    "Service_year_of_vehicle": ["1-2yr", "2-5yrs", "Above 10yr"],
    "Defect_of_vehicle": ["No defect", "7", "5"],
    "Area_accident_occured": ["Residential areas", "Office areas", "Market areas", "Other"],
    "Lanes_or_Medians": ["Undivided Two way", "Two-way (divided)", "One way"],
    "Road_allignment": ["Tangent road with flat terrain", "Escarpments", "Gentle horizontal curve"],
    "Types_of_Junction": ["No junction", "Y Shape", "Crossing", "O Shape"],
    "Road_surface_type": ["Asphalt roads", "Earth roads", "Gravel roads"],
    "Road_surface_conditions": ["Dry", "Wet or damp", "Snow"],
    "Light_conditions": ["Daylight", "Darkness - lights lit", "Darkness - no lighting"],
    "Weather_conditions": ["Normal", "Raining", "Cloudy"],
    "Type_of_collision": ["Vehicle with vehicle collision", "Collision with pedestrians",
                          "Rollover"],
    "Vehicle_movement": ["Going straight", "Turnover", "Reversing", "U-Turn"],
    "Casualty_class": ["na", "Driver or rider", "Pedestrian", "Passenger"],
    "Sex_of_casualty": ["na", "Male", "Female"],
    "Age_band_of_casualty": ["na", "18-30", "31-50", "Under 18"],
    "Work_of_casuality": ["Driver", "Self-employed", "Student", "Other"],
    "Fitness_of_casuality": ["Normal", "Deaf", "Other"],
    "Pedestrian_movement": ["Not a Pedestrian", "Crossing from nearside", "In carriageway"],
    "Cause_of_accident": ["Moving Backward", "Overtaking", "No distancing", "Changing lane"],
}
NULLABLE = ["Defect_of_vehicle", "Service_year_of_vehicle", "Work_of_casuality",
            "Fitness_of_casuality", "Type_of_vehicle", "Driving_experience"]


def make_synthetic_df(n=300, seed=0, with_target=True):
    """DataFrame de texto (como lo lee read_csv) con las 33 columnas reales."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"Num": [str(i + 1) for i in range(n)]})
    hours = rng.integers(0, 24, n)
    minutes = rng.integers(0, 60, n)
    df["Time"] = [f"{h:02d}:{m:02d}:00" for h, m in zip(hours, minutes)]
    for col, values in CATEGORICAL.items():
        df[col] = rng.choice(values, n)
    df["Number_of_vehicles_involved"] = rng.integers(1, 5, n).astype(str)
    df["Number_of_casualties"] = rng.integers(1, 6, n).astype(str)
    df["Casualty_severity"] = rng.choice(["1", "2", "3", "na"], n)
    for col in NULLABLE:
        df.loc[rng.random(n) < 0.05, col] = np.nan
    if with_target:
        # Las tres clases bien representadas (proporciones fijas, escalan con n).
        n_fatal, n_serious = max(3, n // 15), max(6, n // 8)
        labels = np.array(["Slight Injury"] * (n - n_fatal - n_serious)
                          + ["Serious Injury"] * n_serious + ["Fatal injury"] * n_fatal)
        rng.shuffle(labels)
        df["Accident_severity"] = labels
    else:
        df = df.drop(columns=["Casualty_severity"])
    return df


@pytest.fixture
def synthetic_csv(tmp_path):
    path = tmp_path / "datos.csv"
    make_synthetic_df().to_csv(path, index=False)
    return path


@pytest.fixture
def train_args(synthetic_csv, tmp_path):
    """Parámetros mínimos para entrenar rápido (pocos árboles, sin CV)."""
    return dict(data_path=synthetic_csv, output_dir=tmp_path / "models", seed=7,
                n_estimators=15, max_depth=4, cv_folds=0, n_jobs=1, log=lambda m: None)
