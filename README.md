# Proyecto_Modelos1

# Fase 1 — Modelo Predictivo de Severidad de Accidentes de Tránsito

**Proyecto Integrador — Modelos y Simulación de Sistemas I (2026-II)**

## Integrantes del equipo
- _Nombre integrante 1_
- _Nombre integrante 2_
- _Nombre integrante 3_

## Descripción del problema
Se busca predecir la **severidad de un accidente de tránsito** (leve, grave o fatal) a partir
de las condiciones del conductor, el vehículo, la vía, el clima y la víctima involucrada,
utilizando datos reales de accidentes registrados en Addis Abeba, Etiopía. Es un problema de
**clasificación multiclase** sobre un conjunto de datos fuertemente desbalanceado (~86% heridas
leves, ~13% graves, ~1% fatales).

## Fuente del conjunto de datos
`data/train.csv` — conjunto de datos público de accidentes de tránsito de Addis Abeba
(Etiopía), 8.210 observaciones y 33 variables originales, donde cada fila corresponde a una
víctima (*casualty*) dentro de un accidente.

## Objetivo del modelo
Clasificar cada registro según el nivel de severidad del accidente (`Accident_severity`:
`Slight Injury`, `Serious Injury`, `Fatal injury`), como base para futuras fases del proyecto
(scripts ejecutables, API REST y monitoreo).

## Algoritmo utilizado
- **Modelo base (baseline):** `DummyClassifier` (estrategia `most_frequent`).
- **Modelo predictivo:** `RandomForestClassifier` (`n_estimators=400`, `max_depth=7`,
  `class_weight="balanced"`), dentro de un `Pipeline` de scikit-learn que incluye imputación de
  valores faltantes (categoría `"Desconocido"` para variables categóricas, mediana para
  numéricas) y codificación One-Hot de variables categóricas.

## Métrica empleada
**F1-macro**, elegida por el fuerte desbalance de clases (la accuracy por sí sola es engañosa,
ya que un modelo trivial que siempre prediga "Slight Injury" ya obtiene ~86% de accuracy). Se
reportan también accuracy y el reporte de clasificación completo por clase.

## Principales resultados obtenidos

| Modelo | Accuracy | F1-macro |
|---|---|---|
| Modelo base (Dummy) | 0.8624 | 0.3087 |
| Random Forest (modelo predictivo) | 0.8240 | **0.4113** |

- Validación cruzada estratificada (5 folds) sobre el conjunto de entrenamiento:
  F1-macro promedio de **0.3938** (± 0.0375), consistente con el resultado sobre el conjunto de
  prueba.
- El modelo predictivo mejora al modelo base en **+0.1026** de F1-macro, logrando identificar
  parcialmente las clases minoritarias (`Serious Injury`, `Fatal injury`), algo que el modelo
  base no logra en absoluto.
- Se identificó y evitó una fuente de **fuga de información**: la variable `Casualty_severity`
  fue excluida de las variables predictoras por ser, a nivel de víctima, prácticamente
  equivalente a la variable objetivo.

Más detalle (EDA, justificación de decisiones, matriz de confusión, importancia de variables,
interpretación de resultados) en `notebook.ipynb`.

## Estructura de la carpeta
```
fase-1/
├── notebook.ipynb      # Notebook completo, ejecutado de principio a fin
├── modelo.joblib        # Pipeline entrenado (preprocesamiento + modelo), listo para reutilizar
├── data/
│   └── train.csv         # Conjunto de datos utilizado
└── README.md
```

## Instrucciones para ejecutar el notebook

1. Clonar el repositorio y ubicarse en la carpeta `fase-1/`.
2. Crear un entorno virtual e instalar las dependencias:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn joblib jupyter
   ```
3. Asegurarse de que el archivo `data/train.csv` esté presente (ya incluido en el repositorio).
4. Abrir y ejecutar el notebook de principio a fin:
   ```bash
   jupyter notebook notebook.ipynb
   ```
   El notebook es completamente reproducible (semilla fija `random_state=42`) y no requiere
   intervención manual.

### Reutilizar el modelo ya entrenado sin volver a entrenar
```python
import joblib
modelo = joblib.load("modelo.joblib")
predicciones = modelo.predict(nuevos_datos)  # nuevos_datos: DataFrame con las mismas columnas que X
```
