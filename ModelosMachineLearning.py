"""
Clasificación de glóbulos rojos (sanos vs infectados)
Modelos: Random Forest y Red Neuronal (MLP)
Aplicación: Análisis de datos obtenidos de simulación de dielectroforesis
"""

# ============================================================
# 1. Importación de librerías
# ============================================================
# pandas -> manejo de datos en tablas
# numpy -> operaciones numéricas
# sklearn -> modelos de machine learning
# matplotlib -> visualización (opcional en este caso)

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, classification_report

import matplotlib.pyplot as plt


# ============================================================
# 2. Carga y limpieza de datos
# ============================================================
# Se carga el archivo generado previamente con la simulación

archivo_csv = "separacion_celular_RK4.csv"

try:
    df = pd.read_csv(archivo_csv)
    print(f"Archivo '{archivo_csv}' cargado correctamente.")
except FileNotFoundError:
    print("No se encontró el archivo.")
    exit()

# ------------------------------------------------------------
# Selección de variables
# ------------------------------------------------------------
# X -> características (features)
# y -> etiqueta (target)

# En este caso:
# 'X_final' representa la posición final del glóbulo
# 'Real' indica si está sano o infectado

X = df[['X_final']]
y = df['Real']


# ============================================================
# 3. División de datos
# ============================================================
# Se separan datos en:
# entrenamiento (80%) -> para aprender el modelo
# prueba (20%) -> para evaluar desempeño

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# ============================================================
# 4. Normalización de datos
# ============================================================
# Se aplica escalado para mejorar el rendimiento de la red neuronal

scaler = StandardScaler()

# Ajusta el escalador con datos de entrenamiento
X_train_scaled = scaler.fit_transform(X_train)

# Aplica la misma transformación a datos de prueba
X_test_scaled = scaler.transform(X_test)


# ============================================================
# 5. Modelo 1: Random Forest
# ============================================================
# Este modelo crea múltiples árboles de decisión
# Cada árbol "vota" por una clase y se toma la mayoría

modelo_rf = RandomForestClassifier(
    n_estimators=100,   # número de árboles
    random_state=42
)

# Entrenamiento
modelo_rf.fit(X_train, y_train)

# Predicción
pred_rf = modelo_rf.predict(X_test)

# Evaluación
acc_rf = accuracy_score(y_test, pred_rf)


# ============================================================
# 6. Modelo 2: Red Neuronal (MLP)
# ============================================================
# Multi-Layer Perceptron:
# Simula neuronas en capas para aprender patrones no lineales

modelo_mlp = MLPClassifier(
    hidden_layer_sizes=(10, 10),  # dos capas ocultas
    max_iter=1000,                # iteraciones de entrenamiento
    random_state=42
)

# Entrenamiento (usa datos escalados)
modelo_mlp.fit(X_train_scaled, y_train)

# Predicción
pred_mlp = modelo_mlp.predict(X_test_scaled)

# Evaluación
acc_mlp = accuracy_score(y_test, pred_mlp)


# ============================================================
# 7. Comparativa de modelos
# ============================================================
# Se imprimen métricas principales de desempeño

print("\n" + "="*30)
print("   COMPARATIVA DE MODELOS")
print("="*30)

print(f"Random Forest: {acc_rf*100:.2f}% de precisión")
print(f"Red Neuronal:  {acc_mlp*100:.2f}% de precisión")

print("="*30)


# ============================================================
# 8. Reportes detallados
# ============================================================
# Se muestran métricas como:
# precisión, recall y f1-score

print("\nDETALLE RANDOM FOREST:")
print(classification_report(y_test, pred_rf))

print("\nDETALLE RED NEURONAL:")
print(classification_report(y_test, pred_mlp))


# ============================================================
# 9. Interpretación de modelos (para exposición)
# ============================================================
"""
EXPLICACIÓN TEÓRICA:

1. Random Forest:
   - Funciona como un conjunto de árboles de decisión.
   - Cada árbol analiza la posición final del glóbulo.
   - La decisión final se toma por votación.
   - Es robusto y funciona bien con pocos datos.

2. Red Neuronal (MLP):
   - Simula el comportamiento de neuronas biológicas.
   - Aprende relaciones no lineales entre variables.
   - En este caso, detecta patrones complejos en el desplazamiento.
   - Requiere normalización para funcionar correctamente.

INTERPRETACIÓN FÍSICA:
- El modelo aprende que los glóbulos infectados se comportan diferente
  en el campo eléctrico (mayor o menor desplazamiento).
- Esto permite clasificarlos automáticamente.

CONCLUSIÓN:
- Ambos modelos pueden distinguir entre células sanas e infectadas.
- La elección del modelo depende de la complejidad del patrón y los datos disponibles.
"""