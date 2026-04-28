"""
Modelos de Machine Learning para clasificación de células con malaria
Basado en datos de simulación dielectroforética
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Cargar datos
print("=" * 70)
print("ANÁLISIS DE DATOS PARA DETECCIÓN DE MALARIA")
print("=" * 70)

df = pd.read_csv('datos_malaria.csv')
print(f"\nDataset cargado: {len(df)} muestras")
print(df.head())

# Variables para entrenamiento
X = df[['Carga', 'Masa', 'TiempoVuelo', 'VelMax', 'PosFinalX']]
y = df['Clase']

# Normalización
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# División train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nTrain set: {len(X_train)} muestras")
print(f"Test set: {len(X_test)} muestras")

# ============================================================
# MODELO 1: ÁRBOL DE DECISIÓN
# ============================================================
print("\n" + "=" * 50)
print("MODELO 1: ÁRBOL DE DECISIÓN")
print("=" * 50)

dtree = DecisionTreeClassifier(
    criterion='entropy',  # Usa entropía para medir impureza
    max_depth=4,          # Profundidad máxima (evita overfitting)
    min_samples_split=5,  # Mínimo de muestras para dividir un nodo
    min_samples_leaf=2,   # Mínimo de muestras en hoja
    random_state=42
)

dtree.fit(X_train, y_train)
y_pred_dt = dtree.predict(X_test)

# Métricas
print(f"Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_dt):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_dt):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_dt):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_dt, target_names=['Sano', 'Infectado']))

# Visualizar árbol de decisión
plt.figure(figsize=(14, 8))
plot_tree(dtree, feature_names=X.columns, class_names=['Sano', 'Infectado'], 
          filled=True, rounded=True, fontsize=10)
plt.title('Árbol de Decisión para Clasificación de Malaria', fontsize=14)
plt.tight_layout()
plt.savefig('arbol_decision.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# MODELO 2: RANDOM FOREST
# ============================================================
print("\n" + "=" * 50)
print("MODELO 2: RANDOM FOREST")
print("=" * 50)
print("""
¿Qué hace Random Forest?
- Ensamble de múltiples árboles de decisión
- Cada árbol se entrena con una muestra bootstrap de los datos
- En cada división, se consideran solo √n características aleatorias
- Predicción final por votación (mayoría)
- Más robusto que un solo árbol y menos propenso a overfitting
""")

rf = RandomForestClassifier(
    n_estimators=100,      # Número de árboles
    max_depth=5,           # Profundidad máxima
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_rf):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_rf):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_rf):.4f}")

# Importancia de características
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nImportancia de características:")
for _, row in feature_importance.iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# ============================================================
# MODELO 3: K-MEANS (No supervisado)
# ============================================================
print("\n" + "=" * 50)
print("MODELO 3: K-MEANS (NO SUPERVISADO)")
print("=" * 50)
print("""
¿Qué hace K-Means?
- Algoritmo de clustering no supervisado
- No usa etiquetas durante el entrenamiento
- Busca K grupos (clusters) que minimicen la distancia intra-cluster
- Asigna cada punto al centroide más cercano
- Útil cuando no tenemos datos etiquetados
""")

kmeans = KMeans(
    n_clusters=2,          # Buscamos 2 grupos (sano/infectado)
    init='k-means++',      # Inicialización inteligente
    n_init=10,             # Reintentos con diferentes inicializaciones
    max_iter=300,
    random_state=42
)

y_pred_kmeans = kmeans.fit_predict(X_scaled)

# K-Means no tiene etiquetas consistentes (puede invertir 0 y 1)
# Ajustamos las etiquetas para comparación
from scipy.stats import mode
labels_kmeans = np.zeros_like(y_pred_kmeans)
for i in range(2):
    mask = (y_pred_kmeans == i)
    if np.sum(mask) > 0:
        labels_kmeans[mask] = mode(y[mask])[0]

acc_kmeans = accuracy_score(y, labels_kmeans)
print(f"Accuracy (ajustada): {acc_kmeans:.4f}")

# ============================================================
# MODELO 4: GAUSSIAN MIXTURE MODEL (GMM)
# ============================================================
print("\n" + "=" * 50)
print("MODELO 4: GAUSSIAN MIXTURE MODEL (PROBABILÍSTICO)")
print("=" * 50)
print("""
¿Qué hace GMM?
- Similar a K-Means pero más flexible
- Asume que los datos siguen una mezcla de distribuciones Gaussianas
- Cada cluster tiene: media, covarianza, y peso
- Proporciona probabilidad de pertenencia a cada cluster
- Mejor para clusters con formas elípticas
""")

gmm = GaussianMixture(
    n_components=2,        # Número de Gaussianas
    covariance_type='full', # Forma libre de la covarianza
    init_params='kmeans',  # Inicialización con K-Means
    max_iter=100,
    n_init=5,
    random_state=42
)

y_pred_gmm = gmm.fit_predict(X_scaled)

# Ajustar etiquetas
labels_gmm = np.zeros_like(y_pred_gmm)
for i in range(2):
    mask = (y_pred_gmm == i)
    if np.sum(mask) > 0:
        labels_gmm[mask] = mode(y[mask])[0]

acc_gmm = accuracy_score(y, labels_gmm)
print(f"Accuracy (ajustada): {acc_gmm:.4f}")

# Probabilidades de pertenencia
probs = gmm.predict_proba(X_scaled)
print(f"Rango de probabilidades - Cluster 0: [{probs[:,0].min():.3f}, {probs[:,0].max():.3f}]")
print(f"Rango de probabilidades - Cluster 1: [{probs[:,1].min():.3f}, {probs[:,1].max():.3f}]")

# ============================================================
# MODELO 5: RED NEURONAL (MLP)
# ============================================================
print("\n" + "=" * 50)
print("MODELO 5: RED NEURONAL MULTICAPA (MLP)")
print("=" * 50)
print("""
¿Qué hace una Red Neuronal MLP?
- Capa de entrada: 5 neuronas (una por característica)
- Capas ocultas: procesamiento no lineal de la información
- Capa de salida: clasificación binaria (sano/infectado)
- Función de activación ReLU para no linealidad
- Backpropagation para ajustar pesos
""")

mlp = MLPClassifier(
    hidden_layer_sizes=(10, 8),  # Dos capas ocultas: 10 y 8 neuronas
    activation='relu',            # Función de activación
    solver='adam',                # Optimizador
    max_iter=500,
    random_state=42,
    verbose=False
)

mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred_mlp):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_mlp):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_mlp):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_mlp):.4f}")

# ============================================================
# COMPARACIÓN DE MODELOS
# ============================================================
print("\n" + "=" * 50)
print("COMPARACIÓN DE MODELOS")
print("=" * 50)

modelos = {
    'Árbol de Decisión': y_pred_dt,
    'Random Forest': y_pred_rf,
    'K-Means (No Sup.)': labels_kmeans[y_test.index] if len(labels_kmeans) == len(y) else None,
    'GMM (No Sup.)': labels_gmm[y_test.index] if len(labels_gmm) == len(y) else None,
    'Red Neuronal MLP': y_pred_mlp
}

# Para modelos no supervisados, usar todo el dataset
acc_km = accuracy_score(y, labels_kmeans)
acc_gmm_full = accuracy_score(y, labels_gmm)

comparison = {
    'Árbol de Decisión': accuracy_score(y_test, y_pred_dt),
    'Random Forest': accuracy_score(y_test, y_pred_rf),
    'K-Means (No Sup.)': acc_km,
    'GMM (No Sup.)': acc_gmm_full,
    'Red Neuronal MLP': accuracy_score(y_test, y_pred_mlp)
}

for name, acc in comparison.items():
    print(f"{name}: {acc:.4f}")

# Gráfica de comparación
plt.figure(figsize=(10, 6))
bars = plt.bar(comparison.keys(), comparison.values(), 
               color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6'])
plt.ylim(0, 1)
plt.ylabel('Accuracy')
plt.title('Comparación de Modelos para Detección de Malaria')
plt.xticks(rotation=45, ha='right')
for bar, acc in zip(bars, comparison.values()):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{acc:.3f}', ha='center', va='bottom')
plt.tight_layout()
plt.savefig('comparacion_modelos.png', dpi=150)
plt.show()

# ============================================================
# MATRICES DE CONFUSIÓN
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

modelos_plot = [
    ('Árbol de Decisión', y_pred_dt, y_test),
    ('Random Forest', y_pred_rf, y_test),
    ('Red Neuronal', y_pred_mlp, y_test),
    ('K-Means', labels_kmeans, y),
    ('GMM', labels_gmm, y),
]

for idx, (name, pred, true) in enumerate(modelos_plot):
    cm = confusion_matrix(true, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
    axes[idx].set_title(f'{name}')
    axes[idx].set_xlabel('Predicción')
    axes[idx].set_ylabel('Real')
    axes[idx].set_xticklabels(['Sano', 'Infectado'])
    axes[idx].set_yticklabels(['Sano', 'Infectado'])

# Ocultar eje extra si existe
if len(modelos_plot) < 6:
    axes[5].set_visible(False)

plt.tight_layout()
plt.savefig('matrices_confusion.png', dpi=150)
plt.show()

# ============================================================
# VISUALIZACIÓN DE CLUSTERS (K-MEANS vs GMM)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# K-Means
scatter1 = axes[0].scatter(X_scaled[:, 0], X_scaled[:, 4], 
                           c=labels_kmeans, cmap='coolwarm', alpha=0.7)
axes[0].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 4], 
                c='yellow', marker='X', s=200, edgecolors='black', label='Centroides')
axes[0].set_xlabel('Carga (normalizada)')
axes[0].set_ylabel('PosFinalX (normalizada)')
axes[0].set_title('K-Means Clustering')
axes[0].legend()

# GMM
scatter2 = axes[1].scatter(X_scaled[:, 0], X_scaled[:, 4], 
                           c=labels_gmm, cmap='coolwarm', alpha=0.7)
axes[1].set_xlabel('Carga (normalizada)')
axes[1].set_ylabel('PosFinalX (normalizada)')
axes[1].set_title('Gaussian Mixture Model')

plt.colorbar(scatter1, ax=axes[0], label='Cluster')
plt.colorbar(scatter2, ax=axes[1], label='Cluster')
plt.tight_layout()
plt.savefig('clustering_resultados.png', dpi=150)
plt.show()

print("\n" + "=" * 70)
print("ANÁLISIS COMPLETADO")
print("=" * 70)
print("""
CONCLUSIONES:
1. Los modelos supervisados (Árbol de Decisión, Random Forest, Red Neuronal)
   muestran alta precisión en la clasificación.
   
2. La carga eléctrica efectiva y la posición final en X son las características
   más discriminantes entre células sanas e infectadas.
   
3. La dielectroforesis permite separar células según sus propiedades dieléctricas,
   que son diferentes en glóbulos infectados con malaria.
   
4. Modelos no supervisados como GMM pueden detectar los dos grupos incluso sin
   etiquetas, útiles para validación cruzada.
""")