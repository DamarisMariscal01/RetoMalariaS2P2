"""
Simulación de campo eléctrico no uniforme entre placas asimétricas
Modelo vectorizado para visualización de campo y potencial
Aplicación: Dielectroforesis (detección de malaria)
"""

# ============================================================
# 1. Importación de librerías
# ============================================================
# numpy -> cálculos numéricos vectorizados
# matplotlib -> visualización del campo eléctrico y potencial
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 2. Parámetros de las placas
# ============================================================
# Se define el número de cargas que representarán cada placa

num_cargas = 100

# Placa positiva (izquierda, más grande)
# Se distribuyen cargas a lo largo del eje Y
y_cargas_pos = np.linspace(-3.5, 3.5, num_cargas)

# Placa negativa (derecha, más pequeña)
y_cargas_neg = np.linspace(-2.0, 2.0, num_cargas)

# Posición en el eje X de cada placa
x_pos = -0.8
x_neg = 0.8

# Esta diferencia de tamaños genera un campo no uniforme


# ============================================================
# 3. Creación de la malla
# ============================================================
# Se define la región del plano donde se calculará el campo

res = 0.04  # resolución de la malla

x_vec = np.arange(-5, 5 + res, res)
y_vec = np.arange(-5, 5 + res, res)

# meshgrid genera matrices 2D de coordenadas
x, y = np.meshgrid(x_vec, y_vec)

# Inicialización de variables
# Ex, Ey -> componentes del campo eléctrico
# V -> potencial eléctrico
Ex = np.zeros_like(x)
Ey = np.zeros_like(x)
V  = np.zeros_like(x)


# ============================================================
# 4. Cálculo físico (modelo vectorizado)
# ============================================================
# Se aplica el principio de superposición
# El campo total es la suma de todas las contribuciones

# Se define una distancia mínima para evitar singularidades
min_r = 0.1

# ------------------------------------------------------------
# 4.1 Contribución de la placa positiva
# ------------------------------------------------------------
for yc in y_cargas_pos:

    # Vector posición desde la carga al punto de evaluación
    dx = x - x_pos
    dy = y - yc

    # Distancia
    r = np.sqrt(dx**2 + dy**2)

    # Evita división entre cero o valores muy pequeños
    r = np.where(r < min_r, min_r, r)

    # Potencial eléctrico (proporcional a 1/r)
    V  += 1.0 / r

    # Campo eléctrico (proporcional a 1/r³)
    Ex += dx / r**3
    Ey += dy / r**3


# ------------------------------------------------------------
# 4.2 Contribución de la placa negativa
# ------------------------------------------------------------
for yc in y_cargas_neg:

    dx = x - x_neg
    dy = y - yc
    r = np.sqrt(dx**2 + dy**2)

    r = np.where(r < min_r, min_r, r)

    # Se resta porque las cargas son negativas
    V  -= 1.0 / r
    Ex -= dx / r**3
    Ey -= dy / r**3


# ============================================================
# 5. Visualización base
# ============================================================
# Se crea la figura y se define un fondo neutro

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_facecolor("#c5c9cd")


# ============================================================
# 6. Visualización del potencial eléctrico
# ============================================================
# Se usa un mapa de colores para representar el potencial
# Se limita el rango para evitar saturación visual
V_limit = np.max(np.abs(V)) * 0.5
V_plot = np.clip(V, -V_limit, V_limit)

ax.imshow(
    V_plot,
    extent=[-5, 5, -5, 5],
    origin='lower',
    cmap=plt.cm.coolwarm,
    alpha=0.35,
    aspect='equal'
)


# ============================================================
# 7. Líneas equipotenciales
# ============================================================
# Se generan niveles logarítmicos para mejor distribución visual
niveles_pos = np.geomspace(V_limit*0.05, V_limit*0.9, 10)
niveles_neg = -niveles_pos[::-1]
niveles = np.concatenate([niveles_neg, niveles_pos])

# Líneas secundarias (más suaves)
ax.contour(
    x, y, V,
    levels=niveles,
    colors="#ffffff",
    linewidths=0.8,
    alpha=0.7
)

# Líneas principales (más destacadas)
ax.contour(
    x, y, V,
    levels=niveles[::2],
    colors="#e0e0e0",
    linewidths=1.8,
    alpha=1.0
)


# ============================================================
# 8. Campo eléctrico
# ============================================================
# Se usa streamplot para mostrar líneas de campo
ax.streamplot(
    x, y, Ex, Ey,
    color='#2b2b2b',
    linewidth=0.8,
    density=1.4,
    arrowstyle='->'
)


# ============================================================
# 9. Representación de las placas
# ============================================================
# Se dibujan con efecto "glow" para mejorar estética

# Glow placa positiva
ax.plot(
    [x_pos, x_pos],
    [y_cargas_pos[0], y_cargas_pos[-1]],
    color='red',
    linewidth=12,
    alpha=0.2,
    zorder=4
)

# Glow placa negativa
ax.plot(
    [x_neg, x_neg],
    [y_cargas_neg[0], y_cargas_neg[-1]],
    color='blue',
    linewidth=12,
    alpha=0.2,
    zorder=4
)

# Placas sólidas
ax.plot(
    [x_pos, x_pos],
    [y_cargas_pos[0], y_cargas_pos[-1]],
    color='red',
    linewidth=6,
    solid_capstyle='round',
    zorder=5
)

ax.plot(
    [x_neg, x_neg],
    [y_cargas_neg[0], y_cargas_neg[-1]],
    color='blue',
    linewidth=6,
    solid_capstyle='round',
    zorder=5
)


# ============================================================
# 10. Ajustes finales de la gráfica
# ============================================================
# Se limpia la visualización para un resultado más profesional

ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)

# Se eliminan ejes para enfoque visual
ax.set_xticks([])
ax.set_yticks([])
ax.spines[:].set_visible(False)

# Título
ax.set_title(
    'Campo Eléctrico y Equipotenciales entre Placas Asimétricas',
    fontsize=13
)


# ============================================================
# 11. Guardado y visualización
# ============================================================

plt.tight_layout()
plt.savefig('campo_no_uniforme_vectorizado.png', dpi=200)
plt.show()


# ============================================================
# 12. Interpretación física
# ============================================================
# El campo eléctrico es más intenso cerca de la placa pequeña,
# lo que genera un gradiente de campo.
#
# Este gradiente es fundamental en la dielectroforesis, ya que
# permite manipular y separar partículas como células infectadas.
#
# El uso de un modelo vectorizado mejora la eficiencia del cálculo
# y permite obtener resultados más suaves y continuos.