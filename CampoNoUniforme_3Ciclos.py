"""
Simulación de campo eléctrico no uniforme entre electrodos de diferentes tamaños
Aplicación: Dielectroforesis para detección de malaria
"""

# ============================================================
# 1. Importación de librerías
# ============================================================
# numpy -> cálculos numéricos
# matplotlib -> visualización de resultados
# Rectangle -> para dibujar los electrodos

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ============================================================
# 2. Parámetros físicos
# ============================================================
# Se definen las constantes necesarias para el modelo físico

ke = 9e9      # Constante de Coulomb (N·m²/C²)
dq = 1e-9     # Carga de cada punto en los electrodos (C)
d = 1.6       # Separación entre electrodos (m)
t = 0.2       # Grosor del electrodo (m)


# ============================================================
# 3. Geometría de los electrodos
# ============================================================
# Se definen los tamaños de los electrodos
# Electrodo positivo: más grande
# Electrodo negativo: más pequeño
# Esto genera un campo eléctrico no uniforme

Lp = 7        # Largo electrodo positivo (m)
Ln = 4        # Largo electrodo negativo (m)
p = 0.01      # Factor para evitar cargas en los extremos


# ============================================================
# 4. Dominio espacial
# ============================================================
# Se define la región donde se calculará el campo eléctrico

xmin = -d/2
xmax = d/2

# El eje Y cubre completamente el electrodo más grande
ymin = -Lp/2
ymax = Lp/2

# Márgenes extra para visualización
margen_x = 4
margen_y = 2


# ============================================================
# 5. Discretización de cargas
# ============================================================
# Cada electrodo se modela como un conjunto de cargas puntuales

Nq = 30  # Número de cargas por electrodo


# ============================================================
# 6. Creación de la malla 2D
# ============================================================
# Se genera una malla de puntos donde se calculará el campo

Nx, Ny = 50, 50

x = np.linspace(xmin - margen_x, xmax + margen_x, Nx)
y = np.linspace(ymin - margen_y, ymax + margen_y, Ny)

# meshgrid crea matrices 2D para evaluar funciones en todo el plano
X, Y = np.meshgrid(x, y)


# ============================================================
# 7. Posición de las cargas en los electrodos
# ============================================================

# Electrodo positivo (izquierda)
# Se distribuyen las cargas a lo largo del electrodo grande
yp = np.linspace((1 - p) * ymin, (1 - p) * ymax, Nq)
xp = np.full(Nq, xmin - t/2)

# Electrodo negativo (derecha)
# Se distribuyen en el electrodo pequeño
yn = np.linspace(-(1 - p) * Ln/2, (1 - p) * Ln/2, Nq)
xn = np.full(Nq, xmax + t/2)


# ============================================================
# 8. Inicialización de variables
# ============================================================
# Se crean matrices para almacenar:
# Ex, Ey -> componentes del campo eléctrico
# V -> potencial eléctrico

Ex = np.zeros_like(X, dtype=float)
Ey = np.zeros_like(Y, dtype=float)
V = np.zeros_like(X, dtype=float)


# ============================================================
# 9. Cálculo del campo eléctrico (superposición)
# ============================================================
# Se aplica el principio de superposición:
# El campo total es la suma de los campos de cada carga puntual

for j in range(Ny):          # Recorre filas (eje Y)
    for i in range(Nx):      # Recorre columnas (eje X)

        # Punto actual de evaluación
        p_x = X[j, i]
        p_y = Y[j, i]

        # Se suman contribuciones de todas las cargas
        for k in range(Nq):

            # ----------------------------------------
            # Electrodo positivo
            # ----------------------------------------
            rx_p = p_x - xp[k]
            ry_p = p_y - yp[k]
            rp = np.sqrt(rx_p**2 + ry_p**2)

            # Evitar división entre cero
            if rp > 1e-12:
                # Campo eléctrico (Ley de Coulomb)
                Ex[j, i] += ke * dq * rx_p / rp**3
                Ey[j, i] += ke * dq * ry_p / rp**3

                # Potencial eléctrico
                V[j, i] += ke * dq / rp

            # ----------------------------------------
            # Electrodo negativo
            # ----------------------------------------
            rx_n = p_x - xn[k]
            ry_n = p_y - yn[k]
            rn = np.sqrt(rx_n**2 + ry_n**2)

            if rn > 1e-12:
                # Se considera la carga negativa
                Ex[j, i] += ke * (-dq) * rx_n / rn**3
                Ey[j, i] += ke * (-dq) * ry_n / rn**3
                V[j, i] += ke * (-dq) / rn


# ============================================================
# 10. Normalización del campo
# ============================================================
# Se calcula la magnitud del campo eléctrico
# Se normaliza para visualizar solo la dirección

magnitud = np.sqrt(Ex**2 + Ey**2)

Ex_dir = Ex / (magnitud + 1e-10)
Ey_dir = Ey / (magnitud + 1e-10)


# ============================================================
# 11. Función para dibujar electrodos
# ============================================================
# Se representan como rectángulos:
# rojo -> positivo
# azul -> negativo

def dibujar_electrodos(ax):
    electrodo_pos = Rectangle(
        (xmin - t, ymin), t, Lp,
        facecolor='red', alpha=0.7, label='Electrodo (+)'
    )

    electrodo_neg = Rectangle(
        (xmax, -Ln/2), t, Ln,
        facecolor='blue', alpha=0.7, label='Electrodo (-)'
    )

    ax.add_patch(electrodo_pos)
    ax.add_patch(electrodo_neg)


# ============================================================
# 12. Visualización
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))


# ============================================================
# 12.1 Campo eléctrico
# ============================================================
# Se usa streamplot para mostrar líneas de campo

ax1 = axes[0]

ax1.streamplot(
    x, y, Ex, Ey,
    color='black',
    density=2,
    linewidth=0.6,
    arrowstyle='->',
    arrowsize=0.8
)

dibujar_electrodos(ax1)

ax1.set_title('Campo eléctrico no uniforme')
ax1.set_xlabel('x (m)')
ax1.set_ylabel('y (m)')
ax1.set_xlim(xmin - margen_x, xmax + margen_x)
ax1.set_ylim(ymin - margen_y, ymax + margen_y)
ax1.set_aspect('equal')
ax1.grid(alpha=0.2)
ax1.legend(loc='upper right', fontsize='small')


# ============================================================
# 12.2 Potencial eléctrico
# ============================================================
# Se muestra como mapa de colores + equipotenciales

ax2 = axes[1]

im = ax2.pcolormesh(
    X, Y, V,
    cmap='coolwarm',
    shading='gouraud',
    alpha=0.8
)

plt.colorbar(im, ax=ax2, label='Potencial (V)')

ax2.contour(
    X, Y, V,
    levels=20,
    colors='white',
    linewidths=0.5,
    alpha=0.7
)

dibujar_electrodos(ax2)

ax2.set_title('Potencial y equipotenciales')
ax2.set_xlabel('x (m)')
ax2.set_ylabel('y (m)')
ax2.set_xlim(xmin - margen_x, xmax + margen_x)
ax2.set_ylim(ymin - margen_y, ymax + margen_y)
ax2.set_aspect('equal')


# ============================================================
# 13. Guardado y salida
# ============================================================

plt.tight_layout()
plt.savefig('campo_no_uniforme_3_ciclos.png', dpi=200)
plt.show()


# ============================================================
# 14. Interpretación física
# ============================================================
# El campo es más intenso cerca del electrodo pequeño debido a
# la concentración de líneas de campo.
# Esto genera un gradiente eléctrico, fundamental en la
# dielectroforesis para separar partículas (ej. células infectadas).

print("✓ Simulación de campo no uniforme completada")
print(f"  - Electrodo positivo (rojo) tamaño: {Lp*1e3:.2f} mm")
print(f"  - Electrodo negativo (azul) tamaño: {Ln*1e3:.2f} mm")
print("  - Campo más intenso cerca del electrodo pequeño")