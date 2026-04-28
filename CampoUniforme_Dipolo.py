"""
Simulación del campo eléctrico de un dipolo
Sistema: una carga positiva y una carga negativa
Objetivo: visualizar la dirección y magnitud del campo eléctrico
"""

# ============================================================
# 1. Importación de librerías
# ============================================================
# numpy -> cálculos numéricos
# matplotlib -> visualización del campo

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # backend para mostrar la gráfica
import matplotlib.pyplot as plt


# ============================================================
# 2. Definición de constantes físicas
# ============================================================
# Se usa k = 1 para simplificar el modelo (unidades arbitrarias)

k = 1      # constante de Coulomb simplificada
d = 1.0    # distancia de cada carga al origen


# ============================================================
# 3. Posición y valor de las cargas
# ============================================================
# Se define un dipolo:
# carga positiva a la izquierda
# carga negativa a la derecha

carga_pos = (-d, 0)
carga_neg = (d, 0)

q_pos = 1.0
q_neg = -1.0


# ============================================================
# 4. Creación de la malla
# ============================================================
# Se genera una red de puntos donde se calculará el campo

x = np.arange(-3, 3.2, 0.2)
y = np.arange(-3, 3.2, 0.2)

X, Y = np.meshgrid(x, y)


# ============================================================
# 5. Inicialización del campo eléctrico
# ============================================================
# Ex, Ey representan las componentes del campo

Ex = np.zeros_like(X)
Ey = np.zeros_like(Y)


# ============================================================
# 6. Cálculo del campo eléctrico (superposición)
# ============================================================
# Se aplica la ley de Coulomb:
# E = k*q*r / r^3
# Se suma la contribución de ambas cargas

for i in range(len(x)):
    for j in range(len(y)):

        # ----------------------------------------
        # Contribución de la carga positiva
        # ----------------------------------------
        rx_pos = X[i, j] - carga_pos[0]
        ry_pos = Y[i, j] - carga_pos[1]
        r_pos = np.sqrt(rx_pos**2 + ry_pos**2)

        # ----------------------------------------
        # Contribución de la carga negativa
        # ----------------------------------------
        rx_neg = X[i, j] - carga_neg[0]
        ry_neg = Y[i, j] - carga_neg[1]
        r_neg = np.sqrt(rx_neg**2 + ry_neg**2)

        # ----------------------------------------
        # Evitar división entre cero
        # ----------------------------------------
        if r_pos > 0.01:
            Ex[i, j] += k * q_pos * rx_pos / r_pos**3
            Ey[i, j] += k * q_pos * ry_pos / r_pos**3

        if r_neg > 0.01:
            Ex[i, j] += k * q_neg * rx_neg / r_neg**3
            Ey[i, j] += k * q_neg * ry_neg / r_neg**3


# ============================================================
# 7. Normalización del campo
# ============================================================
# Se calcula la magnitud del campo y se normaliza
# para mejorar la visualización de las flechas

magnitud = np.sqrt(Ex**2 + Ey**2)

Ex_norm = Ex / magnitud
Ey_norm = Ey / magnitud


# ============================================================
# 8. Visualización del campo eléctrico
# ============================================================
# Se usa quiver para mostrar vectores del campo

plt.figure(figsize=(10, 8))

plt.quiver(
    X, Y,
    Ex_norm, Ey_norm,
    magnitud,              # color según intensidad
    cmap='viridis',
    alpha=0.8
)

# Barra de color para magnitud
plt.colorbar(label='Magnitud del campo eléctrico')


# ============================================================
# 9. Representación de las cargas
# ============================================================
# Se marcan con colores:
# rojo -> positivo
# azul -> negativo

plt.scatter(
    *carga_pos,
    color='red',
    s=200,
    edgecolors='black',
    zorder=5,
    label='Carga positiva (+)'
)

plt.scatter(
    *carga_neg,
    color='blue',
    s=200,
    edgecolors='black',
    zorder=5,
    label='Carga negativa (-)'
)


# ============================================================
# 10. Configuración de la gráfica
# ============================================================

plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Campo eléctrico de un dipolo (carga positiva y negativa)')

plt.grid(alpha=0.3)
plt.legend()

# Mantener proporciones correctas
plt.axis('equal')

# Límites del sistema
plt.xlim(-3, 3)
plt.ylim(-3, 3)


# ============================================================
# 11. Guardado y visualización
# ============================================================

plt.tight_layout()
plt.savefig('campo_electrico_dipolo.png', dpi=150)
plt.show()


# ============================================================
# 12. Interpretación física
# ============================================================
# Las líneas de campo salen de la carga positiva y entran a la negativa.
# Este comportamiento caracteriza a un dipolo eléctrico.
#
# La intensidad del campo es mayor cerca de las cargas,
# lo cual se observa en el mapa de colores.
#
# Este tipo de sistema es base para entender configuraciones
# más complejas como placas o campos no uniformes.

print("Gráfico del campo eléctrico generado")