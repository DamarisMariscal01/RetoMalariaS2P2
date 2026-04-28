import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. Constantes Físicas Reales
# ============================================================
K_E = 8.987e9  # Constante de Coulomb (N m^2 / C^2)
Q_TOTAL = 1e-9  # Carga total de 1 nanoCoulomb por placa
num_cargas = 100 
q_individual = Q_TOTAL / num_cargas # Carga por cada punto diferencial

# ============================================================
# 2. Parámetros de Placas (en metros)
# ============================================================
y_cargas_pos = np.linspace(-3.5, 3.5, num_cargas)
y_cargas_neg = np.linspace(-2.0, 2.0, num_cargas)

x_pos = -0.8
x_neg = 0.8

# ============================================================
# 3. Malla
# ============================================================
res = 0.04
x_vec = np.arange(-5, 5 + res, res)
y_vec = np.arange(-5, 5 + res, res)
x, y = np.meshgrid(x_vec, y_vec)

Ex, Ey, V = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)

# ============================================================
# 4. Cálculo físico con magnitudes reales
# ============================================================
min_r = 0.1  # Parámetro de suavizado para evitar divisiones por cero

def calcular_contribucion(x_placa, y_cargas, carga_q, signo=1):
    global Ex, Ey, V
    for yc in y_cargas:
        dx = x - x_placa
        dy = y - yc
        r2 = dx**2 + dy**2
        r = np.sqrt(r2)
        r = np.where(r < min_r, min_r, r) # Suavizado
        
        # Potencial: V = k * q / r
        V += signo * (K_E * carga_q) / r
        
        # Campo: E = k * q / r^2 (proyectado en x e y)
        # Usamos r^3 porque dx/r es el coseno del ángulo
        Ex += signo * (K_E * carga_q * dx) / r**3
        Ey += signo * (K_E * carga_q * dy) / r**3

# Ejecutar cálculos
calcular_contribucion(x_pos, y_cargas_pos, q_individual, signo=1)
calcular_contribucion(x_neg, y_cargas_neg, q_individual, signo=-1)

# ============================================================
# 5. Visualización
# ============================================================
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_facecolor("#c5c9cd")

# Normalización para visualización
# El potencial real puede ser de miles de Voltios, escalamos para el mapa de colores
V_max_plot = np.percentile(np.abs(V), 95) 
V_plot = np.clip(V, -V_max_plot, V_max_plot)

ax.imshow(V_plot,
          extent=[-5, 5, -5, 5],
          origin='lower',
          cmap=plt.cm.coolwarm,
          alpha=0.35,
          aspect='equal')

# Equipotenciales
niveles = np.linspace(-V_max_plot, V_max_plot, 21)
ax.contour(x, y, V, levels=niveles, colors="#ffffff", linewidths=0.8, alpha=0.5)

# Campo eléctrico (Streamplot usa la magnitud relativa)
ax.streamplot(x, y, Ex, Ey, color='#2b2b2b', linewidth=0.8, density=1.4, arrowstyle='->')

# Dibujo de Placas
ax.plot([x_pos, x_pos], [y_cargas_pos[0], y_cargas_pos[-1]], color='red', linewidth=6, zorder=5)
ax.plot([x_neg, x_neg], [y_cargas_neg[0], y_cargas_neg[-1]], color='blue', linewidth=6, zorder=5)

ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_title('Simulación con Constantes Reales (k_e y Carga en nC)')
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('campo_no_uniforme_realista.png', dpi=200)
plt.show()