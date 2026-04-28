import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv
from pathlib import Path

# ============================================================
# 1. Constantes y Parámetros Físicos
# ============================================================
K_E = 8.987e9  # Constante de Coulomb
Q_TOTAL = 1e-5  # Carga total por placa
num_cargas = 500 # Puntos de carga por placa
q_individual = Q_TOTAL / num_cargas

x_pos, x_neg = -0.8, 0.8
y_cargas_pos = np.linspace(-3.5, 3.5, num_cargas)
y_cargas_neg = np.linspace(-2.0, 2.0, num_cargas)

dt = 0.01
num_globulos = 40
pasos_trayectoria = 1200

# Parámetros de simulación visual
velocidad_caida_base = 1.35
gravedad_simulada = 0.65
radio_suavizado = 0.35 # Evita que la aceleración explote al tocar las placas

# Límites de estabilidad (Clamping)
aceleracion_max_x = 0.40
aceleracion_max_y = 1.5
velocidad_max_x = 0.40
velocidad_max_y = 4.0

jitter_inicial = 0.035
np.random.seed(42)

# ============================================================
# 2. Campo Eléctrico (AMBAS PLACAS)
# ============================================================
def calcular_campo(px, py):
    ex = np.zeros_like(px)
    ey = np.zeros_like(py)

    # Contribución de la placa POSITIVA (repele cargas positivas)
    for yc in y_cargas_pos:
        dx = px - x_pos
        dy = py - yc
        r = np.sqrt(dx**2 + dy**2)
        r = np.maximum(r, radio_suavizado)
        ex += (K_E * q_individual * dx) / r**3
        ey += (K_E * q_individual * dy) / r**3

    # Contribución de la placa NEGATIVA (atrae cargas positivas)
    for yc in y_cargas_neg:
        dx = px - x_neg
        dy = py - yc
        r = np.sqrt(dx**2 + dy**2)
        r = np.maximum(r, radio_suavizado)
        ex -= (K_E * q_individual * dx) / r**3
        ey -= (K_E * q_individual * dy) / r**3

    return ex, ey

# ============================================================
# 3. Integración RK4 (Optimizada)
# ============================================================
def derivadas(px, py, vx, vy, q_sobre_m):
    ex, ey = calcular_campo(px, py)

    # Aceleración producida por el campo eléctrico + gravedad simulada
    ax = q_sobre_m * ex
    ay = (q_sobre_m * ey) - gravedad_simulada

    # Clamping para estabilidad numérica
    ax = np.clip(ax, -aceleracion_max_x, aceleracion_max_x)
    ay = np.clip(ay, -aceleracion_max_y, aceleracion_max_y)

    return vx, vy, ax, ay

def rk4_step(px, py, vx, vy, dt, q_sobre_m):
    k1 = derivadas(px, py, vx, vy, q_sobre_m)
    k2 = derivadas(px + 0.5*dt*k1[0], py + 0.5*dt*k1[1], vx + 0.5*dt*k1[2], vy + 0.5*dt*k1[3], q_sobre_m)
    k3 = derivadas(px + 0.5*dt*k2[0], py + 0.5*dt*k2[1], vx + 0.5*dt*k2[2], vy + 0.5*dt*k2[3], q_sobre_m)
    k4 = derivadas(px + dt*k3[0], py + dt*k3[1], vx + dt*k3[2], vy + dt*k3[3], q_sobre_m)

    px_new = px + (dt / 6) * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
    py_new = py + (dt / 6) * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
    vx_new = vx + (dt / 6) * (k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
    vy_new = vy + (dt / 6) * (k1[3] + 2*k2[3] + 2*k3[3] + k4[3])

    return px_new, py_new, vx_new, vy_new

# ============================================================
# 4. Inicialización (Lógica Biológica)
# ============================================================
dx_celular = np.random.uniform(0.01, 0.1, num_globulos)
es_sana = dx_celular < 0.05

# Relación carga/masa ajustada para que la influencia de las dos placas sea visible pero controlada
q_sobre_m_base = 2.5e-6 
q_sobre_m = np.where(es_sana, q_sobre_m_base * 0.5, q_sobre_m_base * 1.5)

# Ligeras variaciones para evitar que parezcan un "enjambre" clonado
q_sobre_m *= np.random.uniform(0.9, 1.1, num_globulos)
velocidad_caida = velocidad_caida_base * np.random.uniform(0.9, 1.1, num_globulos)

colores = np.where(es_sana, '#2ecc71', '#e67e22')  # Verde = Sana, Naranja = Infectada
etiquetas = np.where(es_sana, 'Sana', 'Infectada')

# Posiciones con "jitter" (desviación estándar pequeña) centradas arriba
px = np.random.normal(0.0, jitter_inicial, num_globulos)
py = np.random.normal(2.0, jitter_inicial, num_globulos)

vx = np.random.normal(0.0, 0.03, num_globulos)
vy = -velocidad_caida

activa = np.ones(num_globulos, dtype=bool)

caminos_x = [[px[i]] for i in range(num_globulos)]
caminos_y = [[py[i]] for i in range(num_globulos)]

# ============================================================
# 5. Configuración Visual
# ============================================================
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_facecolor("#c5c9cd")
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_xlabel("Posición X")
ax.set_ylabel("Posición Y")
ax.grid(alpha=0.2)
ax.set_aspect('equal')

# Dibujo de placas con Z-order
ax.plot([x_pos, x_pos], [-3.5, 3.5], color='red', lw=12, alpha=0.2, zorder=3)
ax.plot([x_neg, x_neg], [-2.0, 2.0], color='blue', lw=12, alpha=0.2, zorder=3)
ax.plot([x_pos, x_pos], [-3.5, 3.5], color='red', lw=6, solid_capstyle='round', zorder=4)
ax.plot([x_neg, x_neg], [-2.0, 2.0], color='blue', lw=6, solid_capstyle='round', zorder=4)

# Preparar elementos móviles
lineas = []
for i in range(num_globulos):
    linea, = ax.plot([], [], color=colores[i], lw=1.8, alpha=0.7, zorder=5)
    lineas.append(linea)

scatter = ax.scatter(px, py, c=colores, s=90, edgecolors='white', linewidths=1.0, zorder=6)

# Leyenda fantasma
ax.scatter([], [], c='#2ecc71', s=80, edgecolors='white', label='Célula Sana')
ax.scatter([], [], c='#e67e22', s=80, edgecolors='white', label='Célula Infectada')
ax.legend(loc='upper right', framealpha=0.85)

# ============================================================
# 6. Bucle de Animación
# ============================================================
def update(frame):
    global px, py, vx, vy, activa

    if not np.any(activa):
        return lineas + [scatter]

    px[activa], py[activa], vx[activa], vy[activa] = rk4_step(
        px[activa], py[activa], vx[activa], vy[activa], dt, q_sobre_m[activa]
    )

    vx[:] = np.clip(vx, -velocidad_max_x, velocidad_max_x)
    vy[:] = np.clip(vy, -velocidad_max_y, velocidad_max_y)

    activa = (py > -4.95) & (py < 4.95) & (np.abs(px) < 4.95)

    for i in range(num_globulos):
        if activa[i]:
            caminos_x[i].append(px[i])
            caminos_y[i].append(py[i])
        lineas[i].set_data(caminos_x[i], caminos_y[i])

    scatter.set_offsets(np.c_[px, py])
    ax.set_title(f"Separación Celular por Electroforesis | Activas: {np.sum(activa)}/{num_globulos}")

    return lineas + [scatter]

ani = FuncAnimation(fig, update, frames=pasos_trayectoria, interval=20, blit=False)

# ============================================================
# 7. Guardado Seguro de CSV
# ============================================================
csv_guardado = False

def guardar_csv():
    global csv_guardado
    if csv_guardado:
        return

    nombre_csv = 'separacion_celular_RK4.csv'
    rutas_posibles = [
        Path(__file__).parent / nombre_csv if '__file__' in globals() else Path(nombre_csv),
        Path.home() / 'Downloads' / nombre_csv
    ]

    for ruta in rutas_posibles:
        try:
            with open(ruta, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['ID', 'dx', 'X_final', 'Y_final', 'Vx_final', 'Vy_final', 'Estado', 'Pasos'])
                for i in range(num_globulos):
                    writer.writerow([i+1, round(dx_celular[i], 4), round(px[i], 3), round(py[i], 3), 
                                     round(vx[i], 3), round(vy[i], 3), etiquetas[i], len(caminos_x[i])])
            csv_guardado = True
            print(f"✅ CSV guardado exitosamente en: {ruta}")
            return
        except OSError:
            continue
    print("❌ No se pudo guardar el archivo CSV por falta de permisos.")

fig.canvas.mpl_connect('close_event', lambda event: guardar_csv())
plt.show()

if __name__ == "__main__":
    guardar_csv()