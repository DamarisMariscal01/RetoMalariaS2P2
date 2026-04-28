import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv

# ============================================================
# 1. Parámetros físicos (Modelo Dipolar)
# ============================================================
ke = 1.0     # Constante dieléctrica ficticia para estabilidad
qe = 0.05    # Carga de los electrodos
dq = 0.1    # Carga del dipolo inducido en la célula

NUM_GLOBULOS = 200
dt = 0.003   # Paso de tiempo pequeño para precisión
PASOS_MAX = 2000 # Límite de seguridad para el historial

# ============================================================
# 2. Electrodos Asimétricos
# ============================================================
num_cargas = 100

# Placa positiva (Roja, más grande, izquierda)
y_pos = np.linspace(-3.5, 3.5, num_cargas)
x_pos = np.full(num_cargas, -0.8)

# Placa negativa (Azul, más pequeña, derecha)
y_neg = np.linspace(-2.0, 2.0, num_cargas)
x_neg = np.full(num_cargas, 0.8)

# ============================================================
# 3. Definición de Glóbulos
# ============================================================
# Centrados en X entre -0.2 y 0.2 (cerca de x=0)
x0 = np.random.uniform(-0.2, 0.2, NUM_GLOBULOS)
y_inicio = y_neg[-1] 
y0 = np.full(NUM_GLOBULOS, y_inicio)

vx0 = np.zeros(NUM_GLOBULOS)
vy0 = np.full(NUM_GLOBULOS, -2.5) # Velocidad inicial hacia abajo

# Estados y sensibilidad (dx_dipolo)
# 100 sanos (verde) y 100 infectados (naranja)
estados = ["Sano"] * 100 + ["Infectada"] * 100
dx_dipolo = np.array([0.03] * 100 + [0.09] * 100)

historial_x = [[] for _ in range(NUM_GLOBULOS)]
historial_y = [[] for _ in range(NUM_GLOBULOS)]

resultados = []

# ============================================================
# 4. Simulación física (Ciclo WHILE para no cortar el loop)
# ============================================================
print("Iniciando simulación física...")

longitud_maxima_historial = 0

for i in range(NUM_GLOBULOS):
    xe, ye = x0[i], y0[i]
    vx, vy = vx0[i], vy0[i]
    dx = dx_dipolo[i]
    
    contador_pasos = 0
    
    # --- CAMBIO 2: Condición de parada WHILE ---
    # La partícula sigue mientras no haya caído por debajo de las placas
    # y no superemos un límite de seguridad de pasos.
    while ye > -4.5 and contador_pasos < PASOS_MAX:
        
        historial_x[i].append(xe)
        historial_y[i].append(ye)
        
        Fx, Fy = 0, 0
        
        # Modelo Dipolar: Interacción con cada carga de las placas
        for k in range(num_cargas):
            # Distancias (con min_r=0.1 para estabilidad)
            rnp = max(np.sqrt((xe - dx - x_pos[k])**2 + (ye - y_pos[k])**2), 0.1)
            rnn = max(np.sqrt((xe - dx - x_neg[k])**2 + (ye - y_neg[k])**2), 0.1)
            rpp = max(np.sqrt((xe + dx - x_pos[k])**2 + (ye - y_pos[k])**2), 0.1)
            rpn = max(np.sqrt((xe + dx - x_neg[k])**2 + (ye - y_neg[k])**2), 0.1)

            # Fuerzas de Coulomb desglosadas (como pediste)
            Fpp_m = ke * dq * qe / rpp**2
            Fpn_m = ke * dq * qe / rpn**2
            Fnn_m = ke * dq * qe / rnn**2
            Fnp_m = ke * dq * qe / rnp**2

            # Componente X (Suma vectorial)
            Fx += ( -Fpn_m * ((xe + dx - x_neg[k]) / rpn)
                    -Fpp_m * ((xe + dx - x_pos[k]) / rpp)
                    +Fnn_m * ((xe - dx - x_neg[k]) / rnn)
                    +Fnp_m * ((xe - dx - x_pos[k]) / rnp) )

            # Componente Y
            Fy += ( -Fpn_m * ((ye - y_neg[k]) / rpn)
                    -Fpp_m * ((ye - y_pos[k]) / rpp)
                    +Fnn_m * ((ye - y_neg[k]) / rnn)
                    +Fnp_m * ((ye - y_pos[k]) / rnp) )

        # Arrastre (viscosidad) y Gravedad
        Fx -= 0.8 * vx
        Fy -= 0.8 * vy + 2.0  # El +2.0 es la fuerza de caída

        # Integración de Euler
        vx += Fx * dt
        vy += Fy * dt
        xe += vx * dt
        ye += vy * dt
        
        contador_pasos += 1

    if contador_pasos > longitud_maxima_historial:
        longitud_maxima_historial = contador_pasos

    # ============================================================
    # 5. CLASIFICACIÓN FINAL
    # ============================================================
    xe_final = xe

    if xe_final > 0:
        clasificacion = "Infectada"
    else:
        clasificacion = "Sana"

    real = estados[i]

    resultados.append([i, real, clasificacion, xe_final])

print("Simulación física terminada. Generando animación...")

# ============================================================
# 6. ANÁLISIS
# ============================================================
correctos = 0

for r in resultados:
    if r[1] == r[2]:
        correctos += 1

accuracy = correctos / NUM_GLOBULOS

print("\nRESULTADOS:")
print(f"Total: {NUM_GLOBULOS}")
print(f"Correctos: {correctos}")
print(f"Precisión: {accuracy*100:.2f}%")

# ============================================================
# 7. GUARDAR CSV
# ============================================================
with open("resultados_globulos.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["ID", "Real", "Predicho", "X_final"])
    writer.writerows(resultados)

print("CSV guardado.")

# ============================================================
# 8. Visualización y Estética
# ============================================================
fig, ax = plt.subplots(figsize=(8, 10))
ax.set_facecolor("#c5c9cd") # Fondo gris

# Dibujar Electrodos (Glow y Sólido)
for x, yr, col in [( -0.8, y_pos, "red"), (0.8, y_neg, "blue")]:
    ax.plot([x]*2, [yr[0], yr[-1]], color=col, lw=12, alpha=0.2, zorder=1) # Glow
    ax.plot([x]*2, [yr[0], yr[-1]], color=col, lw=6, solid_capstyle='round', zorder=2) # Sólido

# Preparar líneas de trayectoria y cabezas de partículas
colores = ["#2ecc71" if e == "Sano" else "#e67e22" for e in estados]
lineas = [ax.plot([], [], color=c, lw=2, alpha=0.6, zorder=3)[0] for c in colores]
heads = ax.scatter(x0, y0, c=colores, s=70, edgecolors="white", zorder=5)

# Detalles de la gráfica
ax.set_xlim(-4, 4)
ax.set_ylim(-5, 5) # Ajustado para ver el inicio y el final
ax.set_title("Animación: Modelo Dipolar Completo", fontsize=14, fontweight='bold')
ax.set_xticks([]); ax.set_yticks([]) # Limpieza visual
ax.spines[:].set_visible(False)
ax.grid(alpha=0.2)

# ============================================================
# 9. Animación
# ============================================================
def actualizar(frame):
    posiciones_actuales = []
    
    for i, linea in enumerate(lineas):
        # Usamos clamp para no salirnos del historial de cada partícula
        idx = min(frame, len(historial_x[i]) - 1)
        
        linea.set_data(historial_x[i][:idx], historial_y[i][:idx])
        
        # Si la partícula ya terminó, se queda en su última posición
        if idx >= 0:
            posiciones_actuales.append([historial_x[i][idx], historial_y[i][idx]])
        else:
            # Posición de seguridad si el historial está vacío (no debería pasar)
            posiciones_actuales.append([x0[i], y0[i]])

    heads.set_offsets(posiciones_actuales)
    return lineas + [heads]

# Animamos basándonos en la partícula que más tardó en caer
anim = FuncAnimation(fig, actualizar, frames=longitud_maxima_historial, interval=20, blit=True)

plt.show()