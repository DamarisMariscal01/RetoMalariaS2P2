"""
Simulación de separación dielectroforética de glóbulos rojos

Este modelo representa cómo partículas (glóbulos) se mueven dentro de un campo eléctrico
no uniforme generado por electrodos asimétricos.

FÍSICA CLAVE:
- En dielectroforesis (DEP), las partículas neutras desarrollan un dipolo inducido.
- Si el campo eléctrico NO es uniforme, aparece una fuerza neta.
- Esta fuerza depende de qué tan polarizable es la partícula.

IDEA DEL MODELO:
- Glóbulos sanos → menor polarización → menor desviación
- Glóbulos infectados → mayor polarización → mayor desviación lateral

RESULTADO:
- Se separan espacialmente → permite clasificación automática
"""

# ============================================================
# 1. LIBRERÍAS
# ============================================================
# numpy → cálculo numérico eficiente
# matplotlib → visualización + animación
# csv → exportar resultados para machine learning

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv


# ============================================================
# 2. PARÁMETROS FÍSICOS DEL MODELO
# ============================================================
# Aquí se definen constantes que controlan el comportamiento del sistema

ke = 1.0     
# Constante tipo Coulomb (simplificada)
# No usamos unidades reales → solo importa comportamiento relativo

qe = 0.20    
# "Carga" efectiva de los electrodos
# A mayor valor → campo más intenso → mayor separación

dq = 0.1     
# Intensidad del dipolo inducido en cada glóbulo
# Representa qué tan fácil se polariza la célula

NUM_GLOBULOS = 200   
# Número de partículas simuladas

dt = 0.01            
# Paso de tiempo
# Pequeño → más precisión (importante en RK4)

PASOS_MAX = 1200     
# Límite de iteraciones → evita loops infinitos


# ============================================================
# 3. GEOMETRÍA DEL SISTEMA
# ============================================================
# Los electrodos NO son iguales → esto genera campo no uniforme

num_cargas = 100

# Electrodo positivo (izquierda, MÁS grande)
y_pos = np.linspace(-3.5, 3.5, num_cargas)
x_pos = np.full(num_cargas, -0.8)

# Electrodo negativo (derecha, MÁS pequeño)
y_neg = np.linspace(-2.0, 2.0, num_cargas)
x_neg = np.full(num_cargas, 0.8)

"""
INTERPRETACIÓN:
- El electrodo pequeño genera mayor gradiente de campo
- Esto es lo que produce la fuerza dielectroforética
- Sin esta asimetría → no habría separación
"""


# ============================================================
# 4. FUNCIÓN DE ACELERACIÓN (FÍSICA CENTRAL)
# ============================================================
def calcular_aceleracion(xe, ye, vx, vy, dx):

    Fx, Fy = 0, 0

    r_min = 0.25  
    # "Softening"
    # Evita que la fuerza se vuelva infinita cuando r → 0
    # (problema clásico en simulaciones tipo Coulomb)

    for k in range(num_cargas):

        # ----------------------------------------------------
        # IDEA CLAVE:
        # Un dipolo = dos cargas separadas por dx
        # ----------------------------------------------------

        # Distancias a cada extremo del dipolo respecto a electrodos
        rnp = np.sqrt((xe - dx - x_pos[k])**2 + (ye - y_pos[k])**2) + r_min
        rnn = np.sqrt((xe - dx - x_neg[k])**2 + (ye - y_neg[k])**2) + r_min
        rpp = np.sqrt((xe + dx - x_pos[k])**2 + (ye - y_pos[k])**2) + r_min
        rpn = np.sqrt((xe + dx - x_neg[k])**2 + (ye - y_neg[k])**2) + r_min

        # Fuerzas tipo Coulomb
        # F ~ 1/r^2
        Fpp_m = ke * dq * qe / rpp**2
        Fpn_m = ke * dq * qe / rpn**2
        Fnn_m = ke * dq * qe / rnn**2
        Fnp_m = ke * dq * qe / rnp**2

        # ----------------------------------------------------
        # SUMA VECTORIAL DE FUERZAS
        # ----------------------------------------------------
        # Cada término representa interacción dipolo-electrodo

        Fx += (
            -Fpn_m * ((xe + dx - x_neg[k]) / rpn)
            -Fpp_m * ((xe + dx - x_pos[k]) / rpp)
            +Fnn_m * ((xe - dx - x_neg[k]) / rnn)
            +Fnp_m * ((xe - dx - x_pos[k]) / rnp)
        )

        Fy += (
            -Fpn_m * ((ye - y_neg[k]) / rpn)
            -Fpp_m * ((ye - y_pos[k]) / rpp)
            +Fnn_m * ((ye - y_neg[k]) / rnn)
            +Fnp_m * ((ye - y_pos[k]) / rnp)
        )

    # --------------------------------------------------------
    # FUERZAS DEL MEDIO (IMPORTANTE FÍSICAMENTE)
    # --------------------------------------------------------

    # Arrastre viscoso (fluido)
    # F_drag ~ -v
    ax = Fx - 2.5 * vx

    # Gravedad + arrastre vertical
    ay = Fy - 2.5 * vy - 1.2

    return ax, ay


# ============================================================
# 5. CONDICIONES INICIALES
# ============================================================

# Posición inicial (arriba del sistema)
x0 = np.random.uniform(-0.15, 0.15, NUM_GLOBULOS)
y0 = np.full(NUM_GLOBULOS, 3.0)

# Velocidades iniciales
vx0 = np.zeros(NUM_GLOBULOS)
vy0 = np.full(NUM_GLOBULOS, -1.0)

# Clasificación real
estados = ["Sano"] * 100 + ["Infectada"] * 100

# DIFERENCIA CLAVE DEL MODELO:
# dx representa qué tan polarizable es la célula
dx_dipolo = np.array([0.02]*100 + [0.15]*100)

"""
INTERPRETACIÓN:
- dx pequeño → poca interacción → casi no se desvía
- dx grande → mucha interacción → fuerte desviación
"""


# ============================================================
# 6. SIMULACIÓN (RUNGE-KUTTA 4)
# ============================================================
"""
¿Por qué RK4?

- Euler → rápido pero impreciso
- RK4 → mucho más estable y preciso

Aquí es importante porque:
→ estamos integrando fuerzas no lineales
→ pequeños errores crecen rápido
"""

historial_x = [[] for _ in range(NUM_GLOBULOS)]
historial_y = [[] for _ in range(NUM_GLOBULOS)]
resultados = []

print("Iniciando simulación...")

longitud_maxima = 0

for i in range(NUM_GLOBULOS):

    xe, ye = x0[i], y0[i]
    vx, vy = vx0[i], vy0[i]
    dx = dx_dipolo[i]

    pasos = 0

    while ye > -4.5 and pasos < PASOS_MAX:

        historial_x[i].append(xe)
        historial_y[i].append(ye)

        # ---------------- RK4 ----------------
        k1_vx, k1_vy = vx, vy
        k1_ax, k1_ay = calcular_aceleracion(xe, ye, vx, vy, dx)

        k2_vx = vx + 0.5*k1_ax*dt
        k2_vy = vy + 0.5*k1_ay*dt
        k2_ax, k2_ay = calcular_aceleracion(xe + 0.5*k1_vx*dt, ye + 0.5*k1_vy*dt, k2_vx, k2_vy, dx)

        k3_vx = vx + 0.5*k2_ax*dt
        k3_vy = vy + 0.5*k2_ay*dt
        k3_ax, k3_ay = calcular_aceleracion(xe + 0.5*k2_vx*dt, ye + 0.5*k2_vy*dt, k3_vx, k3_vy, dx)

        k4_vx = vx + k3_ax*dt
        k4_vy = vy + k3_ay*dt
        k4_ax, k4_ay = calcular_aceleracion(xe + k3_vx*dt, ye + k3_vy*dt, k4_vx, k4_vy, dx)

        # Actualización final
        xe += (dt/6)*(k1_vx + 2*k2_vx + 2*k3_vx + k4_vx)
        ye += (dt/6)*(k1_vy + 2*k2_vy + 2*k3_vy + k4_vy)

        vx += (dt/6)*(k1_ax + 2*k2_ax + 2*k3_ax + k4_ax)
        vy += (dt/6)*(k1_ay + 2*k2_ay + 2*k3_ay + k4_ay)

        pasos += 1

    if pasos > longitud_maxima:
        longitud_maxima = pasos

    # Clasificación basada en desplazamiento lateral
    resultados.append([
        i,
        estados[i],
        "Infectada" if xe > 0.3 else "Sana",
        xe
    ])


# ============================================================
# 7. EXPORTACIÓN DE DATOS
# ============================================================

with open("resultados_globulos.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["ID", "Real", "Predicho", "X_final"])
    writer.writerows(resultados)

"""
IMPORTANTE:
Este archivo se usa después para Machine Learning.
"""


# ============================================================
# 8. ANIMACIÓN
# ============================================================

fig, ax = plt.subplots(figsize=(8, 10))
ax.set_facecolor("#c5c9cd")

# Electrodos
for x, yr, col in [(-0.8, y_pos, "red"), (0.8, y_neg, "blue")]:
    ax.plot([x]*2, [yr[0], yr[-1]], color=col, lw=12, alpha=0.2)
    ax.plot([x]*2, [yr[0], yr[-1]], color=col, lw=6)

# Colores por tipo
colores = ["#2ecc71" if e == "Sano" else "#e67e22" for e in estados]

lineas = [ax.plot([], [], color=c, lw=2, alpha=0.5)[0] for c in colores]
heads = ax.scatter(x0, y0, c=colores, s=60, edgecolors="white")

ax.set_xlim(-4, 4)
ax.set_ylim(-5, 5)
ax.set_title("Separación Dielectroforética (RK4)")
ax.set_xticks([])
ax.set_yticks([])

def actualizar(f):
    pos = []
    for i, linea in enumerate(lineas):
        idx = min(f, len(historial_x[i]) - 1)
        linea.set_data(historial_x[i][:idx], historial_y[i][:idx])
        pos.append([historial_x[i][idx], historial_y[i][idx]])
    heads.set_offsets(pos)
    return lineas + [heads]

anim = FuncAnimation(fig, actualizar, frames=longitud_maxima, interval=15, blit=True)

plt.show()