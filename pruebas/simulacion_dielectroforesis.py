"""
Simulación de dielectroforesis para separación de glóbulos rojos
Sanos vs Infectados con malaria
Basado en propiedades dieléctricas de estudios reales
"""

from matplotlib.patches import Rectangle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd

# Constantes
ke = 9e9  # N·m²/C²
dq = 1e-9  # Carga de cada punto en electrodos (C)
dt = 1e-7  # Paso de tiempo (s)
b = 0.4e-9  # Coeficiente de arrastre (N·s/m)

# Geometría
d_electrodo = 0.5e-3  # Separación entre electrodos (m)
t = 0.05e-3  # Grosor electrodo (m)
Lp = 1.6e-3  # Largo electrodo positivo (grande) - m
Ln = 0.4e-3  # Largo electrodo negativo (pequeño) - m

# Número de cargas por electrodo
Nq = 500

# Posiciones de cargas en electrodos
def crear_cargas_electrodos():
    """Crea las posiciones de las cargas puntuales en los electrodos"""
    yp = np.linspace(-Lp/2, Lp/2, Nq)
    xp = np.full(Nq, -d_electrodo/2 - t/2)
    
    yn = np.linspace(-Ln/2, Ln/2, Nq)
    xn = np.full(Nq, d_electrodo/2 + t/2)
    
    return xp, yp, xn, yn

xp, yp, xn, yn = crear_cargas_electrodos()

def fuerza_electrica(xe, ye, qe):
    """
    Calcula la fuerza eléctrica sobre una partícula
    qe: carga efectiva de la partícula (diferente para sanos/infectados)
    """
    Fx = 0
    Fy = 0
    
    # Interacción con electrodo negativo
    for k in range(Nq):
        rx = xe - xn[k]
        ry = ye - yn[k]
        r = np.sqrt(rx**2 + ry**2)
        if r > 1e-9:
            Fx += ke * (-dq) * qe * rx / r**3
            Fy += ke * (-dq) * qe * ry / r**3
    
    # Interacción con electrodo positivo
    for k in range(Nq):
        rx = xe - xp[k]
        ry = ye - yp[k]
        r = np.sqrt(rx**2 + ry**2)
        if r > 1e-9:
            Fx += ke * dq * qe * rx / r**3
            Fy += ke * dq * qe * ry / r**3
    
    return Fx, Fy

def simular_celula(x0, y0, qe, m, clase, dt=dt, max_steps=5000):
    """
    Simula la trayectoria de una célula
    Retorna: trayectoria, tiempo_vuelo, velocidad_max, pos_final_x
    """
    xe, ye = x0, y0
    
    # Velocidad inicial (flujo hacia abajo + pequeña variación)
    Vx = np.random.normal(0, 0.002)
    Vy = -0.02 + np.random.normal(0, 0.002)
    trayectoria = [(xe, ye)]
    t_vuelo = 0
    v_max = 0
    
    for step in range(max_steps):
        # Calcular fuerza
        Fx, Fy = fuerza_electrica(xe, ye, qe)
        
        # Fuerza de arrastre
        F_arrastre_x = -b * Vx
        F_arrastre_y = -b * Vy
        
        # Segunda Ley de Newton
        ax = (Fx + F_arrastre_x) / m
        ay = (Fy + F_arrastre_y) / m
        
        # Euler
        Vx += ax * dt
        Vy += ay * dt
        xe += Vx * dt
        ye += Vy * dt
        t_vuelo += dt
        
        # Actualizar velocidad máxima
        v_inst = np.sqrt(Vx**2 + Vy**2)
        v_max = max(v_max, v_inst)
        
        trayectoria.append((xe, ye))
        
        # Condiciones de salida
        if ye > 1.5e-3 or ye < -1.5e-3 or xe > 1.5e-3 or xe < -1.5e-3:
            break
    
    return trayectoria, t_vuelo, v_max, xe, clase

def generar_datos_celulas(n_celulas=200):
    """
    Genera dataset de células sanas e infectadas
    Basado en el estudio: células infectadas tienen mayor respuesta eléctrica
    """
    datos = []
    
    # Parámetros basados en el estudio (conversión a unidades consistentes)
    # Célula sana: qe_base ~ 0.8e-6 C (efectivo)
    # Célula infectada: qe_base ~ 1.4e-6 C (75% mayor)
    
    for i in range(n_celulas):
        # Posición inicial aleatoria
        # Lanzar desde el centro entre electrodos (x ≈ 0)
        x0 = np.random.uniform(-0.05e-3, 0.05e-3)
        y0 = np.random.uniform(-1.2e-3, 1.2e-3)
        
        # Determinar si es infectada (50% probabilidad)
        if np.random.rand() > 0.5:
            # Célula sana
            clase = 0
            qe_base = 0.8e-6
            m = 1.0e-15  # masa aproximada (kg)
            color = 'green'
        else:
            # Célula infectada - mayor respuesta eléctrica
            clase = 1
            # Basado en estudio: incremento de 50-100% en respuesta dieléctrica
            qe_base = 1.4e-6
            # Volumen ocupado por parásito afecta masa
            vol_parasito = 0.1 + np.random.rand() * 0.8
            m = 1.0e-15 + vol_parasito * 0.2e-15
            color = 'red'
        
        # Añadir ruido
        qe = qe_base + np.random.normal(0, 0.25e-6)
        qe = max(0.5e-6, min(2.0e-6, qe))  # Limitar
        
        # Simular
        tray, t_vuelo, v_max, x_final, _ = simular_celula(x0, y0, qe, m, clase)
        
        datos.append({
            'Carga': qe,
            'Masa': m,
            'TiempoVuelo': t_vuelo,
            'VelMax': v_max,
            'PosFinalX': x_final,
            'Clase': clase,
            'Trayectoria': tray
        })
        
        print(f"Célula {i+1}/{n_celulas}: {'INFECTADA' if clase else 'SANA'} | "
              f"q={qe:.2e} C | x_final={x_final*1e3:.2f} mm")
    
    return pd.DataFrame(datos)

def graficar_trayectorias(df, num_muestras=20):
    """Grafica las trayectorias de células sanas e infectadas"""
    plt.figure(figsize=(12, 8))
    
    # Filtrar muestras
    sanas = df[df['Clase'] == 0].head(num_muestras//2)
    infectadas = df[df['Clase'] == 1].head(num_muestras//2)
    
    # Graficar trayectorias de sanas
    for _, row in sanas.iterrows():
        tray = np.array(row['Trayectoria'])
        plt.plot(tray[:, 0]*1e3, tray[:, 1]*1e3, 'g-', alpha=0.5, linewidth=0.8)
        plt.scatter(tray[-1, 0]*1e3, tray[-1, 1]*1e3, c='green', s=30, alpha=0.7)
    
    # Graficar trayectorias de infectadas
    for _, row in infectadas.iterrows():
        tray = np.array(row['Trayectoria'])
        plt.plot(tray[:, 0]*1e3, tray[:, 1]*1e3, 'r-', alpha=0.5, linewidth=0.8)
        plt.scatter(tray[-1, 0]*1e3, tray[-1, 1]*1e3, c='red', s=30, alpha=0.7)
    
    # Dibujar electrodos
    from matplotlib.patches import Rectangle
    electrodo_pos = Rectangle(((-d_electrodo/2 - t/2 - 0.05e-3)*1e3, -Lp/2*1e3), t*1e3, Lp*1e3, facecolor='red', alpha=0.5)
    electrodo_neg = Rectangle(((d_electrodo/2 + t/2 - 0.05e-3)*1e3, -Ln/2*1e3), t*1e3, Ln*1e3, facecolor='blue', alpha=0.5)
    
    plt.gca().add_patch(electrodo_pos)
    plt.gca().add_patch(electrodo_neg)
    
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.title('Trayectorias de glóbulos rojos\nVerde: Sanos | Rojo: Infectados con malaria')
    plt.xlim(-1, 1)
    plt.ylim(-1.5, 1.5)
    plt.grid(alpha=0.3)
    plt.legend(['Sanos', 'Infectados'], loc='upper right')
    plt.tight_layout()
    plt.savefig('trayectorias_celulas.png', dpi=150)
    plt.show()

# Ejecutar simulación
if __name__ == "__main__":
    print("=" * 60)
    print("SIMULACIÓN DE DIELECTROFORESIS PARA DETECCIÓN DE MALARIA")
    print("=" * 60)
    print("\nGenerando datos de células...")
    
    df = generar_datos_celulas(n_celulas=200)
    
    # Guardar datos
    df[['Carga', 'Masa', 'TiempoVuelo', 'VelMax', 'PosFinalX', 'Clase']].to_csv(
        'datos_malaria.csv', index=False, header=['Carga', 'Masa', 'TiempoVuelo', 'VelMax', 'PosFinalX', 'Clase']
    )
    print("\n✓ Datos guardados en 'datos_malaria.csv'")
    
    # Estadísticas básicas
    print("\n--- ESTADÍSTICAS ---")
    print(f"Total células: {len(df)}")
    print(f"Células sanas: {len(df[df['Clase']==0])}")
    print(f"Células infectadas: {len(df[df['Clase']==1])}")
    print(f"\nCarga promedio - Sanas: {df[df['Clase']==0]['Carga'].mean():.2e} C")
    print(f"Carga promedio - Infectadas: {df[df['Clase']==1]['Carga'].mean():.2e} C")
    print(f"Posición final X promedio - Sanas: {df[df['Clase']==0]['PosFinalX'].mean()*1e3:.2f} mm")
    print(f"Posición final X promedio - Infectadas: {df[df['Clase']==1]['PosFinalX'].mean()*1e3:.2f} mm")
    
    # Graficar
    graficar_trayectorias(df)