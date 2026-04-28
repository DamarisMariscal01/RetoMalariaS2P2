"""
Visualización del modelo de dipolo inducido
Explica por qué células infectadas (mayor polarización) se desvían más
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.patches import Rectangle

def graficar_dipolo():
    """Visualiza el modelo de dipolo inducido en células"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Campo uniforme - sin fuerza neta
    ax1 = axes[0]
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_title('Campo Eléctrico Uniforme\nFuerza Neta = 0')
    
    # Flechas de campo uniforme
    for x in np.linspace(-1.5, 1.5, 7):
        ax1.arrow(x, 0.8, 0, -0.5, head_width=0.1, head_length=0.1, 
                  fc='blue', ec='blue', alpha=0.5)
        ax1.arrow(x, -0.8, 0, 0.5, head_width=0.1, head_length=0.1,
                  fc='blue', ec='blue', alpha=0.5)
    
    # Célula (círculo)
    celula = Circle((0, 0), 0.4, facecolor='lightgray', edgecolor='black', linewidth=2)
    ax1.add_patch(celula)
    
    # Dipolo inducido
    ax1.arrow(-0.3, 0, 0.6, 0, head_width=0.08, head_length=0.08,
              fc='red', ec='red', linewidth=2)
    ax1.text(-0.4, -0.1, '-', fontsize=16, ha='center', va='center', color='red')
    ax1.text(0.4, -0.1, '+', fontsize=16, ha='center', va='center', color='red')
    
    # Fuerzas
    ax1.arrow(-0.5, 0.5, 0, -0.3, head_width=0.08, head_length=0.08,
              fc='green', ec='green', linewidth=2)
    ax1.arrow(0.5, 0.5, 0, -0.3, head_width=0.08, head_length=0.08,
              fc='green', ec='green', linewidth=2)
    ax1.text(-0.6, 0.6, 'F', fontsize=12, color='green')
    ax1.text(0.55, 0.6, 'F', fontsize=12, color='green')
    ax1.text(0, -0.9, 'Fuerzas iguales y opuestas → Se cancelan', 
             fontsize=10, ha='center')
    
    # 2. Campo no uniforme - célula sana (dipolo débil)
    ax2 = axes[1]
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_title('Campo No Uniforme\nCélula Sana (dipolo débil)')
    
    # Campo más intenso a la derecha
    for x in np.linspace(-1.5, 1.5, 9):
        length = 0.3 + (x + 1.5) / 6
        ax2.arrow(x, 0.6, 0, -length, head_width=0.08, head_length=0.08,
                  fc='blue', ec='blue', alpha=0.6)
        ax2.arrow(x, -0.6, 0, length, head_width=0.08, head_length=0.08,
                  fc='blue', ec='blue', alpha=0.6)
    
    # Célula sana (dipolo pequeño)
    celula_sana = Circle((0, 0), 0.4, facecolor='lightgreen', edgecolor='darkgreen', linewidth=2)
    ax2.add_patch(celula_sana)
    
    # Dipolo pequeño
    ax2.arrow(-0.15, 0, 0.3, 0, head_width=0.06, head_length=0.06,
              fc='red', ec='red', linewidth=1.5)
    ax2.text(-0.22, -0.08, '-', fontsize=12, ha='center', va='center', color='red')
    ax2.text(0.22, -0.08, '+', fontsize=12, ha='center', va='center', color='red')
    
    ax2.text(0, -1.0, 'Pequeña desviación', fontsize=10, ha='center', color='darkgreen')
    
    # 3. Campo no uniforme - célula infectada (dipolo fuerte)
    ax3 = axes[2]
    ax3.set_xlim(-2, 2)
    ax3.set_ylim(-1.5, 1.5)
    ax3.set_title('Campo No Uniforme\nCélula Infectada (dipolo fuerte)')
    
    # Campo más intenso a la derecha
    for x in np.linspace(-1.5, 1.5, 9):
        length = 0.3 + (x + 1.5) / 6
        ax3.arrow(x, 0.6, 0, -length, head_width=0.08, head_length=0.08,
                  fc='blue', ec='blue', alpha=0.6)
        ax3.arrow(x, -0.6, 0, length, head_width=0.08, head_length=0.08,
                  fc='blue', ec='blue', alpha=0.6)
    
    # Célula infectada (dipolo grande)
    celula_inf = Circle((0, 0), 0.45, facecolor='lightcoral', edgecolor='darkred', linewidth=2)
    ax3.add_patch(celula_inf)
    
    # Dipolo grande
    ax3.arrow(-0.35, 0, 0.7, 0, head_width=0.1, head_length=0.1,
              fc='red', ec='red', linewidth=2.5)
    ax3.text(-0.45, -0.12, '-', fontsize=16, ha='center', va='center', color='red')
    ax3.text(0.45, -0.12, '+', fontsize=16, ha='center', va='center', color='red')
    
    # Fuerza neta
    ax3.arrow(0.2, 0.7, 0.4, 0, head_width=0.08, head_length=0.08,
              fc='orange', ec='orange', linewidth=3)
    ax3.text(0.7, 0.75, 'Fuerza Neta →', fontsize=11, color='orange')
    
    ax3.text(0, -1.0, 'Gran desviación lateral', fontsize=10, ha='center', color='darkred')
    
    # Explicación general
    plt.suptitle('Principio de Dielectroforesis para Detección de Malaria', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('dipolo_inducido.png', dpi=150)
    plt.show()

def graficar_gradiente():
    """Visualiza el gradiente del campo eléctrico"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Gradiente de |E|^2
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, y)
    
    # Simular campo de un electrodo pequeño
    E2 = 1 / (X**2 + (Y + 0.5)**2 + 0.1)
    grad_E2_x = np.gradient(E2, axis=1)
    grad_E2_y = np.gradient(E2, axis=0)
    
    # Normalizar
    mag = np.sqrt(grad_E2_x**2 + grad_E2_y**2)
    grad_x_norm = grad_E2_x / (mag + 1e-10)
    grad_y_norm = grad_E2_y / (mag + 1e-10)
    
    # Streamplot del gradiente
    step = 5
    ax1.streamplot(X[::step, ::step], Y[::step, ::step],
                   grad_x_norm[::step, ::step], grad_y_norm[::step, ::step],
                   color=mag[::step, ::step], cmap='hot', linewidth=1.5)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Gradiente de |E|²\nDirección de la fuerza DEP')
    ax1.set_aspect('equal')
    
    # Perfil de gradiente
    y_profile = np.linspace(-1, 1, 100)
    grad_profile = np.gradient(1 / (0**2 + (y_profile + 0.5)**2 + 0.1))
    
    ax2.plot(y_profile, grad_profile, 'b-', linewidth=2)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Posición y')
    ax2.set_ylabel('∇|E|²')
    ax2.set_title('Perfil del Gradiente')
    ax2.grid(alpha=0.3)
    ax2.fill_between(y_profile, 0, grad_profile, where=(grad_profile > 0), 
                      color='red', alpha=0.3, label='Fuerza hacia electrodo')
    ax2.fill_between(y_profile, 0, grad_profile, where=(grad_profile < 0),
                      color='blue', alpha=0.3, label='Fuerza opuesta')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('gradiente_campo.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    graficar_dipolo()
    graficar_gradiente()