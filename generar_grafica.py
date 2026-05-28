import matplotlib.pyplot as plt
import os

#Datos de entrenamiento y validación (Datos obtenidos del entrenamiento en Entrenamiento.py)
epocas = range(1, 11)
acc = [0.9448, 0.9931, 0.9955, 0.9976, 0.9984, 0.9992, 0.9996, 0.9998, 0.9998, 0.9998]
loss = [0.1594, 0.0399, 0.0243, 0.0173, 0.0127, 0.0101, 0.0081, 0.0068, 0.0057, 0.0049]
val_acc = [0.9894, 0.9903, 0.9935, 0.9935, 0.9935, 0.9935, 0.9935, 0.9935, 0.9943, 0.9943]
val_loss = [0.0564, 0.0428, 0.0288, 0.0231, 0.0211, 0.0197, 0.0183, 0.0169, 0.0159, 0.0158]

# 2. Configurar la figura (Dos gráficas lado a lado)
plt.figure(figsize=(12, 5))

# --- Gráfica 1: Precisión (Accuracy) ---
plt.subplot(1, 2, 1)
plt.plot(epocas, acc, 'b-', label='Entrenamiento (Train)', linewidth=2)
plt.plot(epocas, val_acc, 'g-', label='Validación (Test)', linewidth=2)
plt.title('Curva de Precisión (Accuracy)')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.grid(True)

# --- Gráfica 2: Pérdida (Loss) ---
plt.subplot(1, 2, 2)
plt.plot(epocas, loss, 'b-', label='Entrenamiento (Train)', linewidth=2)
plt.plot(epocas, val_loss, 'r-', label='Validación (Test)', linewidth=2)
plt.title('Curva de Pérdida (Loss)')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.grid(True)

# 3. Crear la carpeta "src" si no existe
if not os.path.exists('src'):
    os.makedirs('src')

# 4. Guardar la imagen de alta calidad
ruta_guardado = os.path.join('src', 'grafica_rendimiento.png')
plt.tight_layout()
plt.savefig(ruta_guardado, dpi=300)

print(f"¡Éxito! Gráfica generada y guardada en: {ruta_guardado}")
plt.show() # Esto también abrirá una ventana con la gráfica para verla directamente.