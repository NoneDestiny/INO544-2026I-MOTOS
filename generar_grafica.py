import matplotlib.pyplot as plt
import os

#Datos de entrenamiento y validación (Datos obtenidos del entrenamiento en Entrenamiento.py)
epocas = range(1, 11)
acc = [0.9374, 0.9933, 0.9969, 0.9979, 0.9985, 0.9996, 0.9998, 0.9998, 1.0000, 1.0000]
loss = [0.1837, 0.0412, 0.0243, 0.0167, 0.0122, 0.0095, 0.0076, 0.0064, 0.0053, 0.0045]
val_acc = [0.9916, 0.9933, 0.9933, 0.9950, 0.9950, 0.9958, 0.9967, 0.9958, 0.9967, 0.9967]
val_loss = [0.0650, 0.0384, 0.0281, 0.0225, 0.0191, 0.0165, 0.0149, 0.0134, 0.0123, 0.0115]

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