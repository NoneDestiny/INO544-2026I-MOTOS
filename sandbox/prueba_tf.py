import tensorflow as tf

print("¡TensorFlow instalado correctamente!")
print(f"Versión de TensorFlow: {tf.__version__}")

# Esto revisa si TensorFlow detecta tu procesador o tarjeta gráfica
dispositivos = tf.config.list_physical_devices()
print(f"Dispositivos detectados para entrenar: {dispositivos}")