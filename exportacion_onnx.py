import tensorflow as tf
import tf2onnx
import os

# 1. Aqui se crea la carpeta model de no existir.
if not os.path.exists('model'):
    os.makedirs('model')

# 2. Se carga el modelo de IA entrenado.
print("Cargando el modelo Keras")
modelo_keras = tf.keras.models.load_model('modelo_motos_IUJO.keras')

# 3. Se define la firma de entrada (Input Signature) 
# Tensor de Entrada: [1, 224, 224, 3] tipo float32
spec = (tf.TensorSpec((1, 224, 224, 3), tf.float32, name="cam_input"),)

# 4. Aca se traduce el modelo Keras a ONNX. 
print("Convirtiendo a formato ONNX")
modelo_prototipo, _ = tf2onnx.convert.from_keras(modelo_keras, input_signature=spec, opset=13)

# 5. Se guarda el archivo final.
ruta_salida = os.path.join('model', 'motos.onnx')
with open(ruta_salida, "wb") as f:
    f.write(modelo_prototipo.SerializeToString())

print(f"¡Éxito total! Modelo exportado y guardado en: {ruta_salida}")