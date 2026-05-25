## 4. Fase de Optimización y Exportación del Modelo

### 4.1. Módulo de Conversión a Formato ONNX

Este módulo se encarga de transformar el modelo entrenado nativo de TensorFlow (`.keras`) a un formato estándar de la industria denominado **ONNX** (*Open Neural Network Exchange*). El objetivo principal de esta conversión es garantizar la interoperabilidad del modelo, permitiendo su despliegue optimizado en plataformas de producción independientes del ecosistema de Python.

---

### Sobre el exportado a .ONNX

El formato `.keras` es excelente para el diseño y entrenamiento, pero suele ser pesado y lento para la inferencia en tiempo real. Al migrar a ONNX obtenemos las siguientes ventajas técnicas:

* **Independencia del Framework:** El modelo final puede ejecutarse sin necesidad de tener instalado TensorFlow o Keras en el entorno de producción.
* **Optimización de Velocidad:** Permite utilizar motores de ejecución de alto rendimiento como *ONNX Runtime*, que aceleran drásticamente el tiempo de respuesta (latencia) al analizar imágenes.
* **Portabilidad:** Facilita la integración del modelo en sistemas nativos escritos en C++, C#, entornos móviles (iOS/Android) o hardware embebido.

---

#### 1. Persistencia de Directorios (`os.makedirs`)
El script valida preventivamente la existencia del directorio de producción llamado `model`. En caso de no ser detectado en la raíz, el sistema lo genera automáticamente para evitar excepciones de tipo *FileNotFoundError* al momento de guardar el archivo binario.

#### 2. Definición de la Firma de Entrada 
Antes de realizar la traducción del modelo, es mandatorio congelar la estructura matemática de los datos que van a ingresar a la red. Esto se logra mediante un objeto de TensorFlow:

```python
spec = (tf.TensorSpec((1, 224, 224, 3), tf.float32, name="cam_input"),)
Este tensor define estrictamente los siguientes parámetros fijos para el motor de inferencia:
```

1 (Batch Size): Configurado en uno, ya que el modelo en producción procesará los fotogramas de la cámara de uno en uno (tiempo real), a diferencia del entrenamiento que usaba lotes de 32.

224, 224, 3: Dimensiones de la imagen en píxeles y sus canales RGB.

tf.float32: Precisión de punto flotante requerida por los pesos matemáticos del modelo.

name="cam_input": Mantiene el nombre exacto de la "puerta de entrada" exigido en la Fase 2.

#### 3. Traducción y Mapeo de Operadores (tf2onnx)
La conversión se ejecuta a través de la función tf2onnx.convert.from_keras, utilizando el parámetro opset=13.

 Nota: El Opset (Operator Set) define la versión de la especificación matemática de ONNX que se va a utilizar. El valor 13 asegura que todas las capas y funciones de activación complejas de MobileNetV2 (como las convoluciones en profundidad y los encadenamientos residuales) sean perfectamente legibles y traducidas sin pérdida de precisión.

#### 4. Serialización y Almacenamiento Físico
Dado que el modelo convertido (modelo_prototipo) se genera inicialmente como un objeto de estructura en la memoria del programa, el script utiliza el método .SerializeToString() para codificar toda la arquitectura y los pesos calculados en una cadena de bytes puros.

Finalmente, se abre un flujo de escritura binaria ("wb") para volcar estos datos al disco duro, creando el archivo físico definitivo: model/motos.onnx.
