## 7. Entorno de prueba TensorFlow

### 7.1. Verificación del Framework y Aceleración por Hardware (`check_env.py`)

Este módulo funciona como un script de diagnóstico inicial. Su objetivo es auditar el entorno de ejecución local para validar la correcta instalación del framework principal (**TensorFlow**) y mapear la disponibilidad de los recursos de hardware (CPU/GPU) asignados para las tareas de cómputo intensivo.

---

### Lógica y Diagnóstico del Sistema

El script realiza tres comprobaciones esenciales antes de permitir el despliegue de los pipelines de entrenamiento o aumento de datos:

#### 1. Validación de Enlace del Framework
Al ejecutar la importación nativa `import tensorflow as tf`, el sistema operativo comprueba las variables de entorno. Imprimir el mensaje *"¡TensorFlow instalado correctamente!"* actúa como una bandera de éxito en el flujo lógico, confirmando que las dependencias binarias están integradas de forma estable.

#### 2. Trazabilidad de Versión (`tf.__version__`)
Registrar la versión exacta instalada es un estándar crítico para la replicabilidad del proyecto. En proyectos de Deep Learning, las APIs de Keras suelen actualizarse o depreciarse rápidamente; documentar este valor previene errores de sintaxis y asegura la retrocompatibilidad del código con el servidor de producción.

#### 3. Auditoría de Hardware de Cómputo (`tf.config.list_physical_devices`)
El entrenamiento de redes neuronales convolucionales pesadas (como la base *MobileNetV2* utilizada en este proyecto) requiere una enorme potencia de cálculo matemático. La función `list_physical_devices()` interroga directamente al backend de TensorFlow para listar los recursos disponibles:

* **`CPU` (Central Processing Unit):** Dispositivo de procesamiento estándar presente en cualquier arquitectura.
* **`GPU` (Graphics Processing Unit):** Tarjetas gráficas dedicadas (como las que usan tecnología NVIDIA CUDA). Si TensorFlow detecta una GPU compatible en esta lista, el pipeline de Keras derivará de forma automática el cálculo de los tensores hacia ella, acelerando el entrenamiento hasta 10 veces en comparación con una CPU tradicional.
