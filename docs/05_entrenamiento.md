## 3. Fase 2 y 3: Configuración, Arquitectura y Entrenamiento del Modelo

### 3.1. Pipeline de Datos y Red Neuronal Convolucional

Este módulo abarca la preparación del pipeline de datos con TensorFlow, la instanciación de la arquitectura de la red mediante *Transfer Learning*, y el ciclo de entrenamiento final del clasificador binario. El sistema está optimizado para identificar si una imagen pertenece a la categoría de "Motos" o "No Motos".

---

### Fase 2: Pipeline de Datos Eficiente (`tf.data`)

El script utiliza la API de alto nivel `tf.keras.utils.image_dataset_from_directory` para estructurar el flujo de imágenes directamente desde el almacenamiento local sin saturar la memoria RAM.

#### 1. División del Dataset (Split de Validación)
Se establece un **`validation_split=0.2`**, lo que segmenta el dataset aumentado en dos subconjuntos independientes:
* **Entrenamiento (`train_dataset`):** Corresponde al 80% de los datos. Se utiliza para actualizar los pesos de la capa de salida.
* **Validación/Pruebas (`validation_dataset`):** Corresponde al 20% de los datos. Funciona como un control de calidad ("examen sorpresa") al final de cada época para evaluar la capacidad de generalización del modelo.

#### 2. Control de Aleatoriedad Estocástica (`seed=123`)
Para mitigar el riesgo de obtener métricas infladas o falsos positivos, se fija una semilla pseudoaleatoria (`seed=123`). Esto garantiza que el split de datos sea idéntico en cada ejecución, impidiendo que una imagen del grupo de validación se filtre en el grupo de entrenamiento si el script se reinicia.

#### 3. Carga por Lotes (Batching)
El parámetro `batch_size=32` define que Keras procesará los tensores en bloques de 32 imágenes por iteración, optimizando el paralelismo de la GPU/CPU.

---

### Arquitectura de la Red Neuronal

El modelo combina el conocimiento matemático de una red profunda entrenada por Google con una capa lineal de salida ajustada a las necesidades del proyecto.

 Entrada: cam_input  -> (224, 224, 3)
         │
         Preprocesamiento  -> Escalado MobileNetV2
                 │
                 Base: MobileNetV2 -> (Cerebro congelado / weights='imagenet')
                         │
                         GlobalAveragePooling2D  -> Aplastamiento de mapas de características
                                 │
                                 Salida: confidence_score  -> (1 neurona + Act. Sigmoide).

#### Componentes del modelo (estructura resumida)

1) Capa de entrada — `cam_input`
    - Tipo: `tf.keras.Input`
    - Forma: 224 × 224 × 3 (Ancho × Alto × Canales RGB)

2) Preprocesamiento
    - Función: `preprocess_input` de MobileNetV2
    - Acción: escala píxeles de [0,255] a [-1,1], requisito para la red base

3) Cerebro base — MobileNetV2
    - Configuración: `include_top=False`, `weights='imagenet'`
    - Rol: extractor de características preentrenado

4) Congelación de pesos
    - `base_model.trainable = False`
    - Objetivo: evitar actualizar los pesos preentrenados durante el ajuste fino

5) Pooling y reducción dimensional
    - Capa: `GlobalAveragePooling2D`
    - Efecto: convierte mapas de características en un vector compacto, reduciendo parámetros y riesgo de sobreajuste

6) Capa de salida — `confidence_score`
    - Tipo: `Dense(1, activation='sigmoid')`
    - Salida: probabilidad continua en [0.0, 1.0] indicando confianza

Este esquema facilita entender el flujo: entrada → preprocesamiento → cerebro base (congelado) → global pooling → salida.

---

### Compilación

El modelo calcula su tasa de error y optimiza sus parámetros mediante los siguientes componentes:

* **Optimizador (`adam` - Adaptive Moment Estimation):** Algoritmo que calcula tasas de aprendizaje adaptativas para cada parámetro, estabilizando y acelerando la convergencia.
* **Función de Pérdida (`binary_crossentropy`):** Métrica encargada de penalizar los errores de predicción del modelo mediante la fórmula de entropía cruzada binaria.

---

### Fase 3: Entrenamiento y Persistencia del Modelo

El entrenamiento se ejecuta a lo largo de **10 Épocas (`EPOCAS = 10`)**, donde cada época representa una revisión completa del conjunto `train_dataset`.

* **Validación Cruzada:** Tras bambalinas, `modelo.fit` evalúa las métricas de pérdida (`loss`) y precisión (`accuracy`) tanto en los datos de estudio como en los de validación de forma simultánea.
* **Persistencia del "Cerebro" (`.keras`):** Finalizado el entrenamiento, la arquitectura completa, las métricas de optimización y los pesos calculados se consolidan en un único archivo serializado de almacenamiento llamado **`modelo_motos_IUJO.keras`**, quedando listo para su despliegue en entornos de producción o inferencia en tiempo real.