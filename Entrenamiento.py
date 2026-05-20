import tensorflow as tf
import os

# --- FASE 1: DIVISION Y PREPROCESAMIENTO DE IMáGENES"
# 1. Esta es la direccion o la carpeta que tiene que buscar para ubicar las 2 subcarpetas
DIRECTORIO_DATOS = os.path.join ("datasets", "dataset_final")

# Mas abajo hace que estas indicaciones sea que las imágenes si o si sean 224x224
ALTO_IMG = 224
ANCHO_IMG = 224
TAMANO_LOTE = 32 # Representa cuantas fotos agarra Keras a la vez.

# 2. El validation_split = 0.2 representa el 20% de las imágenes que tiene que separar para el training y el testing
# En donde el subset clasifica las imágenes para training y testing, respectivamente.
# El seed asegura que la selección aleatoria sea siempre la misma. Así evitamos que si reiniciamos el programa, 
# una foto que la IA ya vio en el 'training' pase accidentalmente al 'testing' y nos dé una precisión falsa.
print("Preparando las imágenes para procesar (80% de training)")
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DIRECTORIO_DATOS,
    validation_split = 0.2,
    subset='training',
    seed=123,
    image_size = (ALTO_IMG, ANCHO_IMG),
    batch_size = TAMANO_LOTE
)

print("Preparando las imágenes para procesar (20% de testing)")
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DIRECTORIO_DATOS,
    validation_split = 0.2,
    subset='validation',
    seed=123,
    image_size = (ALTO_IMG, ANCHO_IMG),
    batch_size = TAMANO_LOTE
)

# 3. Aqui se verifica el trabajo de Keras

nombres_clases = train_dataset.class_names

print(f"\n Categorias detectadas automaticamente: {nombres_clases}")

# --- FASE 2: INVOCANDO A MOBILENET Y CONSTRUYENDO LA RED ---

# 1. La Puerta de Entrada (Directriz del Profesor)
# Definimos el tamaño exacto y el nombre que pidió en la pizarra
puerta_entrada = tf.keras.Input(shape=(224, 224, 3), name="cam_input")

# 2. Adaptador de imagen
# MobileNetV2 necesita que los píxeles estén en un formato matemático específico
x = tf.keras.applications.mobilenet_v2.preprocess_input(puerta_entrada)

# 3. El Cerebro Pre-entrenado (Transfer Learning)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False, # ¡Le quitamos la capa de salida original!
    weights='imagenet' # Usamos el conocimiento que aprendió en Google
)
base_model.trainable = False # Mantenemos los datos del pre-entrenamiento del cerebro para no alterar nada y no causar errores

# Pasamos nuestra imagen por el cerebro
x = base_model(x, training=False)

# 4. Aqui se aplastan los datos matematicos que obtiene al analizar la imagen
x = tf.keras.layers.GlobalAveragePooling2D()(x)

# 5. La Puerta de Salida solicitado por el profesor
# Una sola neurona, función sigmoide y el nombre exacto de la pizarra
puerta_salida = tf.keras.layers.Dense(1, activation='sigmoid', name="confidence_score")(x)

# 6. Se empaqueta los datos con los inputs/outputs que el profesor detalló para el modelado final
modelo = tf.keras.Model(inputs=puerta_entrada, outputs=puerta_salida)

# 7. Definiendo cómo va a aprender con el optimizador "adam" (adaptative moment estimation) usando la formula de la
# Entropia cruzada binaria que es una fórmula matemática que calcula qué tan equivocada estuvo la IA. Si la foto es una moto (1),
# la fuerza a acercarse a 1; si no es moto (0), la fuerza a acercarse a 0, penalizando severamente si la IA adivina mal
modelo.compile(    
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 8. Imprime el resumen del modelo para verificar
print("\n Arquitectura de la Red Neuronal Finalizada:")
modelo.summary()

# --- FASE 3: EL ENTRENAMIENTO ---

# 1. Definiendo los ciclos de estudio. Siendo Epocas ("Epoch") el ciclo de revision de todo el conjunto de datos existentes.
# En cada ciclo nos dará los resultados obtenidos para ver el error de estudio de las imágenes de training, las del testing
# La precision con cada una y por ultimo el valor de precision al analizar imágenes nunca antes vistas.
EPOCAS = 10 

print("\nIniciando el entrenamiento de la Inteligencia Artificial...")

# 2. Aquí le entregamos el 20% de las imágenes separadas (validation_dataset) para que, al final de cada ciclo, 
# se haga un examen sorpresa con imágenes que no ha estudiado y sepamos si realmente aprende o solo adivina.
historial = modelo.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCAS
)

print("\n Entrenamiento finalizado con éxito.")

# 3. Se resguarda lo procesado en el nuevo cerebro para que no se pierda el progreso.
modelo.save("modelo_motos_IUJO.keras")
print("Modelo guardado de forma segura como 'modelo_motos_IUJO.keras'")