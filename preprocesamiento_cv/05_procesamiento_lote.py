# Fase 1 del proyecto. Preprocesamiento de 600 imágenes para un dataset aumentado con el fin de entrenar a una IA con "Vision Artificial".
# Este código busca la creación de una nueva carpeta para las modificaciones (o transformaciones) que se realizaran a las imágenes originales.
# Su alcance es de un bucle anidado que modifique 5 veces (de forma aleatoria) los valores predeterminados de Zoom, Desplazamiento X|Y, Brillo
# Rotación y que se pueda voltear con valores que limitan un mínimo y un máximo para diferentes resultados.

import cv2
import os
import numpy as np
import random 
directorio_script = os.path.dirname(os.path.abspath(__file__))

# Ruta de Carpetas (Origen y Destino).
ruta_dataset_aumentado = os.path.join(directorio_script, "dataset_aumentado")
os.makedirs(ruta_dataset_aumentado, exist_ok=True)

# Esto apunta a la carpeta de las motos, no a ningún archivo en concreto.
ruta_carpeta_motos = os.path.join(directorio_script, "dataset_motos")

print("Iniciando fabrica de Data Augmentation...")

# Este es el primer bucle. Esto recorre cada archivo dentro de la carpeta
for nombre_archivo in os.listdir(ruta_carpeta_motos):

    # Se construye la ruta exacta de la foto que toca en su respectivo turno
    ruta_imagen_actual = os.path.join(ruta_carpeta_motos, nombre_archivo)
    imagen_original = cv2.imread(ruta_imagen_actual)

    # Esto valida que la imagen se leyó correctamente antes de poder hacerle alguna modificación.
    if imagen_original is not None:

        # Este es el segundo bucle. A partir de acá realiza el proceso de "Bucle Anidado".
        # Se solicita que realice 5 ciclos para tener 5 fotos distintas de la foto original para mayor numero de datos dentro de la carpeta para la IA
        for i in range(5):

            # En este punto se le dan los valores especificos que pidió el profesor para no romper la IA
            # Se usa la aleatoriedad para las características y transformaciones que se harán a continuación.
            angulo_azar = random.uniform(20, -20)
            alpha_azar = random.uniform (0.8, 1.2)
            mov_x = random.uniform(22, -22)
            mov_y = random.uniform(22, -22)
            zoom_azar = random.uniform(0.8, 1.2)

            # Se pide que la imagen se divida a la mitad, tanto el total del alto como del ancho para lograr obtener el centro.
            # El .shape[:2] es para que solo se tome 2 valores respectivos de 3.
            # Y se crea una funcion de "centro" para que llame el alto y ancho para dividir con exactitud la mitad de la misma.
            (alto, ancho) = imagen_original.shape[:2]
            centro = (ancho // 2, alto // 2)

            # Esta función de CV2 hace que exista el proceso de la rotación en 2 dimensiones, aplica el zoom y se hace llamado a las funciones 
            # con aleatoriedad. Ademas en np (Abreviación que está al principio de numpy) es el que permite el desplazamiento.
            matriz_rotacion = cv2.getRotationMatrix2D(centro, angulo_azar, zoom_azar)
            matriz_desplazamiento = np.float32 ([
                [1, 0, mov_x],
                [0, 1, mov_y]
            ])

            # Aqui se aplican las transformaciones solicitadas, brillo, flip, rotacion, desplazamiento y zoom.
            imagen_brillo = cv2.convertScaleAbs(imagen_original, alpha=alpha_azar, beta=0)
            imagen_volteada = imagen_brillo
            if random.choice([True, False]):
                imagen_volteada = cv2.flip(imagen_brillo, 1)
            imagen_rotada = cv2.warpAffine(imagen_volteada, matriz_rotacion, (ancho, alto))
            imagen_final = cv2.warpAffine(imagen_rotada, matriz_desplazamiento, (ancho, alto))

            # Para finalizar, empieza el proceso del guardado.
            # 1. Se crea el nuevo nombre del archivo nuevo.
            nombre_archivo_nuevo = f"aug_{i}_{nombre_archivo}"

            # 2. Se construye la ruta final donde se va a almacenar este nuevo archivo modificado.
            ruta_guardado = os.path.join(ruta_dataset_aumentado, nombre_archivo_nuevo)

            # 3. Se hace uso de OpenCV para que guarde el archivo en el disco duro.
            cv2.imwrite(ruta_guardado, imagen_final)

            print(f"Creada: {nombre_archivo_nuevo}")

print("--- Bucle finalizado, revisa la nueva carpeta :D ---")
