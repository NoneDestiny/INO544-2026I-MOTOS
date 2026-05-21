# Esto funciona como prueba para lo que realmente se hará para el archivo "05_procesamiento_lote". NO se documenta en este archivo.
# Por lo que se sugiere ver el archivo antes mencionado para ver su respectiva documentación y entender como funciona.
# De igual manera, sírvase de los anteriores archivos que explican sus funciones por separado de forma detallada.

import cv2
import os
import numpy as np
import random 
directorio_script = os.path.dirname(os.path.abspath(__file__))

ruta_imagen = os.path.join(directorio_script, "dataset_motos", "moto_494.jpg")

imagen_original = cv2.imread(ruta_imagen)

if imagen_original is None:
    print("Error 404 la aplicación no ha encontrado la imagen. Por favor revisa la ruta o el nombre del archivo.")
else:
    angulo_azar = random.uniform(20, -20)
    alpha_azar = random.uniform (0.2, 1.8)
    mov_x = random.uniform(22, -22)
    mov_y = random.uniform(22, -22)
    zoom_azar = random.uniform(0.8, 1.2)
    print(f"Aplicando aleatoriedad de un angulo de: {angulo_azar:.0f}, con un desplazamiento de X: {mov_x:.0f} | Y: {mov_y:.0f}")
    print(f"zoom e imagen volteada al azar aplicado.")


    (alto, ancho) = imagen_original.shape[:2]
    centro = (ancho // 2, alto // 2)

    matriz_rotacion = cv2.getRotationMatrix2D(centro, angulo_azar, zoom_azar)
    matriz_desplazamiento = np.float32 ([
        [1, 0, mov_x],
        [0, 1, mov_y]
    ])

    imagen_brillo = cv2.convertScaleAbs(imagen_original, alpha=alpha_azar, beta=0)
    imagen_volteada = imagen_brillo
    if random.choice([True, False]):
        print("Efecto espejo aplicado!")
        imagen_volteada = cv2.flip(imagen_brillo, 1)
    imagen_rotada = cv2.warpAffine(imagen_volteada, matriz_rotacion, (ancho, alto))
    imagen_final = cv2.warpAffine(imagen_rotada, matriz_desplazamiento, (ancho, alto))


    cv2.imshow("Moto Original", imagen_original)
    cv2.imshow("Moto Transformada", imagen_final)
    
    print("Presiona cualquier tecla para salir")
    cv2.waitKey(0)

    cv2.destroyAllWindows()