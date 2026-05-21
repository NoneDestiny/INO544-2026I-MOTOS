import cv2
import os
import random # <--- herramienta para aleatoriedad
import numpy as np # <--- herramienta para matrices

directorio_script = os.path.dirname(os.path.abspath(__file__))

ruta_imagen = os.path.join(directorio_script, "dataset_motos", "moto_481.jpg")

imagen_original = cv2.imread(ruta_imagen)

if imagen_original is None:
    print("Error 404 la aplicación no ha encontrado la imagen. Por favor revisa la ruta o el nombre del archivo.")
else:
    angulo_azar = random.uniform(20, -20)
    zoom_azar = random.uniform(0.8, 1.2)
    mov_x = random.uniform(-22, 22)
    mov_y = random.uniform(-22, 22)
    print(f"Imagen cargada con éxito. Se aplicarán los valores aleatorios:Rotacion aleatoria de {angulo_azar:.2f}, Zoom aleatoria de {zoom_azar:.2}. Ademas un desplazamiento de X: {mov_x:.0f} | Y: {mov_y:.0f}")


    # 1. Calculamos el centro exacto de la imagen (mitad de ancho y mitad de alto)
    # .shape nos da las medidas reales de la foto
    (alto, ancho) = imagen_original.shape[:2]
    centro = (ancho // 2, alto // 2)

    # 2. Creamos la "Matriz de Rotación" (El plan de giro)
    # Parámetros: (Punto central, Ángulo en grados, Escala/Zoom)
    # Vamos a girarla 20 grados positivos.
    matriz = cv2.getRotationMatrix2D(centro, angulo_azar, zoom_azar)

    # 2.1 Creamos la Matriz de Traslación con numpy
    # La estructura siempre es [[1, 0, movimiento_x], [0, 1, movimiento_y]]
    matriz_desplazamiento = np.float32 ([
        [1, 0, mov_x],
        [0, 1, mov_y]
    ])

    # 3. Ejecutamos el giro final aplicando la matriz a la imagen
    imagen_rotada = cv2.warpAffine(imagen_original, matriz, (ancho, alto))
    imagen_transformada = cv2.warpAffine(imagen_rotada, matriz_desplazamiento, (ancho, alto))

    
    cv2.imshow("Moto Original", imagen_original)
    cv2.imshow("Imagen_transformada", imagen_transformada)

    print("Presiona cualquier tecla para salir")
    cv2.waitKey(0)

    cv2.destroyAllWindows()