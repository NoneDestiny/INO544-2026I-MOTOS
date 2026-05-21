import cv2
import os

directorio_script = os.path.dirname(os.path.abspath(__file__))

ruta_imagen = os.path.join(directorio_script, "dataset_motos", "moto_549.jpg")

print(f"Buscando imagen en:{ruta_imagen}")

# 1. Leer la imagen. 
# OpenCV no la abre visualmente, la convierte en una matriz de números al instante.
imagen_original = cv2.imread(ruta_imagen)

#2 Esto hace que OpenCV confirme si de verdad la imagen solicitada existe o encontró algo.
if imagen_original is None:
    print("Error 404 la aplicación no ha encontrado la imagen. Por favor revisa la ruta o el nombre del archivo.")
else:
    print("La imagen ha sido cargada con éxito. Se aplicará Data Augmentation...")

    # 2. Aplicar el Volteo Horizontal.
    # En OpenCV, el comando es flip. El número 1 significa "voltear en el eje Y (horizontalmente)".
    # Si usaras 0, la voltearía de cabeza.
    imagen_clara = cv2.convertScaleAbs(imagen_original, alpha=0.8, beta= 0)
    imagen_oscura = cv2.convertScaleAbs(imagen_original, alpha=1.2, beta= 0)

    # 3. Mostrar ambas imágenes en ventanas emergentes
    cv2.imshow("Moto Original", imagen_original)
    cv2.imshow("Imagen clara (escalado de brillo))", imagen_clara)
    cv2.imshow("Imagen Oscura (Escalado de Brillos)", imagen_oscura)

    # 4. Magia negra requerida por OpenCV:
    # cv2.waitKey(0) pausa el programa infinitamente hasta que presiones CUALQUIER tecla.
    # Si no pones esto, las ventanas se abren y se cierran en 1 milisegundo y no ves nada.
    print("Presiona cualquier tecla en la ventana de la imagen para cerrar...")
    cv2.waitKey(0)

    # 5. Cerrar las ventanas y limpiar la memoria
    cv2.destroyAllWindows()