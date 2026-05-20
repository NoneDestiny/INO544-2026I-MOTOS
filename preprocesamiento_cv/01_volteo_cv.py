import cv2
import os

directorio_script = os.path.dirname(os.path.abspath(__file__))

ruta_imagen = os.path.join(directorio_script, "dataset_motos", "moto_001.jpg")

print(f"Buscando imagen en:{ruta_imagen}")

# 1. Leer la imagen. 
# OpenCV no la abre visualmente, la convierte en una matriz de números al instante.
imagen_original = cv2.imread(ruta_imagen)

#2 Esto hace que OpenCV confirme si de verdad la imagen solicitada existe o encontró algo.
if imagen_original is None:
    print("❌ ERROR CATASTRÓFICO: OpenCV no encontró la imagen. Revisa el nombre o la ruta.")
else:
    print("✅ Imagen cargada exitosamente. Aplicando Data Augmentation...")

    # 2. Aplicar el Volteo Horizontal.
    # En OpenCV, el comando es flip. El número 1 significa "voltear en el eje Y (horizontalmente)".
    # Si usaras 0, la voltearía de cabeza.
    imagen_volteada = cv2.flip(imagen_original, 1)

    # 3. Mostrar ambas imágenes en ventanas emergentes
    cv2.imshow("Moto Original", imagen_original)
    cv2.imshow("Moto Espejo (Data Augmentation)", imagen_volteada)

    # 4. Funcion requerida por OpenCV:
    # cv2.waitKey(0) pausa el programa infinitamente hasta que presiones CUALQUIER tecla.
    # Si no pones esto, las ventanas se abren y se cierran en 1 milisegundo y no ves nada.
    print("Presiona cualquier tecla en la ventana de la imagen para cerrar...")
    cv2.waitKey(0)

    # 5. Cerrar las ventanas y limpiar la memoria
    cv2.destroyAllWindows()