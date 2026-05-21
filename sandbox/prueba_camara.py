import cv2
import numpy as np
import tensorflow as tf

# 1. De este pequeño script se llama al modelo entrenado.
print("Cargando modelo...")
modelo = tf.keras.models.load_model('modelo_motos_IUJO.keras') 
print("Modelo cargado con éxito.")

# 2. Se enciende la camara web. El valor "0" es para usar la camara integrada.
cap = cv2.VideoCapture(0)

print("Iniciando transmisión en vivo. Presiona la tecla 'q' para salir.")

while True:
    # Se captura el video cuadro por cuadro (o frames)
    ret, frame = cap.read()
    if not ret:
        break

    # --- TESTING Y PREPARACION DE LA IMAGEN PARA LA IA ---
    # a. La imagen capturada de la camara lo va a redimensionar a la resolucion con la que hemos venido trabajando (224x224).
    img_redimensionada = cv2.resize(frame, (224, 224))
    
    # b. Al parecer, usando OpenCV el formato de colores lo lee en BGR. Con este script buscamos de que lo lea en RGB
    # Como lo trabaja Keras/Google.
    img_rgb = cv2.cvtColor(img_redimensionada, cv2.COLOR_BGR2RGB)
    
    # c. Aqui transformamos en formato matematico y agregamos las dimensiones del lote [1, 224, 224, 3]
    img_array = np.expand_dims(img_rgb, axis=0)
    
    # d. Se aplica el traductor de pixeles especificos que trabaja MobileNetV2. Osea, -1 a 1.
    # img_lista = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    # img_lista = img_array / 255.0

    # --- REALIZACION DE PREDICCIÓN ---
    prediccion = modelo.predict(img_array, verbose=0)[0][0] # Extraemos el número exacto
    
    # --- INTERPRETACION DEL RESULTADO ---
    # Aqui dividimos el porcentaje en 50/50. Por lo que a partir de">= 0.5" será nuestra respuesta positiva de que
    # la IA si confirma que es una moto.
    if prediccion >= 0.5:
        # Recordar que .2f es para decimales.
        texto = f"SI ES MOTO ({(prediccion * 100):.2f}%)"
        color = (0, 255, 0) # Nuestro texto afirmativo en Verde en BGR.
    else:
        # Como es probabilidad de que SEA moto, restamos en 1 para saber la seguridad de que NO lo es.
        certeza_no_moto = (1 - prediccion) * 100
        texto = f"NO ES MOTO ({certeza_no_moto:.2f}%)"
        color = (0, 0, 255) # Nuestro texto negativo en Rojo en BGR.

    # Debe dibujar o mostrar el texto en pantalla con su respuesta afirmativa/negativa.
    cv2.putText(frame, texto, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    # Abre una ventana para la transmisión de la cámara en vivo.
    cv2.imshow('Detector de Motos en Vivo', frame)

    # Usaremos "q" para acabar con el ciclo y poder salir del programa,Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberamos el uso de la cámara y cerramos la ventana al concluir.
cap.release()
cv2.destroyAllWindows()