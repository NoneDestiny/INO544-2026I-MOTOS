## 6. Fase de Despliegue e Inferencia en Tiempo Real

### 6.1. Módulo de Clasificación en Video en Vivo

Este módulo implementa el entorno de pruebas (*sandbox*) final para validar el rendimiento del modelo en un escenario del mundo real. Mediante la integración de OpenCV y TensorFlow, el script captura el flujo de video directo de una cámara web, preprocesa los fotogramas dinámicamente y despliega los resultados de la predicción con una superposición gráfica en pantalla en tiempo real.

---

### Lógica de Preprocesamiento de Video Frame a Frame

Las cámaras web capturan video en resoluciones estándar (como HD o Full HD) y en formatos de color nativos del sistema operativo. Para que la red neuronal procese el flujo sin corromper sus pesos matemáticos, cada fotograma (`frame`) pasa por un pipeline estricto de telemetría:

#### 1. Redimensión Espacial (`cv2.resize`)
El fotograma se redimensiona mediante `cv2.resize` a **224 × 224 píxeles**, garantizando que coincida exactamente con las dimensiones de entrada `cam_input` especificadas en la Fase 2.

#### 2. Corrección del Espacio de Color (`cv2.cvtColor`)

> **Nota de Compatibilidad Crítica:** Nos hemos dado cuenta que por diseño histórico, OpenCV decodifica y almacena las imágenes en el orden de canales **BGR** (Blue, Green, Red). Sin embargo, TensorFlow y el modelo base *MobileNetV2* fueron entrenados usando el estándar de la industria **RGB** (Red, Green, Blue). 

El script corrige esta discrepancia mediante el flag `cv2.COLOR_BGR2RGB`. Omitir este paso provocaría que la IA invierta los canales de color, degradando críticamente la precisión del clasificador en producción.

#### 3. Expansión de Dimensiones Tensoriales (`np.expand_dims`)
Las redes neuronales de Keras requieren un tensor de entrada 4D con la estructura [Batch Size, Alto, Ancho, Canales]. Para el procesamiento frame‑a‑frame en tiempo real:

- Batch unitario: la cámara entrega un solo fotograma por iteración.
- Crear el eje de lote: usar NumPy, p. ej. `np.expand_dims(frame, axis=0)`, para añadir el eje de batch.
- Resultado final:
    - Forma del tensor de entrada: [1, 224, 224, 3]
    - Desglose: 1 = tamaño de batch (fotograma único), 224 × 224 = dimensiones espaciales, 3 = canales (RGB)
    - Estado: tensor 4D compatible con Keras / MobileNetV2, listo para la inferencia.

---

### Bucle de Inferencia y Umbral de Decisión

El script ejecuta un bucle infinito `while True` que extrae los fotogramas del hardware de video mediante `cap.read()`. 

La predicción devuelve un valor flotante continuo entre 0.0 y 1.0 gracias a la función de activación sigmoide de la capa de salida. Para clasificar el resultado, se establece un **Umbral de Decisión del 50% (0.5)**:

                Valor de Predicción (Sigmoide) 
              ┌───────────────┴───────────────┐
              ▼                               ▼
          ¿Es >= 0.5?                    ¿Es < 0.5?
              │                               │
              ▼                               ▼
   Clase: [ SI ES MOTO ]            Clase: [ NO ES MOTO ]
   Certeza = Predicción * 100       Certeza = (1 - Predicción) * 100
   Color BGR: (0, 255, 0) [Verde]   Color BGR: (0, 0, 255) [Rojo]

* **Cálculo de Certeza Negativa (probabilidad invertida):**
    - Definición: probabilidad de que el objeto NO sea una motocicleta.
    - Fórmula (porcentaje): `certeza_negativa = (1 - prediccion) * 100`.
    - Ejemplo: si `prediccion = 0.8` → `certeza_negativa = (1 - 0.8) * 100 = 20%`.

---

### Interfaz Gráfica (HUD) y Control de Salida

Los resultados de la inferencia se inyectan directamente sobre el video en ejecución antes de ser renderizados en la pantalla:

* **Superposición de Texto (`cv2.putText`):** Dibuja dinámicamente la clase detectada junto con su porcentaje de certeza con un formato acotado a dos decimales (`:.2f`). El color del texto cambia dinámicamente (Verde para positivo, Rojo para negativo) utilizando la codificación de color nativa BGR de OpenCV.
* **Manejo del Teclado e Interrupción:** El método `cv2.waitKey(1)` analiza el teclado cada milisegundo. Mediante una operación de máscara de bits (`& 0xFF`), detecta si el usuario presiona la tecla **'q'** para romper el ciclo de forma segura.
* **Liberación de Recursos:** Al salir del bucle, el método `cap.release()` apaga físicamente el sensor de la cámara web y desasigna el control del hardware, mientras que `cv2.destroyAllWindows()` destruye los hilos de la ventana gráfica para evitar bloqueos en el sistema operativo.


