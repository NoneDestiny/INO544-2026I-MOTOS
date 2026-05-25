## 2. Fase de Preprocesamiento y Aumento de Datos

### 2.1. Módulo de Data Augmentation por Volteo Horizontal

Este módulo demuestra de forma práctica cómo aplicar técnicas de **Data Augmentation** (Aumento de Datos) utilizando la librería OpenCV. En este caso específico, se realiza una transformación geométrica de volteo horizontal (efecto espejo) sobre una imagen de prueba (`moto_001.jpg`).

El objetivo de esta técnica es generar nuevas muestras de entrenamiento para que el modelo de Machine Learning aprenda a reconocer el objeto (motos) sin importar la dirección en la que estén orientadas.

---

### Lógica y Funcionamiento Técnico

#### 1. Carga y Matriz de Datos
A diferencia de otras librerías visuales, el método `cv2.imread()` no despliega la imagen de inmediato, sino que la transforma instantáneamente en una matriz multidimensional de números (NumPy Array) que representa los píxeles y sus canales de color.

#### 2. Control de Existencia (Validación de Carga)
Dado que OpenCV no lanza una excepción nativa si la ruta del archivo es incorrecta (simplemente devuelve un valor vacío `None`), el script implementa una validación preventiva:

```python
if imagen_original is None:
    # Captura el error antes de intentar procesar una matriz vacía
```