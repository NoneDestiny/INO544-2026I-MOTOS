### 2.3. Módulo de Transformaciones Geométricas Aleatorias

Este módulo implementa técnicas avanzadas de **Data Augmentation** mediante transformaciones afines combinadas. A diferencia de los módulos anteriores, este script utiliza un motor de aleatoriedad para variar la rotación, el zoom y el desplazamiento (traslación) de la imagen de prueba (`moto_481.jpg`).

Esta variabilidad estocástica simula imperfecciones reales en la captura de datos, como cámaras mal niveladas o sujetos descentrados.

---

### Lógica y Funcionamiento Técnico

#### 1. Generación de Hiperparámetros Aleatorios
El script utiliza la librería `random` para definir un espacio de búsqueda de variaciones. Cada vez que se ejecuta, los valores cambian dentro de los siguientes rangos:

| Transformación | Rango de Valores | Descripción Visual |
| :--- | :--- | :--- |
| **Rotación** | `[-20.0, 20.0]` grados | Giro leve a la izquierda o derecha. |
| **Zoom (Escala)** | `[0.8, 1.2]` | De un 20% de alejamiento a un 20% de acercamiento. |
| **Traslación (X, Y)** | `[-22, 22]` píxeles | Desplazamiento lateral y vertical del objeto. |

#### 2. Cálculo de Matrices de Transformación
Para aplicar estos cambios, el script descompone la operación en dos matrices matemáticas:

* **Matriz de Rotación y Escala (`cv2.getRotationMatrix2D`):** Calcula el plano de giro tomando como eje el centro exacto de la imagen (determinado dinámicamente mediante `imagen_original.shape`).
* **Matriz de Traslación (NumPy):** Se construye una matriz de tipo `float32` que define el movimiento en los ejes cartesianos.

#### 3. Aplicación de Transformaciones Afines (`cv2.warpAffine`)
La función `cv2.warpAffine` es la encargada de interpolar los píxeles de la imagen original hacia su nueva posición basándose en las matrices calculadas. 
1. Primero se aplica la **rotación y el zoom**.
2. Sobre ese resultado, se aplica el **desplazamiento**.

---

El script proporciona una retroalimentación detallada por consola antes de mostrar los resultados, imprimiendo los valores exactos generados por el motor de aleatoriedad (con formato de dos decimales para mayor legibilidad).

* **Visualización:** Se contrastan la "Moto Original" y la "Imagen Transformada". Debido a la naturaleza del desplazamiento y la rotación, es normal observar áreas negras (píxeles sin información) en los bordes de la imagen transformada.
* **Gestión de Memoria:** Al igual que en módulos previos, el flujo se mantiene seguro mediante `waitKey(0)` y `destroyAllWindows()` para evitar fugas de memoria en el sistema.