### 2.4. Pipeline Unificado de Data Augmentation

Este módulo representa la integración final de todas las técnicas de transformación previamente validadas por separado (Módulos 02, 03 y 04). Antes de aplicar estas alteraciones de forma masiva a todo el dataset, este archivo funciona como una **prueba de concepto unitaria** utilizando una sola imagen (`moto_494.jpg`).

El objetivo es consolidar en un solo flujo secuencial los cambios geométricos, lumínicos y estocásticos (aleatorios), maximizando la distorsión controlada de la imagen en un único paso de ejecución.

---

### Flujo Secuencial del Pipeline

Para evitar la pérdida de información por recortes o desbordamientos, las transformaciones se ejecutan estrictamente en el siguiente orden:

img_xxx.jpg
    │
    1. Modificación de Brillo (convertScaleAbs con alpha aleatorio)
        │
        2. Decisión Estocástica de Volteo Horizontal (cv2.flip al 50% de probabilidad)
            │
            3. Rotación y Escala Centrada (cv2.getRotationMatrix2D + warpAffine)
                │    
                4. Desplazamiento de Ejes (Matriz NumPy + warpAffine)
                    │
                    img_xxx.jpg

---

### Integración de Parámetros Dinámicos

Cada ejecución del pipeline genera un set único de mutaciones gracias al uso combinado de `random.uniform` y `random.choice`:

* **Luz Fluctuante:** El parámetro `alpha_azar` varía entre `0.2` (imagen críticamente oscura) y `1.8` (imagen altamente sobreexpuesta), simulando entornos de luz extremos.
* **Efecto Espejo Probabilístico:** A través de un condicional booleano aleatorio (`random.choice([True, False])`), el script decide si aplica o no el volteo horizontal (`cv2.flip`). Esto introduce una probabilidad exacta del 50% para esta mutación.
* **Geometría Espacial Avanzada:** Se calculan las dimensiones y el centro del lienzo para aplicar en cascada la rotación/zoom y la traslación lineal en los ejes $X$ y $Y$.

---

Tomando en cuenta las funciones de visualización en pantalla, que están incluidas en la fase de este proyecto, debemos tomarlas como de prueba, ya que en otro caso, deben ser removidas, o dejar comentadas en el script definitivoLas funciones de visualización en pantalla (`cv2.imshow`, `cv2.waitKey`) incluidas en esta fase de pruebas deberán ser removidas o comentadas en el script por lotes definitivo para evitar bloquear los hilos de ejecución del procesador al manejar miles de archivos simultáneos.

---

### 2.5. Automatización de Data Augmentation por Lotes

Este módulo representa el núcleo de la **Fase 1 del proyecto**. Su función principal es el procesamiento masivo y automatizado de un dataset base de 600 imágenes utilizando Visión Artificial. A través de una arquitectura de bucles anidados, el script genera variaciones geométricas y lumínicas aleatorias y controladas para robustecer el entrenamiento de la IA.

---

### Gestión de Entornos y Carpetas Automatizada

El script garantiza la autonomía de su ejecución mediante flujos de control de directorios integrados:
* **Creación Segura del Destino (`os.makedirs`):** El script verifica si existe el directorio `dataset_aumentado`. Al usar el parámetro `exist_ok=True`, si la carpeta no existe, la crea en tiempo de ejecución; si ya existe, continúa sin arrojar errores ni sobrescribir archivos ajenos.
* **Aislamiento de Rutas:** Lee de forma dinámica el origen (`dataset_motos`) procesando los archivos uno a uno de manera secuencial.

---

### Arquitectura del Algoritmo: Bucle Anidado

La multiplicación exponencial del dataset se logra mediante una estructura de doble ciclo (bucle anidado):

Inicio del Script
│
Bucle Externo (Recorre las 600 imágenes originales)
│
├── Carga la imagen actual (cv2.imread)
└── Validación (if imagen_original is not None)
│
Bucle Interno (Ejecuta 5 iteraciones por imagen)
│
├── Genera 5 hiperparámetros al azar (Giro, Brillo, Zoom, X, Y)
├── Calcula matrices físicas de transformación
├── Aplica el pipeline secuencial de OpenCV
└── Guarda el nuevo archivo único en el disco duro

Al procesar las 600 imágenes originales mediante las 5 iteraciones del ciclo interno, el script inyecta un total de **3,000 imágenes modificadas** al dataset final de entrenamiento.

---

### Configuración de Restricciones (Límites de Seguridad)

Para evitar distorsiones severas que rompan la coherencia visual de los objetos (lo que "confundiría" a la IA), el script opera bajo un marco estricto de valores mínimos y máximos validados:

| Parámetro | Rango Operativo | Propósito Técnico |
| :--- | :---: | :--- |
| **Rotación (`angulo_azar`)** | `[-20, 20]` grados | Evita que el vehículo quede completamente invertido. |
| **Brillo (`alpha_azar`)** | `[0.8, 1.2]` | Mantiene la visibilidad de las texturas sin llegar al blanco o negro puro. |
| **Traslación (`mov_x`, `mov_y`)** | `[-22, 22]` píxeles | Desplaza el objeto sin sacarlo por completo del encuadre. |
| **Escala (`zoom_azar`)** | `[0.8, 1.2]` | Controla la distancia de enfoque de la cámara sobre la moto. |

---

### Nomenclatura Estructurada y Guardado Físico

Para evitar conflictos de sobreescritura en el almacenamiento del sistema operativo, el módulo implementa un patrón de nombres único para cada archivo generado en el disco duro mediante `cv2.imwrite`:

Formato: `aug_{índice_iteración}_{nombre_original.ext}`

Componentes:
- `índice_iteración o {i}`: número entero de la iteración del bucle (p. ej. 0..4).
- `nombre_original.ext`: nombre del archivo original incluyendo su extensión (p. ej. `moto_001.jpg`).

Reglas:
- Separar con guiones bajos (`_`).
- Mantener la extensión original del archivo.
- Evitar espacios y caracteres especiales en `nombre_original`.

Ejemplo: `moto_001.jpg` -> `aug_0_moto_001.jpg`, `aug_1_moto_001.jpg`, ..., `aug_4_moto_001.jpg`

* **Ejemplo de salida:** Una imagen original llamada `moto_001.jpg` generará cinco variantes independientes llamadas: `aug_0_moto_001.jpg`, `aug_1_moto_001.jpg`, ..., `aug_4_moto_001.jpg`.

Este esquema asegura la trazabilidad completa de los datos, permitiendo identificar de qué archivo base proviene cada mutación en las fases posteriores del proyecto.
