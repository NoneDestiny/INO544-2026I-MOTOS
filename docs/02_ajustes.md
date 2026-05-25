### 2.2. Módulo de Ajuste de Brillo y Contraste

Este módulo implementa técnicas de **Data Augmentation** enfocadas en la variabilidad lumínica del dataset. A través de la función `cv2.convertScaleAbs()`, el script altera el brillo de una imagen de prueba (`moto_549.jpg`) para generar variaciones claras y oscuras.

El objetivo de este proceso es entrenar al modelo de Inteligencia Artificial para que sea robusto frente a diferentes condiciones de iluminación ambiental (cambios climáticos, sombras, sobreexposición o tomas nocturnas).

---

### Lógica y Funcionamiento Técnico

#### 1. Operación Matemática de Escalado (`cv2.convertScaleAbs`)
A diferencia de los cambios geométricos (como el volteo), el ajuste de brillo altera directamente los valores numéricos de los píxeles en la matriz de la imagen. La función aplica una ecuación lineal básica para cada píxel:

Nuevo Píxel = Alfa * Píxel Original + Beta

El script utiliza los siguientes hiperparámetros para la transformación:

| Variable | Rol Técnico | Valor en el Script | Efecto en la Imagen |
| :---: | :--- | :---: | :--- |
| Alfa | Factor de escala (Contraste/Brillo masivo) | `0.8` | **Reducción de intensidad:** Genera la imagen "clara" (atenuada). |
| Alfa | Factor de escala (Contraste/Brillo masivo) | `1.2` | **Incremento de intensidad:** Genera la imagen "oscura" (más intensa/saturada). |
| Beta | Valor sumado/restado (Brillo constante) | `0` | No se aplica desplazamiento lineal constante. |

> **Cómo nota técnica:** `convertScaleAbs` no solo calcula la operación, sino que trunca automáticamente los valores resultantes para que se mantengan en el rango válido de un color de 8 bits (0 a 255). Si un píxel supera el valor de 255, lo estabiliza en 255 para evitar errores de desbordamiento de memoria (overflow).

---

### Flujo de Control e Interfaz Gráfica

El comportamiento del entorno gráfico mantiene los mismos estándares de seguridad y gestión de memoria que los módulos anteriores:

1. **Validación Preventiva:** Mediante `if imagen_original is None:`, se verifica la existencia del archivo antes de intentar procesar la matriz, arrojando un mensaje de error controlado en caso de falla.
2. **Despliegue Múltiple:** Se abren tres ventanas simultáneas (`cv2.imshow`) para permitir al desarrollador contrastar visualmente el impacto del cambio lumínico: la imagen original, la versión atenuada (`alpha=0.8`) y la versión intensificada (`alpha=1.2`).
3. **Persistencia Visual y Cierre:** Se utiliza `cv2.waitKey(0)` para retener los gráficos en memoria y `cv2.destroyAllWindows()` para limpiar de forma segura los recursos de la memoria RAM asignados al cerrarse.