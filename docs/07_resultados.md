## 5. Fase de Evaluación de Resultados y Rendimiento

### 5.1. Módulo de Graficación e Interpretación de Métricas

Este módulo tiene como objetivo extraer, consolidar y graficar los datos de rendimiento obtenidos durante el entrenamiento del modelo. A través de la librería `matplotlib`, el script genera un reporte visual de alta calidad que contrasta la evolución del aprendizaje entre los datos de entrenamiento y los datos de validación.

---

### Análisis Crítico de los Resultados Obtenidos

El script procesa los arreglos históricos del entrenamiento a lo largo de las 10 épocas. Los datos duros revelan el siguiente comportamiento del modelo:

#### 1. Evolución de la Precisión
* **Entrenamiento:** Inicia en un sólido **93.74%** en la primera época y converge de manera asintótica hasta alcanzar el **100%** (`1.0000`) en las épocas 9 y 10.
* **Validación:** Comienza en un **99.16%** y se estabilice de forma consistente en un **99.67%**. La cercanía milimétrica entre ambas curvas demuestra la excelente capacidad de generalización del clasificador ante datos que nunca antes había visto.

#### 2. Evolución de la Pérdida
* **Entrenamiento:** Cae drásticamente desde `0.1837` hasta un residuo marginal de `0.0045`.
* **Validación:** Disminuye de forma suave y controlada de `0.0650` a `0.0115`. Al no presentar rebotes ni tendencias ascendentes en las últimas épocas, se descarta analíticamente cualquier indicio de *Overfitting* o *Underfitting*.

---

#### 1. Configuración de Lienzo Multi-Panel (`plt.subplot`)
Para facilitar la lectura técnica en documentos o tesis, el script inicializa un lienzo rectangular de 12 × 5 pulgadas y organiza el área en una matriz de subgráficas de 1 fila por 2 columnas (plt.subplot(1, 2, i)).

- Panel izquierdo — plt.subplot(1, 2, 1): muestra las curvas de precisión (accuracy) para entrenamiento y validación.
- Panel derecho — plt.subplot(1, 2, 2): muestra las curvas de pérdida (loss) para entrenamiento y validación.

Esta disposición lado a lado facilita la comparación directa entre métricas y mantiene una presentación coherente y adecuada para informes técnicos.

#### 2. Estandarización Visual del Reporte
Cada gráfica se construye bajo normas estrictas de visualización de datos:
* **Codificación de Colores:** Azul (`b-`) para los flujos de entrenamiento y Verde (`g-`) o Rojo (`r-`) para los flujos de control de validación.
* **Rejillas de Medición (`plt.grid(True)`):** Habilita líneas de cuadrícula para rastrear visualmente el valor numérico exacto de cualquier coordenada (X, Y) (Época vs. Métrica).

#### 3. Ajuste Dinámico y Exportación de Alta Fidelidad
El procesamiento de la imagen final ejecuta dos pasos críticos antes del guardado físico:
* **`plt.tight_layout()`:** Algoritmo que calcula y reajusta automáticamente los márgenes entre las gráficas, impidiendo que los textos de los ejes ($Y$ de la pérdida y de la precisión) se encimen o se corten.
* **Almacenamiento Binario Destino:** Verifica y asegura la persistencia de la carpeta contenedora `src`. Posteriormente, exporta el gráfico con una resolución de **300 DPI** (Puntos por pulgada), calidad estándar requerida para publicaciones impresas o reportes técnicos ejecutivos.
  