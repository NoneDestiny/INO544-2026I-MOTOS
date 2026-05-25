# Documentación Técnica del Proyecto

## 1. Fase de Ingeniería y Validación de Datos

### 1.1. Módulo de Auditoría de Imágenes

Este módulo se encarga de realizar una verificación automatizada sobre el banco de imágenes del proyecto. Su objetivo principal es garantizar que todos los archivos cumplan con los estándares técnicos requeridos por el modelo de aprendizaje profundo (Deep Learning) antes de iniciar la etapa de entrenamiento.

---

### Criterios de Aceptación del Dataset

Para que una imagen sea clasificada con el estado de **"PERFECTA"**, debe cumplir obligatoriamente con los siguientes tres parámetros:

| Parámetro | Requisito Esperado | Notas de la Librería (Pillow) |
| :--- | :--- | :--- |
| **Formato** | `JPEG` | Los archivos con extensión `.jpg` son identificados internamente como `JPEG`. |
| **Espacio de Color** | `RGB` | Tres canales estándar. Se descartan imágenes en escala de grises o CMYK. |
| **Resolución** | `224 x 224` píxeles | Dimensión fija requerida por la capa de entrada de la red neuronal. |

Si un archivo no cumple con alguno de estos puntos, el sistema lo marcará como `ERROR` especificando la falla (ej. *No es RGB*, *Resolución incorrecta*).

---

### Lógica y Funcionamiento Técnico

#### 1. Gestión Dinámica de Rutas
Para evitar errores de ejecución al cambiar de computadora o servidor, el script **no utiliza rutas absolutas**. En su lugar, detecta el directorio del script en ejecución mediante `os.path.dirname` y construye una ruta relativa hacia la carpeta del dataset:
* **Ruta de búsqueda:** `../datasets/dataset_final/dataset_no_motos`

#### 2. Control de Excepciones (Tolerancia a Fallos)
El procesamiento se realiza dentro de un bloque `try-except`. Si el script topa con un archivo dañado, corrupto o que no es una imagen válida, la librería `Pillow` fallará de manera segura. El script capturará este error, registrará que el archivo está corrupto y **continuará con la ejecución** sin detener la auditoría del resto del dataset.

---

### Limitaciones y Solución de Logs (Buffer de Terminal)

> **IMPORTANTE:** El buffer de la terminal de comandos tiene un límite de almacenamiento de líneas. Al procesar carpetas con un volumen masivo de imágenes, los primeros resultados impresos en pantalla se perderán visualmente.

Nota: Probar la siguiente función con la auditoría de imágenes para crear un archivo de texto en la carpeta dónde se guarda el reporte de auditoria.

Solo se debe copiar y pegar en 00_auditoria_imagenes.py

```python
# Crea un archivo de texto en la carpeta para guardar el reporte:
with open("reporte_auditoria.txt", "w") as archivo_texto:
    for nombre in lista_archivos:
        archivo_texto.write(nombre + "\n")
