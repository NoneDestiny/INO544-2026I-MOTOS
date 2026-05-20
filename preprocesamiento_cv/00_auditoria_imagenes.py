# Aqui se hace la debida auditoria para verificar que todos los archivos cumplan su debido parametro.
# En este caso se usa para comprobar que la resolucion, los canales RGB y su formato esten de acuerdo a lo solicitado.
# Como el buffer del terminal tiene un limite, al usarse en carpetas con una gran cantidad de archivos se pierde un porcentaje
# De informacion al inicio y no se termina leyendo.
# Si se desea leer la informacion o los resultados, se puede guardar en un archivo de texto para su revision de esta manera:

# Crea un archivo de texto en tu carpeta:
"""
with open("reporte_auditoria.txt", "w") as archivo_texto:
    for nombre in lista_archivos:
        archivo_texto.write(nombre + "\n")
"""

print("¡Auditoría completada! Revisa el archivo reporte_auditoria.txt")

import os
from PIL import Image

# 1. Descubrimos la ruta exacta donde está guardado ESTE script
directorio_actual = os.path.dirname(os.path.abspath(__file__))

# 2. Construimos la ruta hacia la carpeta uniendo el directorio del script con el nombre de la carpeta y usamos ".." para
# Devolverse en las carpetas.
ruta_dataset = os.path.abspath(os.path.join(directorio_actual, "..", "datasets", "dataset_final", "dataset_no_motos"))

print("--- INICIANDO AUDITORÍA DEL DATASET ---")
print(f"Buscando imágenes en: {ruta_dataset}\n")

# 3. Ahora usamos nuestra la ruta para el bucle de verificación
lista_archivos = os.listdir(ruta_dataset)

for nombre_archivo in lista_archivos:
    # También unimos la ruta dinámicamente para cada archivo individual
    ruta_completa = os.path.join(ruta_dataset, nombre_archivo)
    
    try:
        # Abrimos la imagen
        img = Image.open(ruta_completa)
        
        # Extraemos los datos exactos
        formato = img.format           # Debería ser JPEG
        modo = img.mode                # Debería ser RGB
        ancho, alto = img.size         # Debería ser 224, 224
        
        # Validamos con condicionales (if)
        estado = "PERFECTA"
        errores = []
        
        if formato != "JPEG": # En Pillow, los .jpg se llaman JPEG
            errores.append("No es JPG")
        if modo != "RGB":
            errores.append("No es RGB")
        if ancho != 224 or alto != 224:
            errores.append(f"Resolución incorrecta ({ancho}x{alto})")
            
        if len(errores) > 0:
            estado = f"ERROR: {errores}"
            
        print(f"[{nombre_archivo}] -> {estado}")
        
    except Exception as e:
        print(f"[{nombre_archivo}] -> Archivo corrupto o no es imagen.")

print("\n--- AUDITORÍA FINALIZADA ---")