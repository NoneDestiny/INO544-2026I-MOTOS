import cv2
import numpy as np
import tensorflow as tf
import customtkinter as ctk
from PIL import Image

# 1. De este pequeño script se llama al modelo entrenado.
print("Cargando modelo...")
modelo = tf.keras.models.load_model('modelo_motos_IUJO.keras') 
print("Modelo cargado con éxito.")

# Configuración inicial de customtkinter para un diseño oscuro y limpio
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

class DetectorMotosApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Detector de Motos en Tiempo Real")
        self.geometry("900x700")
        
        # Título principal de la aplicación
        self.label_titulo = ctk.CTkLabel(
            self, 
            text="Detección de Motos con IA", 
            font=ctk.CTkFont(size=28, weight="bold")
        )
        self.label_titulo.pack(pady=(20, 10))
        
        # Contenedor para centrar el video y darle un borde elegante
        self.frame_video = ctk.CTkFrame(self, corner_radius=15)
        self.frame_video.pack(pady=10, padx=20)
        
        self.label_video = ctk.CTkLabel(self.frame_video, text="")
        self.label_video.pack(pady=10, padx=10)
        
        # Panel inferior para mostrar el resultado de manera clara
        self.frame_resultado = ctk.CTkFrame(self, fg_color="transparent")
        self.frame_resultado.pack(pady=(10, 20), fill="x", padx=50)
        
        self.label_resultado = ctk.CTkLabel(
            self.frame_resultado, 
            text="Iniciando cámara...", 
            font=ctk.CTkFont(size=32, weight="bold")
        )
        self.label_resultado.pack()
        
        self.label_probabilidad = ctk.CTkLabel(
            self.frame_resultado, 
            text="", 
            font=ctk.CTkFont(size=18),
            text_color="gray"
        )
        self.label_probabilidad.pack(pady=5)
        
        # Botón para salir
        self.boton_salir = ctk.CTkButton(
            self, 
            text="Cerrar", 
            command=self.on_closing,
            fg_color="#D32F2F",
            hover_color="#B71C1C"
        )
        self.boton_salir.pack(pady=10)
        
        # Iniciar cámara web
        self.cap = cv2.VideoCapture(0)
        
        # Comenzar el bucle de actualización del video
        self.actualizar_video()
        
    def actualizar_video(self):
        ret, frame = self.cap.read()
        if ret:
            # --- PREPARACION DE LA IMAGEN PARA LA IA ---
            img_redimensionada = cv2.resize(frame, (224, 224))
            img_rgb = cv2.cvtColor(img_redimensionada, cv2.COLOR_BGR2RGB)
            img_array = np.expand_dims(img_rgb, axis=0)
            
            # --- PREDICCIÓN ---
            prediccion = modelo.predict(img_array, verbose=0)[0][0]
            
            # --- INTERPRETACION ---
            if prediccion >= 0.5:
                texto = "SI ES MOTO"
                probabilidad = f"Seguridad: {(prediccion * 100):.2f}%"
                color = "#4CAF50"  # Verde minimalista
            else:
                certeza_no_moto = (1 - prediccion) * 100
                texto = "NO ES MOTO"
                probabilidad = f"Seguridad: {certeza_no_moto:.2f}%"
                color = "#F44336"  # Rojo minimalista
                
            # Actualizar textos y colores
            self.label_resultado.configure(text=texto, text_color=color)
            self.label_probabilidad.configure(text=probabilidad)
            
            # --- RENDERIZADO EN CUSTOMTKINTER ---
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)
            
            # Usar CTkImage para una correcta visualización en escalado
            img_tk = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(640, 480))
            
            self.label_video.imgtk = img_tk # Mantener referencia
            self.label_video.configure(image=img_tk)
            
        # Programar la próxima actualización del fotograma (aprox ~30 FPS -> 33 ms)
        self.after(33, self.actualizar_video)
        
    def on_closing(self):
        self.cap.release()
        self.destroy()

if __name__ == "__main__":
    app = DetectorMotosApp()
    # Capturar el evento de la "X" en la ventana
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
