import cv2
import numpy as np
import tensorflow as tf
import customtkinter as ctk
from customtkinter import filedialog
from PIL import Image

# 1. Cargar el modelo entrenado
print("Cargando modelo...")
modelo = tf.keras.models.load_model('modelo_motos_IUJO.keras') 
print("Modelo cargado con éxito.")

# Configuración inicial de customtkinter
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

class DetectorMotosApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Detector de Motos - Panel Dual")
        self.geometry("1300x750")
        
        # Configurar el grid principal de la ventana para que se divida 50/50
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)
        
        # Título principal de la aplicación centrado en la parte superior
        self.label_titulo = ctk.CTkLabel(
            self, 
            text="Detección de Motos con IA", 
            font=ctk.CTkFont(size=32, weight="bold")
        )
        self.label_titulo.grid(row=0, column=0, columnspan=2, pady=(20, 10))
        
        # ==========================================
        # PANEL IZQUIERDO: CÁMARA EN TIEMPO REAL
        # ==========================================
        self.frame_izquierdo = ctk.CTkFrame(self, corner_radius=15)
        self.frame_izquierdo.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        
        self.titulo_izquierdo = ctk.CTkLabel(self.frame_izquierdo, text="Cámara en Vivo", font=ctk.CTkFont(size=24, weight="bold"))
        self.titulo_izquierdo.pack(pady=10)
        
        self.label_video = ctk.CTkLabel(self.frame_izquierdo, text="")
        self.label_video.pack(pady=10, padx=10, expand=True)
        
        self.res_camara_frame = ctk.CTkFrame(self.frame_izquierdo, fg_color="transparent")
        self.res_camara_frame.pack(pady=10, fill="x", padx=20)
        
        self.label_res_camara = ctk.CTkLabel(self.res_camara_frame, text="Iniciando cámara...", font=ctk.CTkFont(size=28, weight="bold"))
        self.label_res_camara.pack()
        
        self.label_prob_camara = ctk.CTkLabel(self.res_camara_frame, text="", font=ctk.CTkFont(size=18), text_color="gray")
        self.label_prob_camara.pack(pady=5)
        
        # ==========================================
        # PANEL DERECHO: IMÁGENES ESTÁTICAS
        # ==========================================
        self.frame_derecho = ctk.CTkFrame(self, corner_radius=15)
        self.frame_derecho.grid(row=1, column=1, padx=20, pady=10, sticky="nsew")
        
        self.titulo_derecho = ctk.CTkLabel(self.frame_derecho, text="Análisis de Imagen Estática", font=ctk.CTkFont(size=24, weight="bold"))
        self.titulo_derecho.pack(pady=10)
        
        self.boton_cargar = ctk.CTkButton(self.frame_derecho, text="Cargar Imagen", command=self.cargar_imagen, font=ctk.CTkFont(size=16, weight="bold"))
        self.boton_cargar.pack(pady=10)
        
        # Un cuadro para mostrar la imagen subida
        self.label_imagen = ctk.CTkLabel(self.frame_derecho, text="Ninguna imagen cargada", font=ctk.CTkFont(size=16), fg_color="#2b2b2b", corner_radius=10)
        self.label_imagen.pack(pady=10, padx=10, expand=True, fill="both")
        
        self.res_imagen_frame = ctk.CTkFrame(self.frame_derecho, fg_color="transparent")
        self.res_imagen_frame.pack(pady=10, fill="x", padx=20)
        
        self.label_res_imagen = ctk.CTkLabel(self.res_imagen_frame, text="-", font=ctk.CTkFont(size=28, weight="bold"))
        self.label_res_imagen.pack()
        
        self.label_prob_imagen = ctk.CTkLabel(self.res_imagen_frame, text="-", font=ctk.CTkFont(size=18), text_color="gray")
        self.label_prob_imagen.pack(pady=5)
        
        # Botón para salir (en el fondo, abarcando ambas columnas)
        self.boton_salir = ctk.CTkButton(
            self, 
            text="Cerrar Aplicación", 
            command=self.on_closing,
            fg_color="#D32F2F",
            hover_color="#B71C1C",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.boton_salir.grid(row=2, column=0, columnspan=2, pady=15)
        
        # Iniciar cámara web
        self.cap = cv2.VideoCapture(0)
        
        # Comenzar el bucle de actualización del video
        self.actualizar_video()

    def evaluar_imagen(self, frame_bgr):
        """Procesa una imagen BGR y retorna los resultados de la predicción."""
        img_redimensionada = cv2.resize(frame_bgr, (224, 224))
        img_rgb = cv2.cvtColor(img_redimensionada, cv2.COLOR_BGR2RGB)
        img_array = np.expand_dims(img_rgb, axis=0)
        
        prediccion = modelo.predict(img_array, verbose=0)[0][0]
        
        if prediccion >= 0.5:
            texto = "SI ES MOTO"
            probabilidad = f"Seguridad: {(prediccion * 100):.2f}%"
            color = "#4CAF50"  # Verde
        else:
            certeza_no_moto = (1 - prediccion) * 100
            texto = "NO ES MOTO"
            probabilidad = f"Seguridad: {certeza_no_moto:.2f}%"
            color = "#F44336"  # Rojo
            
        return texto, probabilidad, color
        
    def actualizar_video(self):
        ret, frame = self.cap.read()
        if ret:
            # Evaluar fotograma
            texto, probabilidad, color = self.evaluar_imagen(frame)
            
            # Actualizar textos
            self.label_res_camara.configure(text=texto, text_color=color)
            self.label_prob_camara.configure(text=probabilidad)
            
            # Renderizar en la interfaz
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)
            img_tk = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(500, 375))
            
            self.label_video.imgtk = img_tk
            self.label_video.configure(image=img_tk)
            
        # Programar la próxima actualización (aprox ~30 FPS)
        self.after(33, self.actualizar_video)

    def cargar_imagen(self):
        file_path = filedialog.askopenfilename(
            title="Seleccionar una Imagen",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            # Cargar la imagen con OpenCV
            img_bgr = cv2.imread(file_path)
            if img_bgr is not None:
                # Evaluar
                texto, probabilidad, color = self.evaluar_imagen(img_bgr)
                
                # Actualizar etiquetas
                self.label_res_imagen.configure(text=texto, text_color=color)
                self.label_prob_imagen.configure(text=probabilidad)
                
                # Renderizar en la interfaz de la derecha
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                
                # Redimensionar para que encaje bien en el panel sin perder proporción
                # Usaremos un tamaño máximo de 500x375 similar al video
                img_pil.thumbnail((500, 375))
                img_tk = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=img_pil.size)
                
                self.label_imagen.imgtk = img_tk
                self.label_imagen.configure(image=img_tk, text="")
            else:
                self.label_res_imagen.configure(text="Error al cargar", text_color="red")
                self.label_prob_imagen.configure(text="")
        
    def on_closing(self):
        self.cap.release()
        self.destroy()

if __name__ == "__main__":
    app = DetectorMotosApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
