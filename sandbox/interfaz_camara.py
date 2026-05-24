import cv2
import numpy as np
import tensorflow as tf
import customtkinter as ctk
from customtkinter import filedialog
from PIL import Image

# ---------------------------------------------------------------------------
# PALETA DE COLORES
# ---------------------------------------------------------------------------
COLOR_BG          = "#0f1117"   # Fondo principal, negro azulado profundo
COLOR_PANEL       = "#1a1d27"   # Fondo de paneles
COLOR_PANEL_BORDE = "#252836"   # Borde sutil de paneles
COLOR_ACENTO      = "#6c63ff"   # Violeta/índigo — color de acento principal
COLOR_ACENTO_HOV  = "#574fd6"   # Acento al hacer hover
COLOR_TEXTO       = "#e8e8f0"   # Texto principal
COLOR_TEXTO_SUB   = "#6b7280"   # Texto secundario / subtítulos
COLOR_VERDE       = "#22c55e"   # Resultado positivo
COLOR_ROJO        = "#ef4444"   # Resultado negativo
COLOR_SEPARADOR   = "#2a2d3e"   # Línea separadora entre paneles

# ---------------------------------------------------------------------------
# FUENTES
# ---------------------------------------------------------------------------
FUENTE_TITULO_APP  = ("Segoe UI", 22, "bold")
FUENTE_SUBTITULO   = ("Segoe UI", 13)
FUENTE_PANEL_TIT   = ("Segoe UI", 15, "bold")
FUENTE_BADGE       = ("Segoe UI", 20, "bold")
FUENTE_PORCENTAJE  = ("Segoe UI", 13)
FUENTE_BOTON       = ("Segoe UI", 13, "bold")
FUENTE_PLACEHOLDER = ("Segoe UI", 13)

# ---------------------------------------------------------------------------
# MODELO
# ---------------------------------------------------------------------------
print("Cargando modelo...")
modelo = tf.keras.models.load_model('modelo_motos_IUJO.keras')
print("Modelo cargado con éxito.")

# Tema base de customtkinter
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")


class DetectorMotosApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Detector de Motos — IUJO")
        self.geometry("1340x760")
        self.minsize(1100, 680)
        self.configure(fg_color=COLOR_BG)

        # Columnas: margen-izq | panel-izq | separador | panel-der | margen-der
        self.grid_columnconfigure(0, weight=0, minsize=20)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=0, minsize=1)
        self.grid_columnconfigure(3, weight=1)
        self.grid_columnconfigure(4, weight=0, minsize=20)
        self.grid_rowconfigure(0, weight=0)  # Cabecera
        self.grid_rowconfigure(1, weight=1)  # Paneles
        self.grid_rowconfigure(2, weight=0)  # Pie

        # =====================================================================
        # CABECERA
        # =====================================================================
        cabecera = ctk.CTkFrame(self, fg_color="transparent")
        cabecera.grid(row=0, column=0, columnspan=5, sticky="ew", pady=(24, 14), padx=28)
        cabecera.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            cabecera,
            text="  Detector de Motos con IA",
            font=ctk.CTkFont(family="Segoe UI", size=22, weight="bold"),
            text_color=COLOR_TEXTO,
            anchor="w"
        ).grid(row=0, column=0, sticky="w")

        ctk.CTkLabel(
            cabecera,
            text="Modelo MobileNetV2  •  Tiempo Real + Imágenes Estáticas",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color=COLOR_TEXTO_SUB,
            anchor="w"
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        # Línea separadora horizontal bajo la cabecera
        sep_h = ctk.CTkFrame(self, height=1, fg_color=COLOR_SEPARADOR)
        sep_h.grid(row=0, column=0, columnspan=5, sticky="sew", pady=(0, 0))

        # =====================================================================
        # PANEL IZQUIERDO — CÁMARA EN VIVO
        # =====================================================================
        self.frame_izq = ctk.CTkFrame(
            self,
            fg_color=COLOR_PANEL,
            corner_radius=14,
            border_width=1,
            border_color=COLOR_PANEL_BORDE
        )
        self.frame_izq.grid(row=1, column=1, padx=(0, 10), pady=8, sticky="nsew")
        self.frame_izq.grid_rowconfigure(1, weight=1)
        self.frame_izq.grid_columnconfigure(0, weight=1)

        # --- Encabezado del panel izquierdo ---
        header_izq = ctk.CTkFrame(self.frame_izq, fg_color="transparent")
        header_izq.grid(row=0, column=0, sticky="ew", padx=20, pady=(18, 8))

        # Punto de acento verde (indicador de "live")
        ctk.CTkLabel(
            header_izq,
            text="●",
            font=ctk.CTkFont(size=11),
            text_color="#22c55e"
        ).pack(side="left", padx=(0, 6))

        ctk.CTkLabel(
            header_izq,
            text="Cámara en Vivo",
            font=ctk.CTkFont(family="Segoe UI", size=15, weight="bold"),
            text_color=COLOR_TEXTO
        ).pack(side="left")

        # --- Área de video ---
        self.label_video = ctk.CTkLabel(
            self.frame_izq,
            text="",
            fg_color=COLOR_BG,
            corner_radius=10
        )
        self.label_video.grid(row=1, column=0, padx=16, pady=8, sticky="nsew")

        # --- Badge de resultado de la cámara ---
        self.badge_camara = ctk.CTkFrame(
            self.frame_izq,
            fg_color=COLOR_BG,
            corner_radius=10
        )
        self.badge_camara.grid(row=2, column=0, padx=16, pady=(6, 18), sticky="ew")
        self.badge_camara.grid_columnconfigure(0, weight=1)

        self.label_res_camara = ctk.CTkLabel(
            self.badge_camara,
            text="Iniciando cámara...",
            font=ctk.CTkFont(family="Segoe UI", size=20, weight="bold"),
            text_color=COLOR_TEXTO_SUB
        )
        self.label_res_camara.grid(row=0, column=0, pady=(12, 2))

        self.label_prob_camara = ctk.CTkLabel(
            self.badge_camara,
            text="",
            font=ctk.CTkFont(family="Segoe UI", size=13),
            text_color=COLOR_TEXTO_SUB
        )
        self.label_prob_camara.grid(row=1, column=0, pady=(0, 12))

        # =====================================================================
        # SEPARADOR VERTICAL
        # =====================================================================
        sep_v = ctk.CTkFrame(self, width=1, fg_color=COLOR_SEPARADOR)
        sep_v.grid(row=1, column=2, padx=0, pady=8, sticky="ns")

        # =====================================================================
        # PANEL DERECHO — IMÁGENES ESTÁTICAS
        # =====================================================================
        self.frame_der = ctk.CTkFrame(
            self,
            fg_color=COLOR_PANEL,
            corner_radius=14,
            border_width=1,
            border_color=COLOR_PANEL_BORDE
        )
        self.frame_der.grid(row=1, column=3, padx=(10, 0), pady=8, sticky="nsew")
        self.frame_der.grid_rowconfigure(1, weight=1)
        self.frame_der.grid_columnconfigure(0, weight=1)

        # --- Encabezado del panel derecho ---
        header_der = ctk.CTkFrame(self.frame_der, fg_color="transparent")
        header_der.grid(row=0, column=0, sticky="ew", padx=20, pady=(18, 8))

        ctk.CTkLabel(
            header_der,
            text="◈",
            font=ctk.CTkFont(size=14),
            text_color=COLOR_ACENTO
        ).pack(side="left", padx=(0, 6))

        ctk.CTkLabel(
            header_der,
            text="Imagen Estática",
            font=ctk.CTkFont(family="Segoe UI", size=15, weight="bold"),
            text_color=COLOR_TEXTO
        ).pack(side="left")

        # Botón de carga — alineado al lado derecho del encabezado
        self.boton_cargar = ctk.CTkButton(
            header_der,
            text="+ Cargar Imagen",
            command=self.cargar_imagen,
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            fg_color=COLOR_ACENTO,
            hover_color=COLOR_ACENTO_HOV,
            corner_radius=8,
            height=32,
            width=140
        )
        self.boton_cargar.pack(side="right")

        # --- Área de imagen estática ---
        self.label_imagen = ctk.CTkLabel(
            self.frame_der,
            text="Ninguna imagen cargada\n\nPresiona '+ Cargar Imagen' para comenzar",
            font=ctk.CTkFont(family="Segoe UI", size=13),
            text_color=COLOR_TEXTO_SUB,
            fg_color=COLOR_BG,
            corner_radius=10
        )
        self.label_imagen.grid(row=1, column=0, padx=16, pady=8, sticky="nsew")

        # --- Badge de resultado de imagen estática ---
        self.badge_imagen = ctk.CTkFrame(
            self.frame_der,
            fg_color=COLOR_BG,
            corner_radius=10
        )
        self.badge_imagen.grid(row=2, column=0, padx=16, pady=(6, 18), sticky="ew")
        self.badge_imagen.grid_columnconfigure(0, weight=1)

        self.label_res_imagen = ctk.CTkLabel(
            self.badge_imagen,
            text="Sin análisis",
            font=ctk.CTkFont(family="Segoe UI", size=20, weight="bold"),
            text_color=COLOR_TEXTO_SUB
        )
        self.label_res_imagen.grid(row=0, column=0, pady=(12, 2))

        self.label_prob_imagen = ctk.CTkLabel(
            self.badge_imagen,
            text="Carga una imagen para obtener el resultado",
            font=ctk.CTkFont(family="Segoe UI", size=13),
            text_color=COLOR_TEXTO_SUB
        )
        self.label_prob_imagen.grid(row=1, column=0, pady=(0, 12))

        # =====================================================================
        # PIE — BOTÓN CERRAR
        # =====================================================================
        pie = ctk.CTkFrame(self, fg_color="transparent")
        pie.grid(row=2, column=0, columnspan=5, pady=(10, 18))

        self.boton_salir = ctk.CTkButton(
            pie,
            text="Cerrar Aplicación",
            command=self.on_closing,
            fg_color="#2a2d3e",
            hover_color="#3a3d50",
            text_color=COLOR_ROJO,
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            corner_radius=8,
            height=36,
            width=180,
            border_width=1,
            border_color=COLOR_ROJO
        )
        self.boton_salir.pack()

        # =====================================================================
        # INICIAR CÁMARA
        # =====================================================================
        self.cap = cv2.VideoCapture(0)
        self.actualizar_video()

    # -------------------------------------------------------------------------
    # LÓGICA DE IA (reutilizable)
    # -------------------------------------------------------------------------
    def evaluar_imagen(self, frame_bgr):
        """Procesa una imagen BGR y retorna texto, probabilidad y color del resultado."""
        img_redimensionada = cv2.resize(frame_bgr, (224, 224))
        img_rgb = cv2.cvtColor(img_redimensionada, cv2.COLOR_BGR2RGB)
        img_array = np.expand_dims(img_rgb, axis=0)

        prediccion = modelo.predict(img_array, verbose=0)[0][0]

        if prediccion >= 0.5:
            texto      = "✓  ES MOTO"
            porcentaje = f"Seguridad: {prediccion * 100:.1f}%"
            color      = COLOR_VERDE
        else:
            certeza    = (1 - prediccion) * 100
            texto      = "✗  NO ES MOTO"
            porcentaje = f"Seguridad: {certeza:.1f}%"
            color      = COLOR_ROJO

        return texto, porcentaje, color

    # -------------------------------------------------------------------------
    # BUCLE DE VÍDEO
    # -------------------------------------------------------------------------
    def actualizar_video(self):
        ret, frame = self.cap.read()
        if ret:
            texto, porcentaje, color = self.evaluar_imagen(frame)
            self.label_res_camara.configure(text=texto, text_color=color)
            self.label_prob_camara.configure(text=porcentaje, text_color=color)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil   = Image.fromarray(frame_rgb)
            img_tk    = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(500, 375))

            self.label_video.imgtk = img_tk
            self.label_video.configure(image=img_tk)

        self.after(33, self.actualizar_video)

    # -------------------------------------------------------------------------
    # CARGA DE IMAGEN ESTÁTICA
    # -------------------------------------------------------------------------
    def cargar_imagen(self):
        file_path = filedialog.askopenfilename(
            title="Seleccionar una Imagen",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if not file_path:
            return

        img_bgr = cv2.imread(file_path)
        if img_bgr is None:
            self.label_res_imagen.configure(text="Error al cargar la imagen", text_color=COLOR_ROJO)
            self.label_prob_imagen.configure(text="Verifica que el archivo sea válido", text_color=COLOR_TEXTO_SUB)
            return

        texto, porcentaje, color = self.evaluar_imagen(img_bgr)
        self.label_res_imagen.configure(text=texto, text_color=color)
        self.label_prob_imagen.configure(text=porcentaje, text_color=color)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil.thumbnail((500, 375))
        img_tk = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=img_pil.size)

        self.label_imagen.imgtk = img_tk
        self.label_imagen.configure(image=img_tk, text="")

    # -------------------------------------------------------------------------
    def on_closing(self):
        self.cap.release()
        self.destroy()


if __name__ == "__main__":
    app = DetectorMotosApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
