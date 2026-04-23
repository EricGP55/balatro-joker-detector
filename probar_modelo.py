import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
import os

# ==========================================
# CONFIGURACIÓN
# ==========================================
# Asegúrate de que esta ruta coincida con donde se guardó tu entrenamiento
MODEL_PATH = "Entrenamiento_Cartas/run_principal/weights/last.pt"

print(f"Cargando modelo desde: {MODEL_PATH}...")

# Comprobación de seguridad
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ No encuentro el archivo: {MODEL_PATH}. ¿Seguro que terminaste de entrenar?")

# Cargar el modelo
model = YOLO(MODEL_PATH)

# ==========================================
# FUNCIÓN DE PREDICCIÓN
# ==========================================
def detectar_cartas(imagen, confianza):
    if imagen is None:
        return None
    
    # 1. Predicción
    results = model.predict(imagen, conf=confianza)
    
    # 2. Dibujar recuadros
    # En versiones recientes de Ultralytics + Gradio, a veces plot()
    # mantiene el formato original de entrada (RGB).
    imagen_resultante = results[0].plot()
    
    # 3. SOLUCIÓN:
    # Prueba devolviendo la imagen directamente sin convertir.
    # Si al hacer esto los colores se ven BIEN, borra la línea de cvtColor.
    # Si al hacer esto los colores se ven MAL (azules), entonces DESCOMENTA la línea de abajo.
    
    # imagen_rgb = cv2.cvtColor(imagen_resultante, cv2.COLOR_BGR2RGB) 
    
    return imagen_resultante

# ==========================================
# INTERFAZ DE GRADIO
# ==========================================
demo = gr.Interface(
    fn=detectar_cartas, # La función que creamos arriba
    
    # Entradas: Una imagen y un slider
    inputs=[
        gr.Image(label="Sube una foto o usa la Webcam", sources=["upload", "webcam"]),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.25, label="Nivel de Confianza (Sensibilidad)")
    ],
    
    # Salida: La imagen pintada
    outputs=gr.Image(label="Detección YOLO"),
    
    title="🃏 Detector de Cartas con YOLOv8",
    description="Sube una imagen de tus cartas para ver si el modelo las reconoce."
)

# Ejecutar la app
if __name__ == "__main__":
    print("✅ Abriendo interfaz web...")
    demo.launch()