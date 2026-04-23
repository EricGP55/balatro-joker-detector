import os
import torch
from ultralytics import YOLO

# ==========================================
# ⚙️ CONFIGURACIÓN DEL PROYECTO
# ==========================================

# Ruta al archivo data.yaml (Asegúrate de que la ruta sea correcta)
DATASET_YAML = os.path.join("Dataset_YOLO_Cartas", "data.yaml")

# Carpeta y nombre del entrenamiento
PROJECT_NAME = "Entrenamiento_Cartas"
RUN_NAME = "run_principal"

# Ruta de los pesos (para reanudar entrenamiento)
LAST_WEIGHTS = os.path.join(PROJECT_NAME, RUN_NAME, "weights", "last.pt")

# PARÁMETROS DE ENTRENAMIENTO
# ---------------------------
EPOCHS_PER_CYCLE = 10   # Pocas épocas para ciclos rápidos
IMG_SIZE = 640          # Tamaño estándar
BATCH_SIZE = 8          # 8 es seguro para Mac M1 de 8GB. Si falla, baja a 4.

def main():
    # 1. DIAGNÓSTICO DE HARDWARE
    print("\n" + "="*40)
    if torch.backends.mps.is_available():
        device_target = 'mps'
        print("🚀 MAC M1/M2 DETECTADO: Usando GPU (Metal Performance Shaders)")
        print("✅ Modo Turbo activado.")
    else:
        device_target = 'cpu'
        print("⚠️ GPU NO DETECTADA: Usando CPU (Más lento)")
    print("="*40 + "\n")

    # 2. CARGAR MODELO (Inteligencia Incremental)
    if os.path.exists(LAST_WEIGHTS):
        print(f"🔄 ENCONTRADO MODELO PREVIO: {LAST_WEIGHTS}")
        print("   -> Continuando el aprendizaje donde se quedó...")
        model = YOLO(LAST_WEIGHTS)
    else:
        print("🆕 NO HAY ENTRENAMIENTO PREVIO")
        print("   -> Empezando desde CERO con yolov8n.pt...")
        model = YOLO("yolov8n.pt") # Modelo Nano (el más ligero)

    # 3. EJECUTAR ENTRENAMIENTO OPTIMIZADO
    print(f"\n>>> Iniciando Tanda de {EPOCHS_PER_CYCLE} Épocas...")
    print(">>> Configuracion: Sin caché (RAM segura), Sin validación (Velocidad máxima)\n")

    model.train(
        data=DATASET_YAML,
        epochs=EPOCHS_PER_CYCLE,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        
        # Guardado de archivos
        project=PROJECT_NAME, 
        name=RUN_NAME,        
        exist_ok=True,        # Sobrescribir en la misma carpeta
        save=True,            # Guardar last.pt y best.pt
        
        # --- 🔧 OPTIMIZACIONES CRÍTICAS PARA MAC M1 (8GB RAM) ---
        device=device_target, # Usa 'mps' (GPU)
        workers=0,            # OBLIGATORIO en Mac para evitar bloqueos
        
        # --- 💾 SALVVIDAS DE MEMORIA RAM ---
        cache=False,          # <--- IMPORTANTE: Lee del SSD, no llena la RAM (Evita colapso)
        
        # --- ⚡ MODO VELOCIDAD (Sin distracciones) ---
        val=False,            # No validar en cada época (Evita error NMS y lentitud)
        plots=False,          # No generar gráficos PNG (Ahorra CPU)
        amp=True              # Precisión mixta (Más rápido en M1)
    )

    print("\n" + "="*40)
    print("✅ TANDA TERMINADA CON ÉXITO")
    print(f"🧠 Cerebro guardado en: {LAST_WEIGHTS}")
    print("👉 Siguiente paso: Borrar imágenes viejas -> Generar nuevas -> Volver a ejecutar este script.")
    print("="*40 + "\n")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Entrenamiento detenido manualmente.")
    except Exception as e:
        print(f"\n❌ Ocurrió un error: {e}")