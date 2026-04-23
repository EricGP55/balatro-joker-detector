import tkinter as tk
from ultralytics import YOLO
import cv2
import numpy as np
import mss
import platform

# ==========================================
# ⚙️ CONFIGURACIÓN
# ==========================================
MODEL_PATH = "Entrenamiento_Cartas/run_principal/weights/last.pt"
CONFIDENCE = 0.5  # Solo mostrar si está 50% seguro

# IMPORTANTE PARA MAC:
# Las pantallas Retina tienen el doble de píxeles de lo que dice el sistema.
# Si ves que captura solo una esquina de lo que quieres, cambia esto a 2.0
RETINA_SCALE = 2.0 

class ScreenScanner:
    def __init__(self):
        # 1. Cargar Modelo
        print("Cargando modelo...")
        try:
            self.model = YOLO(MODEL_PATH)
        except Exception as e:
            print(f"Error cargando modelo: {e}")
            exit()

        # 2. Configurar Capturadora de Pantalla
        self.sct = mss.mss()

        # 3. Crear Ventana "Escáner" (Tkinter)
        self.root = tk.Tk()
        self.root.title("🎯 ESCÁNER (Arrastra esto)")
        
        # Posición y tamaño inicial
        self.root.geometry("640x360+100+100")
        
        # Hacerla semitransparente (Para que veas dónde apuntas)
        # En Mac, 0.3 significa 30% visible.
        self.root.attributes("-alpha", 0.3) 
        
        # Asegurar que esté siempre encima de otras ventanas
        self.root.attributes("-topmost", True)
        
        # Evento para cerrar correctamente
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.running = True

        # Iniciar bucle principal
        self.loop()

    def on_close(self):
        self.running = False
        self.root.destroy()

    def loop(self):
        while self.running:
            try:
                # Actualizar la ventana de Tkinter (para que se pueda mover)
                self.root.update()

                # --- 1. Obtener coordenadas de la ventana ---
                x = self.root.winfo_rootx()
                y = self.root.winfo_rooty()
                w = self.root.winfo_width()
                h = self.root.winfo_height()

                # Ajuste para Mac Retina (Multiplicar por 2 si es necesario)
                monitor = {
                    "top": int(y * RETINA_SCALE),
                    "left": int(x * RETINA_SCALE),
                    "width": int(w * RETINA_SCALE),
                    "height": int(h * RETINA_SCALE)
                }

                # --- 2. Capturar Pantalla ---
                # sct.grab devuelve BGRA, necesitamos quitar la A (Alpha)
                screenshot = np.array(self.sct.grab(monitor))
                
                # Convertir de BGRA a BGR (formato estándar OpenCV/YOLO)
                frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

                # --- 3. Predicción YOLO ---
                results = self.model.predict(frame, conf=CONFIDENCE, verbose=False)
                
                # Dibujar resultados
                annotated_frame = results[0].plot()

                # --- 4. Mostrar Ventana de Predicción ---
                cv2.imshow("🤖 VISTA DEL MODELO (Presiona Q para salir)", annotated_frame)

                # Salir si pulsan Q en la ventana de OpenCV
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.on_close()
                    break

            except tk.TclError:
                # Si se cierra la ventana de Tkinter a la fuerza
                break
            except Exception as e:
                print(f"Error en el bucle: {e}")
                break
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    scanner = ScreenScanner()