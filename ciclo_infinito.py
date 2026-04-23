import os
import shutil
import subprocess
import time
import sys

# ==========================================
# CONFIGURACIÓN
# ==========================================
CICLOS_A_EJECUTAR = 125
# He subido esto a 4 horas (14400s) para cubrir tus 3.5h
TIMEOUT_ENTRENAMIENTO = 600 

def limpiar_dataset():
    """Borra las imágenes para liberar espacio."""
    dataset_path = "Dataset_YOLO_Cartas"
    if os.path.exists(dataset_path):
        try:
            shutil.rmtree(dataset_path)
            print("   🗑️  Dataset eliminado para ahorrar espacio.")
        except Exception as e:
            print(f"   ⚠️  No se pudo borrar carpeta (puede estar en uso): {e}")

def main():
    print(f"--- 🌙 INICIANDO MODO NOCTURNO: {CICLOS_A_EJECUTAR} CICLOS ---")
    print(f"--- ⚠️  TIMEOUT ajustado a {TIMEOUT_ENTRENAMIENTO/3600:.1f} horas por ciclo ---")
    
    start_total = time.time()

    for i in range(1, CICLOS_A_EJECUTAR + 1):
        ciclo_start = time.time()
        print(f"\n" + "="*60)
        print(f"   🔄 CICLO {i} DE {CICLOS_A_EJECUTAR} | Hora: {time.strftime('%H:%M:%S')}")
        print("="*60)

        # 1. Limpieza inicial (Por seguridad)
        limpiar_dataset()

        # 2. Generar
        print("\n[1/3] 🎨 Generando imágenes...")
        try:
            # Esperamos a que termine de generar
            subprocess.run([sys.executable, "generar_dataset.py"], check=True)
        except Exception as e:
            print(f"❌ Error generando imágenes: {e}")
            continue # Saltar al siguiente ciclo si falla este

        # 3. Entrenar
        print("\n[2/3] 🧠 Entrenando (Paciencia, esto tarda)...")
        try:
            # Aquí lanzamos el entrenamiento
            subprocess.run(
                [sys.executable, "entrenar.py"], 
                check=True,
                timeout=TIMEOUT_ENTRENAMIENTO
            )
            print("✅ Entrenamiento del ciclo finalizado.")
            
        except subprocess.TimeoutExpired:
            print(f"\n⏰ ¡TIMEOUT! Pasaron {TIMEOUT_ENTRENAMIENTO}s.")
            print("🔪 Se cortó el entrenamiento (los pesos se guardaron). Pasando al siguiente.")
        except Exception as e:
            print(f"❌ Error en entrenamiento: {e}")

        # 4. Limpieza final
        print("\n[3/3] 🧹 Limpiando disco...")
        limpiar_dataset()
        
        duracion = (time.time() - ciclo_start) / 60
        print(f"✨ Ciclo {i} completado en {duracion:.1f} minutos.")

    total_horas = (time.time() - start_total) / 3600
    print(f"\n🎉 ¡PROCESO TERMINADO! Tiempo total: {total_horas:.2f} horas.")

if __name__ == "__main__":
    main()