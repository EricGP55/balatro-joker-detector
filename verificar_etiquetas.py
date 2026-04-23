import cv2
import os
import random
import re

# ==========================================
# CONFIGURACIÓN
# ==========================================
BASE_DIR = "Dataset_YOLO_Cartas"
IMG_DIR = os.path.join(BASE_DIR, "images/train")
LABEL_DIR = os.path.join(BASE_DIR, "labels/train")
YAML_FILE = os.path.join(BASE_DIR, "data.yaml")

def cargar_nombres_clases(yaml_path):
    """
    Lee el archivo data.yaml para saber qué nombre corresponde a cada número.
    Busca una línea que empiece por "names: [...]"
    """
    nombres = {}
    if not os.path.exists(yaml_path):
        print(f"⚠️ AVISO: No encuentro {yaml_path}. Solo verás números.")
        return nombres

    try:
        with open(yaml_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            if "names:" in line:
                # Extraer la lista de nombres. Ejemplo: names: ['as', 'rey']
                # Limpiamos corchetes y comillas
                content = line.split("names:")[1].strip()
                content = content.replace("[", "").replace("]", "").replace("'", "").replace('"', "")
                lista_nombres = [n.strip() for n in content.split(",")]
                
                # Convertir a diccionario {0: 'as', 1: 'rey'}
                for idx, nombre in enumerate(lista_nombres):
                    nombres[idx] = nombre
                break
        
        print(f"✅ Cargados {len(nombres)} nombres de cartas desde data.yaml")
        return nombres
    except Exception as e:
        print(f"⚠️ Error leyendo data.yaml: {e}")
        return {}

def dibujar_yolo(img, labels, nombres_dict):
    h_img, w_img, _ = img.shape
    
    for line in labels:
        parts = line.strip().split()
        if len(parts) < 5: continue

        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

        # --- Desnormalizar coordenadas ---
        x_c_px = int(x_center * w_img)
        y_c_px = int(y_center * h_img)
        w_px = int(width * w_img)
        h_px = int(height * h_img)
        
        x_min = int(x_c_px - (w_px / 2))
        y_min = int(y_c_px - (h_px / 2))
        x_max = int(x_c_px + (w_px / 2))
        y_max = int(y_c_px + (h_px / 2))

        # --- DIBUJAR ---
        # 1. Rectángulo Verde
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        # 2. Obtener el NOMBRE REAL
        nombre_carta = nombres_dict.get(class_id, f"ID: {class_id}")
        
        # 3. Fondo negro para el texto (para que se lea bien)
        (text_w, text_h), _ = cv2.getTextSize(nombre_carta, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x_min, y_min - 20), (x_min + text_w, y_min), (0, 255, 0), -1)
        
        # 4. Texto Negro sobre fondo verde
        cv2.putText(img, nombre_carta, (x_min, y_min - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return img

def main():
    if not os.path.exists(IMG_DIR):
        print(f"❌ No encuentro la carpeta: {IMG_DIR}")
        return

    # 1. Cargar el diccionario de nombres
    nombres_dict = cargar_nombres_clases(YAML_FILE)

    # 2. Listar imágenes
    files = [f for f in os.listdir(IMG_DIR) if f.endswith(('.jpg', '.png'))]
    
    if not files:
        print("❌ No hay imágenes en la carpeta train.")
        return

    random.shuffle(files) # Mezclar

    print(f"\n🔍 Visualizador Listo. Mostrando {len(files)} imágenes.")
    print("--------------------------------------------------")
    print(" 👉 Presiona [ESPACIO] para ver la siguiente imagen.")
    print(" 👉 Presiona [Q] para salir.")
    print("--------------------------------------------------")

    for filename in files:
        img_path = os.path.join(IMG_DIR, filename)
        txt_filename = filename.rsplit('.', 1)[0] + ".txt"
        label_path = os.path.join(LABEL_DIR, txt_filename)

        img = cv2.imread(img_path)
        if img is None: continue

        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                labels = f.readlines()
        
        # Dibujamos pasando el diccionario de nombres
        img_con_boxes = dibujar_yolo(img, labels, nombres_dict)

        cv2.imshow("Verificar Etiquetas (ESPACIO: Siguiente, Q: Salir)", img_con_boxes)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()