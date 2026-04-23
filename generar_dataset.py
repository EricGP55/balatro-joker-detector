import cv2
import numpy as np
import os
import random
import shutil
import json
from tqdm import tqdm

# ==========================================
# ⚙️ CONFIGURACIÓN GENERAL
# ==========================================

CARDS_DIR = "Comodines"
BACKGROUNDS_DIR = "Fondos"
OUTPUT_BASE_DIR = "Dataset_YOLO_Cartas"
STATS_FILE = "card_stats.json"

TOTAL_IMAGES_TO_GENERATE = 250
TRAIN_RATIO = 0.8
IMAGE_EXTS = ['.jpg', '.jpeg', '.png', '.bmp']

# ==========================================
# 📏 CONFIGURACIÓN DE ZOOM EXTREMO
# ==========================================

# RANGO DE ESCALA (Respecto al ancho de la imagen)
# 0.05 = 5% del ancho (DIMINUTA, vista de mesa completa)
# 0.50 = 50% del ancho (GIGANTE, primer plano)
CAMERA_ZOOM_RANGE = (0.05, 0.50)

# Factor para las cartas "Mini" (Wee Joker)
# Serán un 60% del tamaño de las normales en esa foto
MINI_CARD_FACTOR = 0.60 

# Lista de cartas que son físicamente más pequeñas
ESCALAS_ESPECIALES = [
    "Wee_Joker",
    # Añade aquí otros nombres si tienes
]

# Solapamiento máximo (20%)
MAX_OVERLAP_ALLOWED = 0.20 

# ==========================================
# 🧮 FUNCIONES MATEMÁTICAS
# ==========================================

def ensure_dir(directory):
    if not os.path.exists(directory): os.makedirs(directory)

def load_images_from_folder(folder, load_with_alpha=False):
    images, filenames = [], []
    if not os.path.exists(folder):
        print(f"❌ Error: No existe la carpeta {folder}")
        return [], []
    for filename in os.listdir(folder):
        if any(filename.lower().endswith(ext) for ext in IMAGE_EXTS):
            path = os.path.join(folder, filename)
            flag = cv2.IMREAD_UNCHANGED if load_with_alpha else cv2.IMREAD_COLOR
            img = cv2.imread(path, flag)
            if img is not None:
                if load_with_alpha and img.shape[2] != 4: continue
                images.append(img)
                filenames.append(os.path.splitext(filename)[0])
    return images, filenames

def calculate_iou_overlap(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    if inter_area == 0: return 0
    return inter_area / min(w1*h1, w2*h2)

def overlay_transparent(bg, overlay, x, y):
    bg_h, bg_w, _ = bg.shape
    ol_h, ol_w, _ = overlay.shape
    if x >= bg_w or y >= bg_h: return bg
    w, h = min(ol_w, bg_w - x), min(ol_h, bg_h - y)
    if w <= 0 or h <= 0: return bg
    
    crop_ol = overlay[:h, :w]
    crop_bg = bg[y:y+h, x:x+w]
    alpha = crop_ol[:, :, 3] / 255.0
    alpha_inv = 1.0 - alpha
    
    bg_part = (alpha_inv[..., None] * crop_bg).astype(np.uint8)
    ol_part = (alpha[..., None] * crop_ol[:, :, :3]).astype(np.uint8)
    bg[y:y+h, x:x+w] = cv2.add(bg_part, ol_part)
    return bg

def convert_to_yolo_bbox(bbox, bg_w, bg_h):
    x, y, w, h = bbox
    return (x + w/2)/bg_w, (y + h/2)/bg_h, w/bg_w, h/bg_h

def load_stats(card_names):
    stats = {name: 0 for name in card_names}
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, 'r') as f:
                loaded = json.load(f)
                for k, v in loaded.items():
                    if k in stats: stats[k] = v
        except: pass
    return stats

def save_stats(stats):
    with open(STATS_FILE, 'w') as f: json.dump(stats, f, indent=4)

# ==========================================
# 🚀 MAIN
# ==========================================

def main():
    random.seed()
    
    temp_imgs_dir = os.path.join(OUTPUT_BASE_DIR, "temp_images")
    temp_lbls_dir = os.path.join(OUTPUT_BASE_DIR, "temp_labels")
    ensure_dir(temp_imgs_dir); ensure_dir(temp_lbls_dir)

    print("--- Cargando recursos ---")
    card_imgs, card_names = load_images_from_folder(CARDS_DIR, load_with_alpha=True)
    bg_imgs, _ = load_images_from_folder(BACKGROUNDS_DIR)
    
    if not card_imgs or not bg_imgs:
        print("❌ Error: Faltan imágenes.")
        return

    stats = load_stats(card_names)
    class_map = {name: i for i, name in enumerate(sorted(card_names))}

    print(f"--- Generando {TOTAL_IMAGES_TO_GENERATE} imágenes (Modo Zoom Lejano) ---")
    print(f"--- Rango de escala: {CAMERA_ZOOM_RANGE[0]*100}% a {CAMERA_ZOOM_RANGE[1]*100}% ---")
    
    generated_files = []

    for i in tqdm(range(TOTAL_IMAGES_TO_GENERATE)):
        bg = random.choice(bg_imgs).copy()
        # Estandarizar fondo a 1024px de ancho para consistencia
        if bg.shape[1] != 1024:
            s = 1024 / bg.shape[1]
            bg = cv2.resize(bg, (1024, int(bg.shape[0] * s)))
        bg_h, bg_w, _ = bg.shape

        # 1. DECIDIR ZOOM GLOBAL
        current_zoom = random.uniform(CAMERA_ZOOM_RANGE[0], CAMERA_ZOOM_RANGE[1])

        # 2. DECIDIR CANTIDAD DE CARTAS SEGÚN EL ZOOM
        # Si las cartas son diminutas (lejos), ponemos muchas para llenar la mesa.
        # Si son gigantes (cerca), ponemos pocas.
        if current_zoom < 0.10:   # < 10% (Muy lejos)
            num_cards = random.randint(5, 12)
        elif current_zoom < 0.20: # < 20% (Medio lejos)
            num_cards = random.randint(3, 7)
        else:                     # > 20% (Cerca)
            num_cards = random.randint(1, 3)

        labels_txt = []
        placed_boxes = []

        # Pesos para balanceo de dataset
        weights = [1.0 / ((stats[n] + 1) ** 3) for n in card_names]
        sum_w = sum(weights)
        weights = [w/sum_w for w in weights]

        for _ in range(num_cards):
            idx = random.choices(range(len(card_imgs)), weights=weights, k=1)[0]
            card_img = card_imgs[idx]
            card_name = card_names[idx]

            # Variación orgánica
            organic_var = random.uniform(0.90, 1.10)
            
            # Calcular escala
            if card_name in ESCALAS_ESPECIALES:
                final_scale = current_zoom * MINI_CARD_FACTOR * organic_var
            else:
                final_scale = current_zoom * organic_var

            # Calcular tamaño en píxeles
            target_w = int(bg_w * final_scale)
            
            # 🛑 SEGURIDAD: Evitar que sea invisible
            if target_w < 16: target_w = 16
            
            ratio = card_img.shape[0] / card_img.shape[1]
            target_h = int(target_w * ratio)
            
            resized_card = cv2.resize(card_img, (target_w, target_h), interpolation=cv2.INTER_AREA)

            # Intentar colocar (Anti-Solapamiento)
            for _ in range(30):
                max_x, max_y = bg_w - target_w, bg_h - target_h
                if max_x <= 0 or max_y <= 0: break
                
                px, py = random.randint(0, max_x), random.randint(0, max_y)
                new_box = [px, py, target_w, target_h]
                
                collision = False
                for existing in placed_boxes:
                    if calculate_iou_overlap(new_box, existing) > MAX_OVERLAP_ALLOWED:
                        collision = True; break
                
                if not collision:
                    bg = overlay_transparent(bg, resized_card, px, py)
                    placed_boxes.append(new_box)
                    stats[card_name] += 1
                    yb = convert_to_yolo_bbox((px, py, target_w, target_h), bg_w, bg_h)
                    labels_txt.append(f"{class_map[card_name]} {' '.join(f'{x:.6f}' for x in yb)}")
                    break
        
        if labels_txt:
            bname = f"tiny_v6_{i:05d}"
            cv2.imwrite(os.path.join(temp_imgs_dir, bname + ".jpg"), bg, [cv2.IMWRITE_JPEG_QUALITY, 85])
            with open(os.path.join(temp_lbls_dir, bname + ".txt"), "w") as f:
                f.write("\n".join(labels_txt))
            generated_files.append(bname)

    # 3. Guardar y Mover
    save_stats(stats)
    final_dirs = {
        'train_img': os.path.join(OUTPUT_BASE_DIR, 'images/train'), 'val_img': os.path.join(OUTPUT_BASE_DIR, 'images/val'),
        'train_lbl': os.path.join(OUTPUT_BASE_DIR, 'labels/train'), 'val_lbl': os.path.join(OUTPUT_BASE_DIR, 'labels/val')
    }
    for d in final_dirs.values(): ensure_dir(d)

    random.shuffle(generated_files)
    split = int(len(generated_files) * TRAIN_RATIO)
    
    for idx, f in enumerate(generated_files):
        subset = 'train' if idx < split else 'val'
        shutil.move(os.path.join(temp_imgs_dir, f+".jpg"), os.path.join(final_dirs[f'{subset}_img'], f+".jpg"))
        shutil.move(os.path.join(temp_lbls_dir, f+".txt"), os.path.join(final_dirs[f'{subset}_lbl'], f+".txt"))
    
    shutil.rmtree(temp_imgs_dir); shutil.rmtree(temp_lbls_dir)
    
    with open(os.path.join(OUTPUT_BASE_DIR, "data.yaml"), "w") as f:
        f.write(f"train: ../images/train\nval: ../images/val\nnc: {len(class_map)}\nnames: {sorted(list(class_map.keys()))}")

    print("✅ Dataset generado. Ahora incluye cartas DIMINUTAS.")

if __name__ == "__main__":
    main()