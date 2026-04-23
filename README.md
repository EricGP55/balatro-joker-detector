# Balatro Comodines Scan

Este proyecto es un escáner y modelo de detección para los comodines (jokers) del juego Balatro. Utiliza YOLOv8 para el entrenamiento y detección de las cartas en la pantalla.

## Instalación

1. Clona el repositorio:
   ```bash
   git clone <URL_DEL_REPOSITORIO>
   cd Balatro_Comodines_Scan
   ```

2. Crea un entorno virtual e instala las dependencias:
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Uso

- `generar_dataset.py`: Script para generar el conjunto de datos de entrenamiento.
- `entrenar.py`: Script para entrenar el modelo YOLO.
- `escaner_pantalla.py`: Captura y analiza la pantalla para detectar los comodines.
- `probar_modelo.py`: Permite probar el modelo entrenado con imágenes.
- `descargar_comodines.py`: Script para descargar imágenes de los comodines.

## Estructura del Proyecto

- `Comodines/`: Carpeta con imágenes de comodines.
- `Fondos/`: Fondos para data augmentation o generación de dataset.
- `Entrenamiento_Cartas/`: Dataset generado para el entrenamiento de YOLO.
- `yolov8n.pt`: Pesos base de YOLOv8 nano.
