# codigo preprocessing/preprocess_rgb.py para convertir imagenes a RGB, redimensionar y normalizar

from pathlib import Path
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm

# =========================
# CONFIGURACION DEFINIMOS NUESTRO WORKSPACE Y LAS RUTAS A USAR
# =========================

ROOT_DIR = Path(__file__).resolve().parents[1]
#ROOT_DIR = Path.cwd() # directorio actual
CSV_PATH = ROOT_DIR / "robot" / "global_clean.csv"

RAW_IMAGES_DIR = ROOT_DIR / "robot" / "imagenes"
OUTPUT_DIR = ROOT_DIR / "robot" / "processed" / "rgb"

# print(RAW_IMAGES_DIR)

IMAGE_SIZE = (128, 128) # Definimos el tamaño para las imagenes

# Crear carpeta destino
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# CARGAR CSV
# =========================

df = pd.read_csv(CSV_PATH) # Leemos el CSV usamos el que ya se ha limpiado

print("Preprocesamiento RGB ")
print(f"Total imágenes en CSV: {len(df)}")

# =========================
# PROCESAMIENTO
# =========================

processed = 0
skipped = 0
missing = 0

for img_path in tqdm(df["image_path"], desc="Procesando imágenes"):
    img_name = Path(img_path).name  # Obtenemos el nombre de la imagen
    src_path = RAW_IMAGES_DIR / img_name
    dst_path = OUTPUT_DIR / img_name

    # Si ya existe entonces no reprocesar
    if dst_path.exists():
        skipped += 1
        continue

    # Si no existe la imagen original, contar como faltante
    if not src_path.exists():
        missing += 1
        continue

    # Leer imagen
    img = cv2.imread(str(src_path))
    if img is None:
        missing += 1
        continue

    # De BGR a RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize a 128x128
    img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

    # Normalizamos imagen rango [0,1]
    img = img.astype(np.float32) / 255.0

    # Tranformamos a uint8 para guardar
    img = (img * 255).astype(np.uint8)

    # Guardar
    cv2.imwrite(str(dst_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    processed += 1

# =========================
# RESUMEN
# =========================

print("\n====== RESUMEN ======")
print(f"Procesadas: {processed}")
print(f"Omitidas (ya existían anteriormente): {skipped}")
print(f"Faltantes/errores: {missing}")
print("\n \nPreprocesamiento RGB terminado\n")
