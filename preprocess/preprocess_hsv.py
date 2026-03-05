# 7 preprocessing/preprocess_hsv.py

from pathlib import Path
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm

# =========================
# CONFIGURACIÓN
# =========================

ROOT_DIR = Path(__file__).resolve().parents[1]
#ROOT_DIR = Path.cwd()
CSV_PATH = ROOT_DIR / "robot" / "global_clean.csv"
RAW_IMAGES_DIR = ROOT_DIR / "robot" / "imagenes"
OUTPUT_DIR = ROOT_DIR / "robot" / "processed" / "hsv"

IMAGE_SIZE = (128, 128)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# CARGAR CSV
# =========================

df = pd.read_csv(CSV_PATH)

print("====== PREPROCESAMIENTO HSV ======")
print(f"Total imágenes en CSV: {len(df)}")

# =========================
# PROCESAMIENTO
# =========================

processed = 0
skipped = 0
missing = 0

for img_path in tqdm(df["image_path"], desc="Procesando imágenes HSV"):
    img_name = Path(img_path).name

    src_path = RAW_IMAGES_DIR / img_name
    dst_path = OUTPUT_DIR / img_name

    if dst_path.exists():
        skipped += 1
        continue

    if not src_path.exists():
        missing += 1
        continue

    img = cv2.imread(str(src_path))
    if img is None:
        missing += 1
        continue

    # =========================
    # PREPROCESAMIENTO HSV
    # =========================

    # BGR → RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize
    img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

    # RGB → HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Separar canales
    h, s, v = cv2.split(hsv)

    # Contrast Limited Adaptive Histogram Equalization CLAHE (visión artificial)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    v_clahe = clahe.apply(v)
    hsv_processed = cv2.merge([v_clahe, v_clahe, v_clahe])

    # Reconstruir 3 canales (H, S, 0)
    #hsv_processed = cv2.merge([h, s, np.zeros_like(h)])

    # Guardar (convertir a BGR para OpenCV)
    #hsv_bgr = cv2.cvtColor(hsv_processed, cv2.COLOR_HSV2BGR)
    cv2.imwrite(str(dst_path), hsv_processed)

    processed += 1

# =========================
# RESUMEN
# =========================

print("\n====== RESUMEN ======")
print(f"Procesadas: {processed}")
print(f"Omitidas (ya existían): {skipped}")
print(f"Faltantes/errores: {missing}")
print(" Preprocesamiento HSV terminado")
