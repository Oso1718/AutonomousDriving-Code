# 6 preprocessing/preprocess_sobel.py

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
OUTPUT_DIR = ROOT_DIR / "robot" / "processed" / "sobel"

IMAGE_SIZE = (128, 128)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# CARGAR CSV
# =========================

df = pd.read_csv(CSV_PATH)

print("====== PREPROCESAMIENTO SOBEL ======")
print(f"Total imágenes en CSV: {len(df)}")

# =========================
# PROCESAMIENTO
# =========================

processed = 0
skipped = 0
missing = 0

for img_path in tqdm(df["image_path"], desc="Procesando imágenes Sobel"):
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
    # PREPROCESAMIENTO SOBEL
    # =========================

    # BGR → Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize
    gray = cv2.resize(gray, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

    # Sobel X y Y
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Magnitud
    sobel = np.sqrt(sobel_x**2 + sobel_y**2)

    # Normalizar a [0,255]
    sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX)
    sobel = sobel.astype(np.uint8)

    # Replicar a 3 canales
    sobel_3ch = cv2.merge([sobel, sobel, sobel])

    # Guardar
    cv2.imwrite(str(dst_path), sobel_3ch)
    processed += 1

# =========================
# RESUMEN
# =========================

print("\n====== RESUMEN ======")
print(f"Procesadas: {processed}")
print(f"Omitidas (ya existían): {skipped}")
print(f"Faltantes/errores: {missing}")
print("Preprocesamiento Sobel terminado")
