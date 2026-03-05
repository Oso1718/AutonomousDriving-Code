# training/verify_processed_datasets.py

from pathlib import Path
import pandas as pd

# =========================
# CONFIGURACIÓN
# =========================

ROOT_DIR = Path(__file__).resolve().parents[1]
#ROOT_DIR = Path.cwd()

CSV_PATH = ROOT_DIR / "robot" / "global_clean.csv"

RGB_DIR = ROOT_DIR / "robot" / "processed" / "rgb"
SOBEL_DIR = ROOT_DIR / "robot" / "processed" / "sobel"
HSV_DIR = ROOT_DIR / "robot" / "processed" / "hsv"

# =========================
# CARGAR CSV
# =========================

df = pd.read_csv(CSV_PATH)

csv_images = set(df["image_path"].apply(lambda p: Path(p).name))

# =========================
# INDEXAR DIRECTORIOS
# =========================

rgb_images = set(p.name for p in RGB_DIR.glob("*.jpg"))
sobel_images = set(p.name for p in SOBEL_DIR.glob("*.jpg"))
hsv_images = set(p.name for p in HSV_DIR.glob("*.jpg"))

# =========================
# RESULTADOS
# =========================

print("\n====== VERIFICACIÓN DE DATASET ======\n")

print(f"Registros en CSV            : {len(csv_images)}")
print(f" Imágenes RGB procesadas    : {len(rgb_images)}")
print(f" Imágenes Sobel procesadas  : {len(sobel_images)}")
print(f" Imágenes HSV procesadas    : {len(hsv_images)}\n")

# =========================
# INTERSECCIÓN TOTAL
# =========================

common = csv_images & rgb_images & sobel_images & hsv_images
print(f" Completas en las 3 modalidades: {len(common)}\n")

# =========================
# FALTANTES
# =========================

missing_rgb = csv_images - rgb_images
missing_sobel = csv_images - sobel_images
missing_hsv = csv_images - hsv_images

print(f" Faltantes RGB   : {len(missing_rgb)}")
print(f" Faltantes Sobel : {len(missing_sobel)}")
print(f" Faltantes HSV   : {len(missing_hsv)}")

# =========================
# GUARDAR LISTAS (opcional)
# =========================

REPORT_DIR = ROOT_DIR / "results" / "metrics"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

pd.DataFrame({"missing_rgb": list(missing_rgb)}).to_csv(
    REPORT_DIR / "missing_rgb.csv", index=False
)
pd.DataFrame({"missing_sobel": list(missing_sobel)}).to_csv(
    REPORT_DIR / "missing_sobel.csv", index=False
)
pd.DataFrame({"missing_hsv": list(missing_hsv)}).to_csv(
    REPORT_DIR / "missing_hsv.csv", index=False
)

print("\n Reportes guardados en results/metrics/")
print("========================================\n")
