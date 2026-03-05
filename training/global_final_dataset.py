# training/create_global_final.py

from pathlib import Path
import pandas as pd

# =========================
# CONFIGURACIÓN
# =========================

ROOT_DIR = Path(__file__).resolve().parents[1]

CSV_INPUT = ROOT_DIR / "robot" / "global_clean.csv"
CSV_OUTPUT = ROOT_DIR / "robot" / "global_final.csv"

RGB_DIR = ROOT_DIR / "robot" / "processed" / "rgb"
SOBEL_DIR = ROOT_DIR / "robot" / "processed" / "sobel"
HSV_DIR = ROOT_DIR / "robot" / "processed" / "hsv"

# =========================
# CARGAR CSV
# =========================

df = pd.read_csv(CSV_INPUT)

print("====== CREANDO GLOBAL_FINAL.CSV ======")
print(f"Filas originales: {len(df)}")

# =========================
# INDEXAR IMÁGENES DISPONIBLES
# =========================

rgb_images = set(p.name for p in RGB_DIR.glob("*.jpg"))
sobel_images = set(p.name for p in SOBEL_DIR.glob("*.jpg"))
hsv_images = set(p.name for p in HSV_DIR.glob("*.jpg"))

valid_images = rgb_images & sobel_images & hsv_images

print(f"Imágenes válidas en las 3 modalidades: {len(valid_images)}")

# =========================
# FILTRAR CSV
# =========================

df["image_name"] = df["image_path"].apply(lambda p: Path(p).name)

df_final = df[df["image_name"].isin(valid_images)].copy()

df_final.drop(columns=["image_name"], inplace=True)

# =========================
# GUARDAR CSV FINAL
# =========================

df_final.to_csv(CSV_OUTPUT, index=False)

# =========================
# RESUMEN
# =========================

print("\n====== RESUMEN ======")
print(f"Filas finales: {len(df_final)}")
print(f"Filas eliminadas: {len(df) - len(df_final)}")
print(f" Archivo guardado en: {CSV_OUTPUT}")
print("El archivo --global_final.csv-- creado correctamente")
