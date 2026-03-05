'''1 Este script construye la ruta y permite obtener todos los datos en un solo CSV 
    obtenidos de la carpeta data_logs para crear un dataset unificado
    llamado global.csv'''

from pathlib import Path
import pandas as pd
import shutil

# =========================
# CONFIGURACIÓN
# =========================

ROOT_DIR = Path.cwd()
DATA_LOGS_DIR = ROOT_DIR / "data_logs"

OUTPUT_DIR = ROOT_DIR / "robot"
IMAGES_OUT = OUTPUT_DIR / "imagenes"
LIDAR_OUT = OUTPUT_DIR / "lidar"

GLOBAL_CSV_PATH = OUTPUT_DIR / "global.csv"

IMAGES_OUT.mkdir(parents=True, exist_ok=True)
LIDAR_OUT.mkdir(parents=True, exist_ok=True)

# =========================
# 1. INDEXAR TODAS LAS IMÁGENES
# =========================

print("Indexando imágenes en disco...")

image_index = {}
lidar_index = {}

for img_path in DATA_LOGS_DIR.rglob("*.jpg"):
    name = img_path.name

    if name.startswith("image_"):
        image_index[name] = img_path
    elif name.startswith("lidar_image_"):
        lidar_index[name] = img_path

print(f"RGB encontradas: {len(image_index)}")
print(f"LiDAR encontradas: {len(lidar_index)}")

# =========================
# 2. LEER Y UNIR TODOS LOS CSV
# =========================

csv_files = list(DATA_LOGS_DIR.rglob("data_log_*.csv"))

if not csv_files:
    raise RuntimeError("No se encontraron CSVs.")

dfs = []
for csv_path in csv_files:
    df = pd.read_csv(csv_path)
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

print(f"Dataset unificado: {len(data)} filas")

# =========================
# 3. CREAR RUTAS RELATIVAS NUEVAS
# =========================

def extract_name(path_str):
    return Path(path_str).name

data["image_path"] = data["image_filename"].apply(
    lambda x: f"imagenes/{extract_name(x)}"
)

data["lidar_path"] = data["lidar_image_filename"].apply(
    lambda x: f"lidar/{extract_name(x)}"
)

# =========================
# 4. COPIAR IMÁGENES USANDO EL ÍNDICE
# =========================

missing_rgb = 0
missing_lidar = 0

for _, row in data.iterrows():
    # ---------- RGB ----------
    img_name = extract_name(row["image_filename"])
    dst_img = IMAGES_OUT / img_name

    src_img = image_index.get(img_name)

    if src_img is None:
        missing_rgb += 1
    else:
        if not dst_img.exists():
            shutil.copy2(src_img, dst_img)

    # ---------- LiDAR ----------
    lidar_name = extract_name(row["lidar_image_filename"])
    dst_lidar = LIDAR_OUT / lidar_name

    src_lidar = lidar_index.get(lidar_name)

    if src_lidar is None:
        missing_lidar += 1
    else:
        if not dst_lidar.exists():
            shutil.copy2(src_lidar, dst_lidar)

print(f"RGB faltantes: {missing_rgb}")
print(f"LiDAR faltantes: {missing_lidar}")

# =========================
# 5. GUARDAR CSV GLOBAL (SIN BORRAR NADA)
# =========================

OUTPUT_DIR.mkdir(exist_ok=True)
data.to_csv(GLOBAL_CSV_PATH, index=False)

print("Dataset final creado correctamente")
print(f"CSV: {GLOBAL_CSV_PATH}")
print(f"RGB: {IMAGES_OUT}")
print(f"LiDAR: {LIDAR_OUT}")
