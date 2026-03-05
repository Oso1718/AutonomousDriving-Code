from pathlib import Path
import pandas as pd
import shutil

# =========================
# CONFIGURACIÓN
# =========================

# Establecemos los directorios raiz y donde se encuentran los datos
ROOT_DIR = Path.cwd() # Directorio raíz del proyecto
DATA_LOGS_DIR = ROOT_DIR / "data_logs" # Directorio donde están los logs de datos

# Donde guardaremos los datos
OUTPUT_DIR = ROOT_DIR / "robot" # Carpeta de datos
IMAGES_OUT = OUTPUT_DIR / "imagenes" # Imagenes 
LIDAR_OUT = OUTPUT_DIR / "lidar" # Lidar

GLOBAL_CSV_PATH = OUTPUT_DIR / "global.csv" #Archivo csv final donde se conglomeran todos los datos

# Crear carpetas destino
IMAGES_OUT.mkdir(parents=True, exist_ok=True)
LIDAR_OUT.mkdir(parents=True, exist_ok=True)

# =========================
# 1. ENCONTRAR TODOS LOS CSV
# =========================

csv_files = list(DATA_LOGS_DIR.rglob("data_log_*.csv"))

if not csv_files:
    raise RuntimeError("❌ No se encontraron archivos CSV.")

print(f"✔ Encontrados {len(csv_files)} CSVs")

# =========================
# 2. UNIR TODOS LOS CSV
# =========================

dfs = []

for csv_path in csv_files:
    df = pd.read_csv(csv_path)
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

print(f"Dataset unificado: {len(data)} filas")

# =========================
# 3. CREAR RUTAS RELATIVAS NUEVAS
# =========================

def extract_filename(path_str):
    return Path(path_str).name

data["image_path"] = data["image_filename"].apply(
    lambda x: f"imagenes/{extract_filename(x)}"
)

data["lidar_path"] = data["lidar_image_filename"].apply(
    lambda x: f"lidar/{extract_filename(x)}"
)

# =========================
# 4. COPIAR IMÁGENES
# =========================

missing_images = 0
missing_lidar = 0

for idx, row in data.iterrows():
    # Imagen RGB
    src_img = ROOT_DIR / row["image_filename"]
    dst_img = IMAGES_OUT / Path(row["image_path"]).name

    if src_img.exists():
        if not dst_img.exists():
            shutil.copy2(src_img, dst_img)
    else:
        missing_images += 1

    # Imagen LiDAR
    src_lidar = ROOT_DIR / row["lidar_image_filename"]
    dst_lidar = LIDAR_OUT / Path(row["lidar_path"]).name

    if src_lidar.exists():
        if not dst_lidar.exists():
            shutil.copy2(src_lidar, dst_lidar)
    else:
        missing_lidar += 1

print(f"⚠ Imágenes RGB faltantes: {missing_images}")
print(f"⚠ Imágenes LiDAR faltantes: {missing_lidar}")

# =========================
# 5. LIMPIAR Y GUARDAR CSV
# =========================

# Opcional: eliminar columnas viejas
# data = data.drop(columns=["image_filename", "lidar_image_filename"])

OUTPUT_DIR.mkdir(exist_ok=True)
data.to_csv(GLOBAL_CSV_PATH, index=False)

print("✅ Dataset final creado correctamente")
print(f"📄 CSV: {GLOBAL_CSV_PATH}")
print(f"🖼 RGB: {IMAGES_OUT}")
print(f"📡 LiDAR: {LIDAR_OUT}")
