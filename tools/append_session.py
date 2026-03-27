# 1.5 Este script permite agregar nuevas sesiones de datos a un dataset existente.

from pathlib import Path
import pandas as pd
import shutil
import sys

# =========================
# USO
# =========================
# python append_session.py path/a/nueva_sesion

SESSION_DIR = Path(sys.argv[1])

ROOT_DIR = Path(__file__).resolve().parents[1]
#ROOT_DIR = Path.cwd()

OUTPUT_DIR = ROOT_DIR / "robot"
IMAGES_OUT = OUTPUT_DIR / "imagenes"
LIDAR_OUT = OUTPUT_DIR / "lidar"
GLOBAL_CSV = OUTPUT_DIR / "global.csv"

# =========================
# VALIDACIONES
# =========================

if not SESSION_DIR.exists():
    raise RuntimeError(" La carpeta de sesión no existe")

if not GLOBAL_CSV.exists():
    raise RuntimeError(" global.csv no existe")

# =========================
# INDEXAR IMÁGENES DE LA SESIÓN
# =========================

image_index = {}
lidar_index = {}

for img in SESSION_DIR.rglob("*.jpg"):
    name = img.name
    if name.startswith("image_"):
        image_index[name] = img
    elif name.startswith("lidar_image_"):
        lidar_index[name] = img

# =========================
# LEER CSV DE LA SESIÓN
# =========================

csv_files = list(SESSION_DIR.rglob("data_log_*.csv"))
if not csv_files:
    raise RuntimeError(" No se encontró CSV en la sesión")

#new_data = pd.read_csv(csv_files[0])
new_data = pd.read_csv(
    csv_files[0],
    engine="python",
    on_bad_lines="skip"
)

new_data = new_data.dropna(
    subset=["image_filename", "lidar_image_filename"]
) #Eliminamos filas con datos faltantes

new_data = new_data.reset_index(drop=True) #Reseteamos el indice

print(f"Filas leídas de la sesión: {len(new_data)}")
# =========================
# CREAR RUTAS RELATIVAS
# =========================

def name_only(p):
    if pd.isna(p):
        return None
    return Path(p).name


new_data["image_path"] = new_data["image_filename"].apply(
    lambda x: f"imagenes/{name_only(x)}"
)

new_data["lidar_path"] = new_data["lidar_image_filename"].apply(
    lambda x: f"lidar/{name_only(x)}"
)

# =========================
# COPIAR IMÁGENES
# =========================

for _, row in new_data.iterrows():
    # RGB
    img_name = name_only(row["image_filename"])
    src = image_index.get(img_name)
    dst = IMAGES_OUT / img_name

    if src and not dst.exists():
        shutil.copy2(src, dst)

    # LiDAR
    lidar_name = name_only(row["lidar_image_filename"])
    src = lidar_index.get(lidar_name)
    dst = LIDAR_OUT / lidar_name

    if src and not dst.exists():
        shutil.copy2(src, dst)

# =========================
# APPEND AL CSV GLOBAL
# =========================

global_data = pd.read_csv(GLOBAL_CSV)
final_data = pd.concat([global_data, new_data], ignore_index=True)

final_data.to_csv(GLOBAL_CSV, index=False)

print(" Sesión agregada correctamente")
print(f" Total filas: {len(final_data)}")
