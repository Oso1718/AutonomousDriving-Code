# 4 Existe una imagen y un lidar con cada registro del dataset?

from pathlib import Path
import pandas as pd

ROOT_DIR = Path.cwd()
DATASET_DIR = ROOT_DIR / "robot"
CSV_PATH = DATASET_DIR / "global_clean.csv"

IMAGES_DIR = DATASET_DIR / "imagenes"
LIDAR_DIR = DATASET_DIR / "lidar"

data = pd.read_csv(CSV_PATH)

missing_rgb = []
missing_lidar = []

for idx, row in data.iterrows():
    rgb_path = DATASET_DIR / row["image_path"]
    lidar_path = DATASET_DIR / row["lidar_path"]

    if not rgb_path.exists():
        missing_rgb.append(idx)

    if not lidar_path.exists():
        missing_lidar.append(idx)

print("====== VALIDACIÓN DATASET ======")
print(f"Total filas: {len(data)}")
print(f"RGB faltantes: {len(missing_rgb)}")
print(f"LiDAR faltantes: {len(missing_lidar)}")

if missing_rgb or missing_lidar:
    print("Dataset NO está completamente íntegro")
else:
    print("Dataset íntegro y listo para entrenamiento")
