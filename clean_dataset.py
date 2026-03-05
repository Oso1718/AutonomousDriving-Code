# 3 clean_dataset.py para 
from pathlib import Path
import pandas as pd

ROOT = Path.cwd()
DATASET = ROOT / "robot"
CSV_PATH = DATASET / "global_dedup.csv"

data = pd.read_csv(CSV_PATH)

def exists(rel_path):
    return (DATASET / rel_path).exists()

mask = data["image_path"].apply(exists) & data["lidar_path"].apply(exists)

clean_data = data[mask].reset_index(drop=True)

clean_path = DATASET / "global_clean.csv"
clean_data.to_csv(clean_path, index=False)

print(" Dataset limpio creado")
print(f" {clean_path}")
print(f"Filas originales: {len(data)}")
print(f"Filas limpias: {len(clean_data)}")
