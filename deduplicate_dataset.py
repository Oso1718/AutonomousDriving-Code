# 2 deduplicate_dataset.py Este script elimina filas duplicadas en el dataset basado en las rutas de imagen y lidar.

from pathlib import Path
import pandas as pd

ROOT = Path.cwd()
DATASET = ROOT / "robot"
CSV_PATH = DATASET / "global.csv"
OUT_PATH = DATASET / "global_dedup.csv"

data = pd.read_csv(CSV_PATH)

def sample_key(row):
    return (
        Path(row["image_path"]).name,
        Path(row["lidar_path"]).name
    )

seen = set()
rows = []

for _, row in data.iterrows():
    key = sample_key(row)
    if key not in seen:
        seen.add(key)
        rows.append(row)

clean = pd.DataFrame(rows).reset_index(drop=True)

clean.to_csv(OUT_PATH, index=False)

print("\n======DEDUPLICACIÓN COMPLETA======\n")
print(f"Filas originales : {len(data)}")
print(f"Filas finales    : {len(clean)}")
print(f"Duplicados quitados: {len(data) - len(clean)}")
print(f"Archivo generado : {OUT_PATH}")


# quick_check.py (temporal)

df = pd.read_csv("robot/global_dedup.csv")

keys = df.apply(
    lambda r: (
        Path(r["image_path"]).name,
        Path(r["lidar_path"]).name
    ),
    axis=1
)

print("\n\nDuplicados:", keys.duplicated().sum())
