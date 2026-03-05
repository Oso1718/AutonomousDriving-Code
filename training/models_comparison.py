import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
metrics_dir = ROOT_DIR / "results" / "metrics"
dfs = []

for csv in metrics_dir.glob("metrics_*.csv"):
    dfs.append(pd.read_csv(csv))

df = pd.concat(dfs)

# MAE validation comparison
plt.figure()
for mode in df["mode"].unique():
    subset = df[df["mode"] == mode]
    plt.plot(subset["epoch"], subset["val_mae"], label=mode)

plt.xlabel("Epoch")
plt.ylabel("Validation MAE")
plt.title("Validation MAE Comparison")
plt.legend()
plt.show()

# Loss validation comparison
plt.figure()
for mode in df["mode"].unique():
    subset = df[df["mode"] == mode]
    plt.plot(subset["epoch"], subset["val_loss"], label=mode)
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("Validation Loss Comparison")
plt.legend()
plt.show()

# Guardar el DataFrame consolidado
output_file = metrics_dir / "metrics_consolidadas.csv"
df.to_csv(output_file, index=False)
print(f"Métricas consolidadas guardadas en: {output_file}")

