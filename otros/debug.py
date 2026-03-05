from pathlib import Path
import pandas as pd
import shutil

# =========================
# CONFIGURACIÓN
# =========================

ROOT_DIR = Path.cwd()
DATA_LOGS_DIR = ROOT_DIR / "data_logs"

print(f"Directorio raíz: {ROOT_DIR}")
print(f"Directorio de registros de datos: {DATA_LOGS_DIR}")