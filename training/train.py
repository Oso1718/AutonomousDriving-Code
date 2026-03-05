# 8 Algortimo de entrenamiento training/train.py
# Importamos las librerias
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt


import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

from sklearn.model_selection import train_test_split

# Mis librerias
from training.metrics import save_metrics
from training.graphic_plots import plot_history


# =========================
# ARGUMENTOS PARA EL PROGRAMA
# =========================

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, required=True,
                    choices=["rgb", "sobel", "hsv"]) # Este argumento define que dataset de imagenes sera procesado
parser.add_argument("--epochs", type=int, default=20) # Numero de epocas de entrenamiento para el modelo
parser.add_argument("--batch_size", type=int, default=32) #Tamano del batch para el entrenamiento 32 por default
parser.add_argument("--lr", type=float, default=1e-3)

args = parser.parse_args()

# =========================
# PATHS A LOS DIRECTORIOS A USAR
# =========================

ROOT_DIR = Path(__file__).resolve().parents[1]

CSV_PATH = ROOT_DIR / "robot" / "global_final.csv" # Usamos el dataset final ya filtrado
IMAGE_DIR = ROOT_DIR / "robot" / "processed" / args.mode 

MODEL_DIR = ROOT_DIR / "models" / "cnn" # Directorio para guardar los modelos entrenados
MODEL_DIR.mkdir(parents=True, exist_ok=True)

PLOTS_DIR = ROOT_DIR / "results" / "plots" # Directorio para guardar las graficas de entrenamiento
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# CONFIG
# =========================

IMG_SIZE = (128, 128)
N_OUTPUTS = 2  # linear + angular velocity

# =========================
# DATA LOADER
# =========================

def load_image(path):
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32) / 255.0
    return img

# Construir dataset
def build_dataset(df):
    X = []
    y = []

    print("Cargando imágenes en memoria")
    for _, row in tqdm(df.iterrows(), total=len(df)):

        # Ruta de imagen procesada
        img_path = IMAGE_DIR / Path(row["image_path"]).name
        if not img_path.exists():
            continue

        # Cargar imagen
        img = load_image(img_path)
        X.append(img)

        # Extraer comando de velocidad
        cmd = eval(row["velocity_cmd"])  # [v_linear, v_angular]
        y.append([cmd[0], cmd[1]])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)




# =========================
# MODELO RED NEURONAL TIPO CNN
# =========================

def build_cnn_model(input_shape, lr):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, 3, strides=2, padding="same"),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.Conv2D(64, 3, strides=2, padding="same"),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.Conv2D(128, 3, strides=2, padding="same"),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.GlobalAveragePooling2D(),

        layers.Dense(64, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(N_OUTPUTS)
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=["mse",
                 "mae"]
    )

    return model

# =========================
# PRINCIPAL (FLUJO DE TRABAJO)
# =========================

def main():

    print(f"Entrenando CNN con preprocesamiento: {args.mode}")

    df = pd.read_csv(CSV_PATH)

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    X_train, y_train = build_dataset(train_df)
    X_val, y_val = build_dataset(val_df)

    print(f"Train samples: {len(X_train)}")
    print(f"Val samples:   {len(X_val)}")


    # =========================
    # VERIFICAR ALGUNAS IMAGENES CARGADAS
    # =========================
    plt.figure(figsize=(20, 20))
    sample_img_paths = list(IMAGE_DIR.glob("*.jpg"))[:5]

    for i, img_path in enumerate(sample_img_paths):
        img = load_image(img_path)
        plt.subplot(3, 3, i + 1)
        plt.imshow(img)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


    model = build_cnn_model(
        input_shape=(128, 128, 3),
        lr=args.lr
    )

    model.summary()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1
    )

    #=======================
    # PLOT DE GRAFICAS DE ENTRENAMIENTO
    #=======================
    
    # MSE Plot
    plt.figure()
    plt.plot(history.history["mse"], label="Train MSE")
    plt.plot(history.history["val_mse"], label="Validation MSE")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error (Promedio de Error Cuadratico)")
    plt.title("Evolución del MSE por época")
    plt.legend()
    plt.grid(True)
    plt.show()

    # MAE Plot
    plt.figure()
    plt.plot(history.history["mae"], label="Train MAE")
    plt.plot(history.history["val_mae"], label="Validation MAE")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Absolute Error")
    plt.title("Evolución del MAE por época")
    plt.legend()
    plt.grid(True)
    plt.show()


    # =========================
    # GUARDAR MODELO Y METRICAS
    # =========================
    save_metrics(history, args.mode, ROOT_DIR / "results" / "metrics") # Guardamos las metricas de entrenamiento
    plot_history(history, args.mode, ROOT_DIR / "results" / "plots") # Graficamos las metricas de entrenamiento

    #model_name = f"cnn_{args.mode}.keras"
    #model_path = MODEL_DIR / model_name
    #model.save(model_path)
    #print(f"Modelo guardado en: {model_path}")

    export_path = MODEL_DIR / f"cnn_{args.mode}_savedmodel"
    model.export(export_path)

    print(f"Modelo exportado en: {export_path}")


if __name__ == "__main__":
    main()
