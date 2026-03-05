# 9 Visual Transformer Implementation
# Importamos las librerias
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

from sklearn.model_selection import train_test_split

# Mis librerias
from training.metrics import save_metrics
from training.graphic_plots import plot_history

# =========================
# ARGUMENTOS PARA EL PROGRAMA
# # =========================
parser = argparse.ArgumentParser(
    description="Entrenamiento de un modelo Visual Transformer"
)

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

MODEL_DIR = ROOT_DIR / "models" / "transformer" # Directorio para guardar los modelos entrenados
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
# Visual Transformer Components
# =========================
class PatchEmbedding(layers.Layer):
    def __init__(self, patch_size, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.projection = layers.Dense(embed_dim)

    def call(self, images):
        batch_size = tf.shape(images)[0]

        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID"
        )

        patch_dim = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dim])

        return self.projection(patches)


class PositionalEmbedding(layers.Layer):
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        self.pos_emb = self.add_weight(
            shape=(1, num_patches, embed_dim),
            initializer="random_normal"
        )

    def call(self, x):
        return x + self.pos_emb

class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()

        self.att = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim
        )

        self.ffn = models.Sequential([
            layers.Dense(ff_dim, activation="gelu"),
            layers.Dense(embed_dim),
        ])

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

        self.drop1 = layers.Dropout(rate)
        self.drop2 = layers.Dropout(rate)

    def call(self, x, training=False):
        attn = self.att(x, x, training=training)
        x = self.norm1(x + self.drop1(attn, training=training))

        ffn = self.ffn(x)
        return self.norm2(x + self.drop2(ffn, training=training))



def build_vit_model(input_shape, lr):

    patch_size = 16
    embed_dim = 128
    num_heads = 4
    ff_dim = 256
    num_layers = 4

    inputs = layers.Input(shape=input_shape)

    num_patches = (input_shape[0] // patch_size) ** 2

    x = PatchEmbedding(patch_size, embed_dim)(inputs)
    x = PositionalEmbedding(num_patches, embed_dim)(x)

    for _ in range(num_layers):
        x = TransformerEncoder(embed_dim, num_heads, ff_dim)(x)

    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(2)(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=optimizers.Adam(lr),
        loss="mse",
        metrics=["mae"]
    )

    return model



# =========================
# PRINCIPAL (FLUJO DE TRABAJO)
# =========================

if __name__ == "__main__":
    
    print(f"Entrenando Visual Transformer con preprocesamiento: {args.mode}")

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

    model = build_vit_model(
        input_shape=(128, 128, 3),
        lr=args.lr
    )

    model.summary()

    history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=args.epochs,
    batch_size=args.batch_size
    )

    # =========================
    # GUARDAR MODELO Y METRICAS
    # =========================
    save_metrics(history, "vit_"+ args.mode, ROOT_DIR / "results" / "metrics") # Guardamos las metricas de entrenamiento
    plot_history(history, "vit_"+ args.mode, ROOT_DIR / "results" / "plots") # Graficamos las metricas de entrenamiento

    export_path = MODEL_DIR / f"vit_{args.mode}_savedmodel"
    model.export(export_path)

    print(f"Modelo exportado en: {export_path}")
