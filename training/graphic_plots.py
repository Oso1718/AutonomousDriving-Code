import matplotlib.pyplot as plt

def plot_history(history, mode, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Loss
    plt.figure()
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title(f"Loss - {mode.upper()}")
    plt.legend()
    plt.savefig(output_dir / f"loss_{mode}.png")
    plt.close()

    # MAE
    plt.figure()
    plt.plot(history.history["mae"], label="train_mae")
    plt.plot(history.history["val_mae"], label="val_mae")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.title(f"MAE - {mode.upper()}")
    plt.legend()
    plt.savefig(output_dir / f"mae_{mode}.png")
    plt.close()

    print(f"Gráficas guardadas para {mode}")
