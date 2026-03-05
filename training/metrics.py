import pandas as pd

def save_metrics(history, mode, output_dir):
    df = pd.DataFrame(history.history)
    df["epoch"] = df.index + 1
    df["mode"] = mode

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"metrics_{mode}.csv"
    df.to_csv(csv_path, index=False)

    print(f"Métricas guardadas en: {csv_path}")
