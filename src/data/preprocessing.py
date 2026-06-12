import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def load_data() -> pd.DataFrame:
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases"
        "/wine-quality/winequality-red.csv"
    )
    df = pd.read_csv(url, sep=";")
    df.to_csv(RAW_DIR / "winequality-red.csv", index=False)
    print(f"Dataset carregado: {df.shape[0]} linhas, {df.shape[1]} colunas")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    df = df.copy()
    df["quality_label"] = pd.cut(
        df["quality"],
        bins=[0, 4, 6, 10],
        labels=["low", "medium", "high"]
    )
    print(f"Após limpeza: {df.shape[0]} linhas")
    return df


def split_data(df: pd.DataFrame):
    X = df.drop(columns=["quality", "quality_label"])
    y = df["quality_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)

    print(f"Treino: {X_train.shape[0]} | Teste: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)
    split_data(df)
    print("Preprocessing concluído!")