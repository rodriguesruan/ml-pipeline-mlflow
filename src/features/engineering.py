import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
import pickle

PROCESSED_DIR = Path("data/processed")


def load_processed_data():
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze()
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


def create_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["acid_ratio"] = X["fixed_acidity"] / (X["volatile_acidity"] + 1e-6)
    X["sulfur_ratio"] = X["free_sulfur_dioxide"] / (X["total_sulfur_dioxide"] + 1e-6)
    X["alcohol_density"] = X["alcohol"] / (X["density"] + 1e-6)
    return X


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.columns.tolist()
    preprocessor = ColumnTransformer(transformers=[
        ("scaler", StandardScaler(), numeric_features)
    ])
    return preprocessor


def encode_labels(y_train, y_test):
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    return y_train_enc, y_test_enc, le


def run_feature_engineering():
    X_train, X_test, y_train, y_test = load_processed_data()

    X_train = create_features(X_train)
    X_test = create_features(X_test)

    preprocessor = build_preprocessor(X_train)
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    y_train_enc, y_test_enc, le = encode_labels(y_train, y_test)

    with open(PROCESSED_DIR / "preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)
    with open(PROCESSED_DIR / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    print(f"Features criadas: {X_train.shape[1]} colunas")
    print(f"Classes: {le.classes_}")
    return X_train_scaled, X_test_scaled, y_train_enc, y_test_enc, le


if __name__ == "__main__":
    run_feature_engineering()
    print("Feature engineering concluído!")