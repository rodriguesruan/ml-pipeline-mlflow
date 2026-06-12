import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.features.engineering import run_feature_engineering

PROCESSED_DIR = Path("data/processed")
mlflow.set_tracking_uri("sqlite:///mlflow.db")
EXPERIMENT_NAME = "wine-quality-classification"

MODELS = {
    "random_forest": RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42
    ),
    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, random_state=42
    ),
    "logistic_regression": LogisticRegression(
        max_iter=1000, random_state=42
    ),
}


def train_model(name, model, X_train, X_test, y_train, y_test, le):
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=name):
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_weighted")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

        mlflow.log_param("model_name", name)
        mlflow.log_param("n_features", X_train.shape[1])

        if hasattr(model, "n_estimators"):
            mlflow.log_param("n_estimators", model.n_estimators)
        if hasattr(model, "max_depth"):
            mlflow.log_param("max_depth", model.max_depth)
        if hasattr(model, "learning_rate"):
            mlflow.log_param("learning_rate", model.learning_rate)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_weighted", f1)
        mlflow.log_metric("cv_f1_mean", cv_mean)
        mlflow.log_metric("cv_f1_std", cv_std)

        mlflow.sklearn.log_model(model, artifact_path="model")

        report = classification_report(y_test, y_pred, target_names=le.classes_)
        report_path = PROCESSED_DIR / f"report_{name}.txt"
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(str(report_path))

        print(f"\n{name}")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  F1:        {f1:.4f}")
        print(f"  CV F1:     {cv_mean:.4f} ± {cv_std:.4f}")

    return accuracy, f1


if __name__ == "__main__":
    print("Rodando feature engineering...")
    X_train, X_test, y_train, y_test, le = run_feature_engineering()

    print("\nIniciando experimentos no MLflow...")
    results = {}
    for name, model in MODELS.items():
        acc, f1 = train_model(name, model, X_train, X_test, y_train, y_test, le)
        results[name] = {"accuracy": acc, "f1": f1}

    best = max(results, key=lambda x: results[x]["f1"])
    print(f"\nMelhor modelo: {best} (F1: {results[best]['f1']:.4f})")
    print("\nAcesse http://127.0.0.1:5000 para ver os experimentos no MLflow!")