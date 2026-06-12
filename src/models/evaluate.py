import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pickle
import numpy as np
from pathlib import Path

from src.features.engineering import run_feature_engineering

PROCESSED_DIR = Path("data/processed")
SCREENSHOTS_DIR = Path("screenshots")

mlflow.set_tracking_uri("sqlite:///mlflow.db")


def get_best_run(experiment_name: str, metric: str = "f1_weighted"):
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"]
    )
    return runs[0]


def plot_confusion_matrix(y_test, y_pred, labels, model_name: str):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix — {model_name}")
    plt.ylabel("Real")
    plt.xlabel("Previsto")
    path = SCREENSHOTS_DIR / f"confusion_matrix_{model_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Salvo: {path}")
    return path


def evaluate_best_model():
    X_train, X_test, y_train, y_test, le = run_feature_engineering()

    best_run = get_best_run("wine-quality-classification")
    model_name = best_run.data.params.get("model_name", "best_model")
    print(f"\nMelhor modelo: {model_name}")
    print(f"F1: {best_run.data.metrics['f1_weighted']:.4f}")
    print(f"Accuracy: {best_run.data.metrics['accuracy']:.4f}")

    model_uri = f"runs:/{best_run.info.run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)

    y_pred = model.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    plot_confusion_matrix(y_test, y_pred, le.classes_, model_name)

    with mlflow.start_run(run_id=best_run.info.run_id):
        cm_path = SCREENSHOTS_DIR / f"confusion_matrix_{model_name}.png"
        mlflow.log_artifact(str(cm_path))

    print("\nAvaliação concluída!")


if __name__ == "__main__":
    evaluate_best_model()