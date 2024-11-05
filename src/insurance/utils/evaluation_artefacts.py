import mlflow
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, roc_auc_score
import os

def plot_confusion_matrix(model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> None:
    """Plot confusion matrix and log it as an artifact in MLflow."""
    disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.title(f"{model_name} - Confusion Matrix")
    plt.savefig(f"{model_name}_confusion_matrix.png")
    mlflow.log_artifact(f"{model_name}_confusion_matrix.png")
    plt.close()  # Close the plot to free memory
    os.remove(f"{model_name}_confusion_matrix.png")  # Clean up

def plot_roc_auc(model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> None:
    """Plot ROC-AUC curve and log it as an artifact in MLflow."""
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title(f"{model_name} - ROC-AUC Curve")
    plt.savefig(f"{model_name}_roc_auc.png")
    mlflow.log_artifact(f"{model_name}_roc_auc.png")
    plt.close()
    os.remove(f"{model_name}_roc_auc.png")

def plot_feature_importance(model, feature_names: list, model_name: str) -> None:
    """Plot feature importance for tree-based models and log it as an artifact in MLflow."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        sorted_indices = importances.argsort()
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(sorted_indices)), importances[sorted_indices], align='center')
        plt.yticks(range(len(sorted_indices)), [feature_names[i] for i in sorted_indices])
        plt.title(f"{model_name} - Feature Importance")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.savefig(f"{model_name}_feature_importance.png")
        mlflow.log_artifact(f"{model_name}_feature_importance.png")
        plt.close()
        os.remove(f"{model_name}_feature_importance.png")

# Example usage
def log_model_diagnostics(model, X_test: pd.DataFrame, y_test: pd.Series, feature_names: list, model_name: str) -> None:
    """Log confusion matrix, ROC-AUC, and feature importance for a model."""
    plot_confusion_matrix(model, X_test, y_test, model_name)
    plot_roc_auc(model, X_test, y_test, model_name)
    plot_feature_importance(model, feature_names, model_name)
