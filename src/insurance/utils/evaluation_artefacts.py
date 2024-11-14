import mlflow
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from insurance import logging


class ModelDiagnosticsLogger:
    def __init__(self, pipeline: Pipeline, X_test: pd.DataFrame,
                 y_test: pd.Series,
                 model_name: str,
                 artefact_path: str=None,
                 mlflow_tracking: bool = False
                 ):
        self.pipeline = pipeline
        self.model = pipeline.named_steps['model']
        self.X_test = X_test
        self.y_test = y_test
        self.model_name = model_name
        self.feature_names = self.get_feature_names()
        self.artefact_path = artefact_path if artefact_path else os.path.join(mlflow.get_artifact_uri(), "artefacts")
        self.mlflow_tracking = mlflow_tracking

    def get_feature_names(self):
        """Extract feature names from a pipeline that may contain ColumnTransformer and PCA."""
        column_transformer = None
        pca_step_name = None
        pca_n_components = None

        # Identify ColumnTransformer and PCA in the pipeline
        for name, step in self.pipeline.named_steps.items():
            if isinstance(step, ColumnTransformer):
                column_transformer = step
            elif isinstance(step, PCA):
                pca_step_name = name
                pca_n_components = step.n_components_

        # Extract feature names from ColumnTransformer
        if column_transformer:
            feature_names = []
            for name, transformer, columns in column_transformer.transformers_:
                if transformer != 'drop':
                    if hasattr(transformer, 'get_feature_names_out'):
                        feature_names.extend(transformer.get_feature_names_out(columns))
                    else:
                        feature_names.extend(columns)
        else:
            # Fallback to input features if no ColumnTransformer is present
            feature_names = self.X_test.columns.tolist()

        # If PCA exists, rename features as PC components
        if pca_step_name:
            feature_names = [f"PC{i+1}" for i in range(pca_n_components)]

        return feature_names

    def plot_confusion_matrix(self):
        """Plot and log confusion matrix as an MLflow artifact."""
        try:
            ConfusionMatrixDisplay.from_estimator(self.pipeline, self.X_test, self.y_test)
            plt.title(f"{self.model_name} - Confusion Matrix")
            artifact_path = os.path.join(self.artefact_path, f"{self.model_name}_confusion_matrix.png")
            plt.savefig(artifact_path, bbox_inches='tight')
            if self.mlflow_tracking:
                mlflow.log_artifact(artifact_path)
            plt.close()
            #os.remove(artifact_path)
        except Exception as e:
            logging.info(f"Error occurred while plot_confusion_matrix: {str(e)}")


    def display_feature_importance(self, n_top=10):
        """
        This function takes in a dictionary of models, the dataset X, y, and the feature names.
        It fits each model, extracts feature importances (if available),
        and plots the top n features.

        Parameters:
        models (dict): A dictionary containing model names and their respective model objects.
        X (np.ndarray): Feature dataset.
        y (pd.Series): Target variable.
        feature_names (list): List of feature names after transformations.
        n_top (int): Number of top features to display. Default is 10.

        Returns:
        None
        """

        print(f"Feature ranking for model: {self.model_name}")
        try:
            # Check if the model has the attribute `feature_importances_`
            if hasattr(self.model, 'feature_importances_'):
                # Get feature importances
                importance_scores = self.model.feature_importances_

            else:
                #print(f"{self.model_name} does not support feature importances.")
                importance_scores = self.model.coef_[0]

            # Create a DataFrame from feature names and importances
            data = {'Feature': self.feature_names, 'Score': importance_scores}
            df = pd.DataFrame(data)


            # Take the absolute value of the score
            df['Abs_Score'] = np.abs(df['Score'])

            df_sorted = df.sort_values(by="Abs_Score", ascending=False)
            if n_top:
                # Sort by absolute value of score in descending order (top 10)
                df_sorted = df_sorted.head(n_top)

            # Define a color palette based on score values (positive = green, negative = red)
            colors = ["green" if score > 0 else "red" for score in df_sorted["Score"]]
            plt.figure(figsize=(12, 8))
            # Create the bar chart with Seaborn
            sns.barplot(x="Feature", y="Score", hue="Feature", legend=False, data=df_sorted, palette=colors)

            # Customize the plot for better visual appeal
            plt.xlabel("Feature")
            plt.ylabel("Feature Importance Score")
            plt.title(f"Feature Importance in {self.model_name} Classification")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            artifact_path = os.path.join(self.artefact_path, f"{self.model_name}_feature_importance_score.png")
            plt.savefig(artifact_path, bbox_inches='tight')
            if self.mlflow_tracking:
                mlflow.log_artifact(artifact_path)
            plt.close()
            #os.remove(artifact_path)

        except Exception as e:
            logging.info(f"Error occurred while extracting feature importances: {str(e)}")

    def plot_roc_auc_curve_seaborn(self):
        """
        Plots the ROC curve and calculates the AUC score for a given model using Seaborn.

        Args:
            model: The trained model to evaluate.
            X_test: The test data features.
            y_test: The test data labels.
            title: Title for the plot (default: "ROC Curve").
        """
        try:
            y_pred_proba = self.pipeline.predict_proba(self.X_test)[:, 1]
            figsize=(8, 6)

            # Calculate ROC curve and AUC score
            fpr, tpr, thr = roc_curve(self.y_test, y_pred_proba)
            auc = roc_auc_score(self.y_test, y_pred_proba)

            # Create the plot
            plt.figure(figsize=figsize)

            sns.lineplot(x=fpr, y=tpr, label=f'ROC curve (AUC = {auc:.3f})', color='violet', linewidth=2)

            # Plot baseline performance line
            plt.plot([0, 1], [0, 1], linestyle='--', label='Baseline performance', color='gray')

            # Set axis labels and title
            plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=14)
            plt.ylabel('True Positive Rate (Sensitivity)', fontsize=14)
            plt.title(f"{self.model_name} - ROC-AUC Curve", fontsize=16)
            plt.legend(loc=4)

            artifact_path = os.path.join(self.artefact_path, f"{self.model_name}_roc_auc_curve.png")
            plt.savefig(artifact_path, bbox_inches='tight')
            if self.mlflow_tracking:
                mlflow.log_artifact(artifact_path)
            plt.close()
            #os.remove(artifact_path)
        except Exception as e:
            logging.info(f"Error occurred while plotting ROC curve: {str(e)}")


    def log_model_diagnostics(self):
        """Log all diagnostics to MLflow, including confusion matrix, ROC-AUC, and feature importance."""
        self.plot_confusion_matrix()
        self.display_feature_importance()
        self.plot_roc_auc_curve_seaborn()
        logging.info(f"All diagnostics for {self.model_name} logged to MLflow.")


# Usage example
# evaluator = ModelDiagnosticsLogger(trained_pipeline, X_test, y_test, "model")
# evaluator.log_model_diagnostics()
