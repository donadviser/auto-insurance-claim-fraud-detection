import copy
import matplotlib.pyplot as plt
import shap
import mlflow
import pandas as pd
import os
import sys
from sklearn.pipeline   import Pipeline
from sklearn.compose    import ColumnTransformer
from insurance import CustomException
from insurance import logging

class SHAPLogger:
    def __init__(self, pipeline: Pipeline,
                 X_test: pd.DataFrame,
                 model_name:str="model",
                 artefact_dir_path: str="",
                 mlflow_tracking = False
                 ):
        # Store model, data, and name
        self.pipeline = pipeline
        self.X_test = X_test
        self.model_name = model_name
        self.artefact_dir_path = artefact_dir_path
        self.mlflow_tracking = mlflow_tracking

        self.tree_based_classifiers = [
            'RandomForestClassifier',
            'GradientBoostingClassifier',
            'ExtraTreesClassifier',
            'DecisionTreeClassifier',
            'HistGradientBoostingClassifier',
            'AdaBoostClassifier',
            "CatBoostClassifier",
        ]

        self.non_sklearn_classifiers = [
            "XGBClassifier",                # xgboost
            "LGBMClassifier",               # lightgbm
            "LogisticRegression",
        ]


        # Prepare the preprocessor and transform data
        self._prepare_data()

    def _prepare_data(self):
        shap.initjs()
        try:
            # Deep copy the pipeline to separate the preprocessor from the model
            preprocessor = copy.deepcopy(self.pipeline)
            preprocessor.steps = [(name, step) for name, step in preprocessor.steps if name != 'model']

            # Transform X_test using the preprocessor and retrieve transformed feature names
            self.X_test_transformed = preprocessor.transform(self.X_test)

            if isinstance(preprocessor.named_steps['column_transformer'], ColumnTransformer):
                self.feature_names = preprocessor.named_steps['column_transformer'].get_feature_names_out()
                logging.info(f'1:self.feature_names: {self.feature_names}')
            else:
                self.feature_names = self.X_test.columns  # Use original names if no transformation
                logging.info(f'2:self.feature_names: {self.feature_names}')
            # Confirm shape match with SHAP
            if self.X_test_transformed.shape[1] != len(self.feature_names):
                raise ValueError(f"Mismatch: {self.X_test_transformed.shape[1]} transformed features vs {len(self.feature_names)} feature names.")

            # Convert transformed data to DataFrame for SHAP with feature names
            self.X_test_transformed_df = pd.DataFrame(self.X_test_transformed, columns=self.feature_names)

            # Extract the final model from the pipeline
            self.model = self.pipeline.named_steps['model']

        except Exception as e:
            raise CustomException(e, sys)

    def log_summary_plot(self):
        try:
            plt.figure(figsize=(10, 6))
            print(f"Inside self.model_name: {self.model_name}")
            print(f"Length of shap values[0]: {len(self.shap_values[:,0])}")
            print(f"Length of X_test_transformed_df: {self.X_test_transformed_df.shape}")
            if self.model_name == 'KNearestNeighbors':
                shap.summary_plot(self.shap_values[:,0], self.X_test_transformed_df.iloc[0,:],show=False)
            else:
                shap.summary_plot(self.shap_values, self.X_test_transformed_df, show=False)
            plt.title(f"{self.model_name} - SHAP Summary Plot")
            file_path = os.path.join(self.artefact_dir_path, f"{self.model_name}_shap_summary_plot.png")
            plt.savefig(file_path, bbox_inches='tight')
            if self.mlflow_tracking:
                mlflow.log_artifact(file_path)
                os.remove(file_path)
            plt.close()

        except Exception as e:
            raise CustomException(e, sys)

    def log_force_plot(self, index=0):
        try:
            plt.figure(10,4)
            shap_values_single = self.shap_values.values[index]
            expected_value = self.explainer.expected_value
            shap.force_plot(expected_value, shap_values_single, self.X_test_transformed_df.iloc[index], matplotlib=True, show=False)
            file_path = os.path.join(self.artefact_dir_path, f"{self.model_name}_shap_force_plot.png")
            plt.savefig(file_path, bbox_inches='tight')
            if self.mlflow_tracking:
                mlflow.log_artifact(file_path)
                os.remove(file_path)
            plt.close()
        except Exception as e:
            raise CustomException(e, sys)

    def log_waterfall_plot(self, index=0):
        try:
            plt.figure()
            shap.plots.waterfall(self.shap_values[index], show=False)
            file_path = os.path.join(self.artefact_dir_path, f"{self.model_name}_waterfall_plot.png")
            plt.savefig(file_path, bbox_inches='tight')
            if self.mlflow_tracking:
                mlflow.log_artifact(file_path)
                os.remove(file_path)
            plt.close()
        except Exception as e:
            raise CustomException(e, sys)

    def log_bar_plot(self):
        try:
            plt.figure()
            shap.plots.bar(self.shap_values, show=False)
            file_path = os.path.join(self.artefact_dir_path, f"{self.model_name}_shap_mean_plot.png")
            plt.savefig(file_path, bbox_inches='tight')
            if self.mlflow_tracking:
                mlflow.log_artifact(file_path)
                os.remove(file_path)
            plt.close()
        except Exception as e:
            raise CustomException(e, sys)

    def log_beeswarm_plot(self):
        try:
            plt.figure()
            shap.plots.beeswarm(self.shap_values, show=False)
            file_path = os.path.join(self.artefact_dir_path, f"{self.model_name}_shap_beeswarm_plot.png")
            plt.savefig(file_path, bbox_inches='tight')
            if self.mlflow_tracking:
                mlflow.log_artifact(file_path)
                os.remove(file_path)
            plt.close()
        except Exception as e:
            raise CustomException(e, sys)

    def log_all(self):
        """Log all SHAP visualizations."""
        self.log_summary_plot()
        self.log_force_plot()
        self.log_waterfall_plot()
        self.log_bar_plot()
        self.log_beeswarm_plot()
