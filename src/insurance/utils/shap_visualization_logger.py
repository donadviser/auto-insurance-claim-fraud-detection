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
                 X_train: pd.DataFrame,
                 X_test: pd.DataFrame,
                 model_name:str="model",
                 artefact_dir_path: str="",
                 mlflow_tracking = False):
        # Store model, data, and name
        self.pipeline = pipeline
        self.X_train = X_train
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

            # Transform Xtrain and X_test using the preprocessor and retrieve transformed feature names
            self.X_train_transformed = preprocessor.transform(self.X_train)
            self.X_test_transformed = preprocessor.transform(self.X_test)

            if isinstance(preprocessor.named_steps['column_transformer'], ColumnTransformer):
                self.feature_names = preprocessor.named_steps['column_transformer'].get_feature_names_out()
            else:
                self.feature_names = self.X_train.columns  # Use original names if no transformation

            # Confirm shape match with SHAP
            if self.X_train_transformed.shape[1] != len(self.feature_names):
                raise ValueError(f"Mismatch: {self.X_train_transformed.shape[1]} transformed features vs {len(self.feature_names)} feature names.")

            # Convert transformed data to DataFrame for SHAP with feature names
            self.X_train_transformed_df = pd.DataFrame(self.X_train_transformed, columns=self.feature_names)
            self.X_test_transformed_df = pd.DataFrame(self.X_test_transformed, columns=self.feature_names)

            # Extract the final model from the pipeline
            self.model = self.pipeline.named_steps['model']

        except Exception as e:
            raise CustomException(e, sys)

    def log_summary_plot(self):
        """Log summary plot."""
        logging.info("Log summary plot")
        try:
            plt.figure(figsize=(10, 6))

            if self.model_name in ["RandomForestClassifier", "DecisionTreeClassifier"]:
                # Create a Tree SHAP explainer and calculate SHAP values
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(self.X_train_transformed_df)
                shap.summary_plot(shap_values[:,:,0], self.X_test_transformed_df, show=False)
            elif self.model_name in ["LGBMClassifier", "CatBoostClassifier", "GradientBoostingClassifier"]:
                explainer = shap.Explainer(self.model)
                shap_values = explainer.shap_values(self.X_train_transformed_df)
                shap.summary_plot(shap_values, self.X_test_transformed_df, show=False) # work for LGBMClassifier, CatBoostClassifier, GradientBoostingClassifier
            elif self.model_name in ['KNearestNeighbors']:
                self.explainer = shap.KernelExplainer(self.model.predict_proba, self.X_train_transformed_df)
                self.shap_values = self.explainer.shap_values(self.X_test_transformed_df.iloc[0,:])
                shap.summary_plot(self.shap_values[:,0], self.X_test_transformed_df.iloc[0,:],show=False)

            plt.title(f"{self.model_name} - SHAP Summary Plot")
            file_path = os.path.join(self.artefact_dir_path, f"{self.model_name}_shap_summary_plot.png")
            logging.info(f"file_path: {file_path}")
            plt.savefig(file_path, bbox_inches='tight')
            if self.mlflow_tracking:
                mlflow.log_artifact(file_path)
                os.remove(file_path)
            plt.close()

            logging.info(f"{self.model_name}_shap_summary_plot saved")
        except Exception as e:
            logging.info(f"Error while creating shap summary plot: {str(e)}")

    def log_waterfall_plot(self, index=0):
        try:
            if self.model_name in ["XGBClassifier"]:
                plt.figure()
                explainer = shap.Explainer(self.model, self.X_train_transformed_df)
                self.shap_values = explainer(self.X_test_transformed_df)
                shap.plots.waterfall(self.shap_values[index], show=False) #worked for XGBClassifier, GradientBoostingClassifier
                #shap.plots.waterfall(shap_values[0,:,0]) #worked for XGBClassifier, GradientBoostingClassifier
                file_path = os.path.join(self.artefact_dir_path, f"{self.model_name}_waterfall_plot.png")
                plt.savefig(file_path, bbox_inches='tight')
                if self.mlflow_tracking:
                    mlflow.log_artifact(file_path)
                    os.remove(file_path)
                plt.close()
                logging.info(f"{self.model_name}_waterfall_plot saved")
        except Exception as e:
            logging.info(f"Error while creating shap summary plot: {str(e)}")


    def log_beeswarm_plot(self, index=0):
        plt.figure()
        if self.model_name == 'LogisticRegression':
            self.explainer = shap.Explainer(self.model, self.X_train_transformed_df)
            self.shap_values = self.explainer(self.X_test_transformed_df)
            shap.plots.beeswarm(self.shap_values, show=False)
            #shap.plots.force(self.shap_values)
            plt.title(f"{self.model_name} - SHAP Beeswarm Plot")
            file_path = os.path.join(self.artefact_dir_path, f"{self.model_name}_beeswarm_plot.png")
            plt.savefig(file_path, bbox_inches='tight')
            if self.mlflow_tracking:
                mlflow.log_artifact(file_path)
                os.remove(file_path)
            plt.close()
            logging.info(f"{self.model_name}_shap_beeswarm_plot saved")

    def log_all(self):
        """Log all SHAP visualizations."""
        logging.info("Log all SHAP visualizations")
        self.log_summary_plot()
        self.log_waterfall_plot()
        self.log_beeswarm_plot()
        """self.log_force_plot()

        self.log_bar_plot()
        """
