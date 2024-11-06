import copy
import matplotlib.pyplot as plt
import shap
import mlflow
import pandas as pd
import os
from sklearn.pipeline   import Pipeline
from sklearn.compose    import ColumnTransformer

class SHAPLogger:
    def __init__(self, pipeline: Pipeline, X_test: pd.DataFrame, model_name:str="model", artefact_dir_path: str=""):
        # Store model, data, and name
        self.pipeline = pipeline
        self.X_test = X_test
        self.model_name = model_name
        self.artefact_dir_path = artefact_dir_path
        
        self.tree_based_classifiers = [
            'RandomForestClassifier',
            'GradientBoostingClassifier',
            'ExtraTreesClassifier',
            'DecisionTreeClassifier',
            'HistGradientBoostingClassifier',
            'AdaBoostClassifier',
        ]
        
        self.non_sklearn_classifiers = [
            "XGBClassifier",                # xgboost
            "LGBMClassifier",               # lightgbm
            "CatBoostClassifier",           # catboost
            "LogisticRegression",
            'KNeighborsClassifier'
        ]
        
        
        # Prepare the preprocessor and transform data
        self._prepare_data()

    def _prepare_data(self):
        # Deep copy the pipeline to separate the preprocessor from the model
        preprocessor = copy.deepcopy(self.pipeline)
        preprocessor.steps = [(name, step) for name, step in preprocessor.steps if name != 'model']
        
        # Transform X_test using the preprocessor and retrieve transformed feature names
        self.X_test_transformed = preprocessor.transform(self.X_test)
        
        if isinstance(preprocessor.named_steps['column_transformer'], ColumnTransformer):
            self.feature_names = preprocessor.named_steps['column_transformer'].get_feature_names_out()
        else:
            self.feature_names = self.X_test.columns  # Use original names if no transformation
        
        # Confirm shape match with SHAP
        if self.X_test_transformed.shape[1] != len(self.feature_names):
            raise ValueError(f"Mismatch: {self.X_test_transformed.shape[1]} transformed features vs {len(self.feature_names)} feature names.")
        
        # Convert transformed data to DataFrame for SHAP with feature names
        self.X_test_transformed_df = pd.DataFrame(self.X_test_transformed, columns=self.feature_names)
        
        # Extract the final model from the pipeline
        self.model = self.pipeline.named_steps['model']
        
        # Set up SHAP explainer for Tree-based model
        if self.model_name  in self.non_sklearn_classifiers:
            explainer = shap.Explainer(self.model, self.X_test_transformed_df)
            self.shap_values = explainer(self.X_test_transformed_df)
        else:
            explainer = shap.TreeExplainer(self.model)
            self.shap_values = explainer(self.X_test_transformed_df)
            
            # Handle cases where shap_values have multiple outputs or extra dimensions
            if len(self.shap_values.values.shape) > 2:
                # Take the SHAP values for the first output, assuming binary classification
                self.shap_values = self.shap_values[:, :, 1]

    def log_summary_plot(self):
        plt.figure(figsize=(10, 6))
        shap.summary_plot(self.shap_values, self.X_test_transformed_df, show=False)
        plt.title("f{self.model_name} - SHAP Summary Plot")
        file_path = os.path.join(self.artefact_dir_path, f"{self.model_name}_shap_summary_plot.png")
        plt.savefig(file_path, bbox_inches='tight')
        mlflow.log_artifact(file_path)
        plt.close()

    def log_force_plot(self, index=0):
        plt.figure(10,4)
        shap_values_single = self.shap_values.values[index]
        expected_value = self.explainer.expected_value
        shap.force_plot(expected_value, shap_values_single, self.X_test_transformed_df.iloc[index], matplotlib=True, show=False)
        file_path = os.path.join(self.artefact_dir_path, f"{self.model_name}_shap_force_plot.png")
        plt.savefig(file_path, bbox_inches='tight')
        mlflow.log_artifact(file_path)
        plt.close()
    
    def log_waterfall_plot(self, index=0):
        plt.figure()
        shap.plots.waterfall(self.shap_values[index], show=False)
        file_path = os.path.join(self.artefact_dir_path, f"{self.model_name}_waterfall_plot.png")
        plt.savefig(file_path, bbox_inches='tight')
        mlflow.log_artifact(file_path)
        plt.close()

    def log_bar_plot(self):
        plt.figure()
        shap.plots.bar(self.shap_values, show=False)
        file_path = os.path.join(self.artefact_dir_path, f"{self.model_name}_shap_mean_plot.png")
        plt.savefig(file_path, bbox_inches='tight')
        mlflow.log_artifact(file_path)
        plt.close()
    
    def log_beeswarm_plot(self):
        plt.figure()
        shap.plots.beeswarm(self.shap_values, show=False)
        file_path = os.path.join(self.artefact_dir_path, f"{self.model_name}_shap_beeswarm_plot.png")
        plt.savefig(file_path, bbox_inches='tight')
        mlflow.log_artifact(file_path)
        plt.close()
    
    def log_all(self):
        """Log all SHAP visualizations."""
        self.log_summary_plot()
        self.log_force_plot()
        self.log_waterfall_plot()
        self.log_bar_plot()
        self.log_beeswarm_plot()
