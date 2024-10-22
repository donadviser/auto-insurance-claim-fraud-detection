
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder, StandardScaler, OrdinalEncoder, PowerTransformer, RobustScaler, MinMaxScaler,
    FunctionTransformer,MinMaxScaler)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline

from .custom_transformers import (
    DropRedundantColumns,
    CreateNewFeature,
    ReplaceClassTransformer,
)
from optuna.samplers import TPESampler
from sklearn.decomposition import PCA



from imblearn.over_sampling import (
    RandomOverSampler,
    SMOTE,
    ADASYN,
)
from imblearn.under_sampling import (
    RandomUnderSampler,
    NearMiss,
)
from imblearn.combine import (
    SMOTEENN,
    SMOTETomek
)



class HyperparameterTuner:
    """
    HyperparameterTuner to return hyperparameters for each classifier.
    """
    def get_params(self, trial: optuna.Trial, classifier_name: str):
        if classifier_name == "RandomForest":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 2, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            }
        elif classifier_name == "DecisionTree":
            return {
                "max_depth": trial.suggest_int("max_depth", 2, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            }
        elif classifier_name == "LGBM":
            return {
                "objective": "binary",
                "metric": "binary_logloss",
                "verbosity": -1,
                "boosting_type": "gbdt",
                "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 2, 256),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            }
        elif classifier_name == "XGBoost":
            return {
                "verbosity": 0,
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
                "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
                "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
                "subsample": trial.suggest_float("subsample", 0.2, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
            }
        elif classifier_name == "CatBoost":
            return {
                "objective": trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),
                "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
                "depth": trial.suggest_int("depth", 1, 12),
                "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
                "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),
            }
        elif classifier_name == "LogisticRegression":
            params = {
                "C": trial.suggest_float('C', 1e-10, 1000, log=True),
                "max_iter": trial.suggest_int('max_iter', 1, 1000, log=False),
                "l1_ratio": trial.suggest_float('l1_ratio', 0, 1, log=False),
                "solver": trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'sag', 'saga']),
            }
            # Adjust penalties based on solver
            if params['solver'] == 'lbfgs':
                params['penalty'] = trial.suggest_categorical('lbfgs', ['l2', None])
            elif params['solver'] == 'liblinear':
                params['penalty'] = trial.suggest_categorical('liblinear', ['l1', 'l2'])
            elif params['solver'] == 'sag':
                params['penalty'] = trial.suggest_categorical('sag', ['l2', None])
            else:
                params['penalty'] = trial.suggest_categorical('saga', ['elasticnet', 'l1', 'l2', None])
            return params
        else:
            raise ValueError(f"Invalid classifier name: {classifier_name}")
        

class ModelFactory:
    """
    A class to create model instances with additional parameters for specific classifiers.

    Attributes:
        model_name (str): The name of the model to be instantiated.
        best_params (dict): The best hyperparameters for the model.
    """

    def __init__(self, model_name: str, best_params: dict):
        """
        Initialize the ModelFactory with a model name and parameters.
        
        Args:
            model_name (str): The name of the model.
            best_params (dict): Hyperparameters for the model.
        """
        self.model_name = model_name
        self.best_params = best_params

    def get_model_instance(self):
        """
        Creates a model instance based on the model name with additional classifier-specific parameters.

        Returns:
            A model instance with the appropriate parameters.
        """
        # Dictionary of model classes
        model_dict = {
            "LGBM": LGBMClassifier,
            "XGBoost": XGBClassifier,
            "CatBoost": CatBoostClassifier,
            "RandomForest": RandomForestClassifier,
            "DecisionTree": DecisionTreeClassifier,
            "LogisticRegression": LogisticRegression,
            "SVC": SVC
        }

        # Check if the model exists in the model_dict
        if self.model_name not in model_dict:
            raise ValueError(f"Model {self.model_name} is not supported.")

        # Create a model instance with specific parameters
        if self.model_name == "LGBM":
            return model_dict[self.model_name](**self.best_params, random_state=42, verbose=-1)  # Add verbose for LGBM
        elif self.model_name == "RandomForest":
            return model_dict[self.model_name](**self.best_params, random_state=42, n_jobs=-1)  # Add n_jobs for RandomForest
        elif self.model_name == "SVC":
            return model_dict[self.model_name](**self.best_params, random_state=42, probability=True)  # Add probability for SVC
        elif self.model_name == "CatBoost":
            return model_dict[self.model_name](**self.best_params, random_state=42, verbose=0)  # Suppress CatBoost verbosity
        else:
            return model_dict[self.model_name](**self.best_params, random_state=42)  # Default for other models
        
        
class PipelineManager:
    """
    A class that handles both building and modifying pipelines dynamically.
    This class supports both scikit-learn's Pipeline and imbalanced-learn's Pipeline.

    It allows the construction of the initial pipeline and the insertion of steps 
    at any position within the pipeline.
    """

    def __init__(self, pipeline_type='ImbPipeline'):
        """
        Initialize the PipelineManager with a specified pipeline type.

        Args:
            pipeline_type (str): The type of pipeline to use ('ImbPipeline' or 'Pipeline').
        """
        if pipeline_type == 'ImbPipeline':
            self.pipeline = ImbPipeline(steps=[])
        elif pipeline_type == 'Pipeline':
            self.pipeline = Pipeline(steps=[])
        else:
            raise ValueError("Unsupported pipeline type. Choose 'ImbPipeline' or 'Pipeline'.")

    def add_step(self, step_name, step_object, position=None):
        """
        Add a transformation step to the pipeline.

        Args:
            step_name (str): Name of the step to add.
            step_object (object): The transformer or estimator object (e.g., scaler, classifier).
            position (int or None): Optional; the position to insert the step.
                                    If None, the step is appended at the end of the pipeline.
        """
        if position is None:
            self.pipeline.steps.append((step_name, step_object))
        else:
            self.pipeline.steps.insert(position, (step_name, step_object))

    def remove_step(self, step_name):
        """
        Remove a step from the pipeline by its name.

        Args:
            step_name (str): The name of the step to remove.
        """
        self.pipeline.steps = [(name, step) for name, step in self.pipeline.steps if name != step_name]

    def replace_step(self, step_name, new_step_object):
        """
        Replace an existing step in the pipeline with a new step.

        Args:
            step_name (str): The name of the step to replace.
            new_step_object (object): The new transformer or estimator object.
        """
        for i, (name, step) in enumerate(self.pipeline.steps):
            if name == step_name:
                self.pipeline.steps[i] = (step_name, new_step_object)
                break

    def get_pipeline(self):
        """
        Get the constructed or modified pipeline.

        Returns:
            Pipeline: The constructed or modified pipeline object.
        """
        return self.pipeline
    
    
class PreprocessingPipeline:
    """
    A class that encapsulates the preprocessing steps for feature engineering,
    imputation, scaling, encoding, and transformations. This can be inserted into
    the overall pipeline before the model fitting step.
    """
    def __init__(self, bins_hour, names_period, drop_columns, numerical_features,
                 onehot_features, ordinal_features, transform_features):
        """
        Initialize the PreprocessingPipeline with necessary parameters.

        Args:
            bins_hour: Parameters for creating new features from hourly bins.
            names_period: Period names for feature creation.
            drop_columns: Columns to be dropped from the dataset.
            numerical_features: List of numerical features for processing.
            onehot_features: List of categorical features for OneHot encoding.
            ordinal_features: List of ordinal features for Ordinal encoding.
            transform_features: Features that require power transformation.
        """
        self.bins_hour = bins_hour
        self.names_period = names_period
        self.drop_columns = drop_columns
        self.numerical_features = numerical_features
        self.onehot_features = onehot_features
        self.ordinal_features = ordinal_features
        self.transform_features = transform_features
        
        

    def build(self, step_name=None):
        """
        Build the preprocessing pipeline with feature creation, transformation, 
        imputation, scaling, and encoding steps.
        
        Returns:
            Transformer: The appropriate transformer for the given step.
        """
        
        if step_name == "create_new_features":
            return CreateNewFeature(bins_hour=self.bins_hour, names_period=self.names_period)
        
        if step_name == "replace_class":
            return ReplaceClassTransformer(target_value="?", replacement_value='unknown')
        
        if step_name == "drop_cols":
            return DropRedundantColumns(redundant_cols=self.drop_columns)
        
        if step_name == 'column_transformer':
            return ColumnTransformer(
                transformers=[
                    ('numerical', Pipeline([
                        ('imputer', SimpleImputer(strategy='mean')),  # Mean imputation for numerical features
                        #('scaler', StandardScaler())
                    ]), self.numerical_features),
                    
                    ('categorical', Pipeline([
                        ('imputer', SimpleImputer(strategy='most_frequent')),  # Mode imputation for categorical features
                        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
                    ]), self.onehot_features),
                    
                    ('ordinal', Pipeline([
                        ('imputer', SimpleImputer(strategy='most_frequent')),  # Mode imputation for ordinal features
                        ('ordinal', OrdinalEncoder())
                    ]), self.ordinal_features),
                    
                    ('power_transform', Pipeline([
                        ('imputer', SimpleImputer(strategy='mean')),  # Mean imputation for features needing power transformation
                        ('power_transformer', PowerTransformer(method='yeo-johnson'))
                    ]), self.transform_features),
                ],
                remainder='passthrough'
            )
            
            
            
class ResamplerSelector:
    """
    A class to select and return a resampling algorithm based on a given parameter or 
    from a trial suggestion if available.

    Attributes:
        trial (optuna.trial, optional): The trial object for hyperparameter optimization.
    """

    def __init__(self, trial=None, random_state=42):
        """
        Initialize the ResamplerSelector with an optional trial for hyperparameter optimization.

        Args:
            trial (optuna.trial, optional): An optional trial object for suggesting resampling strategies.
            random_state (int): Random seed for reproducibility. Default is 42.
        """
        self.trial = trial
        self.random_state = random_state

    def get_resampler(self, resampler=None):
        """
        Return the resampling algorithm based on the provided `resampler` parameter.
        If `resampler` is not given, it is suggested from the trial.

        Args:
            resampler (str, optional): The resampling method ('RandomOverSampler', 'ADASYN', etc.). 
                                       If not provided, it will be suggested from the trial (if available).

        Returns:
            resampler_obj (object): The resampling instance based on the selected method.
        """
        if resampler is None and self.trial:
            resampler = self.trial.suggest_categorical(
                'resampler', ['RandomOverSampler', 'ADASYN', 'RandomUnderSampler', 'NearMiss', 
                              'SMOTEENN', 'SMOTETomek']
            )

        if resampler == 'RandomOverSampler':
            return RandomOverSampler(random_state=self.random_state)
        elif resampler == 'ADASYN':
            return ADASYN(random_state=self.random_state)
        elif resampler == 'RandomUnderSampler':
            return RandomUnderSampler(random_state=self.random_state)
        elif resampler == 'NearMiss':
            return NearMiss()
        elif resampler == 'SMOTEENN':
            return SMOTEENN(random_state=self.random_state, sampling_strategy='minority' )
        elif resampler == 'SMOTETomek':
            return SMOTETomek(random_state=self.random_state)
        else:
            raise ValueError(f"Unknown resampler: {resampler}")
        
        
class ScalerSelector:
    """
    A class to select and return a scaling algorithm based on a given parameter or 
    from a trial suggestion if available.

    Attributes:
        trial (optuna.trial, optional): The trial object for hyperparameter optimization.
    """

    def __init__(self, trial=None):
        """
        Initialize the ScalerSelector with an optional trial for hyperparameter optimization.

        Args:
            trial (optuna.trial, optional): An optional trial object for suggesting resampling strategies.
        """
        self.trial = trial

    def get_scaler(self, scaler_name=None):
        """
        Return the scaling algorithm based on the provided `scaler_name` parameter.
        If `scaler_name` is not given, it is suggested from the trial.

        Args:
            scaler_name (str, optional): The scalring method ('MinMaxScaler', 'StandardScaler', etc.). 
                                       If not provided, it will be suggested from the trial (if available).

        Returns:
            rscaler_obj (object): The scaling instance based on the selected method.
        """ 
         
        # -- Instantiate scaler (skip scaler for CatBoost as it handles categorical features internally)
        if scaler_name is None and self.trial:
            scaler_name = self.trial.suggest_categorical("scaler", ['minmax', 'standard', 'robust'])
            
        if scaler_name == "minmax":
            return MinMaxScaler()
        elif scaler_name == "standard":
            return StandardScaler()
        elif scaler_name == "robust":
            return RobustScaler()
        else:
            raise ValueError(f"Unknown scaler: {scaler_name}")
        
        
class DimensionalityReductionSelector:
    """
    A class to select and return a dimensionality reduction algorithm based on a given parameter 
    or from a trial suggestion if available.

    Attributes:
        trial (optuna.trial, optional): The trial object for hyperparameter optimization.
    """

    def __init__(self, trial=None):
        """
        Initialize the DimensionalityReductionSelector with an optional trial for hyperparameter optimization.

        Args:
            trial (optuna.trial, optional): An optional trial object for suggesting dimensionality reduction strategies.
        """
        self.trial = trial

    def get_dimensionality_reduction(self, dim_red=None):
        """
        Return the dimensionality reduction algorithm based on the provided `dim_red` parameter.
        If `dim_red` is not given, it is suggested from the trial.

        Args:
            dim_red (str, optional): The dimensionality reduction method ('PCA' or None). If not provided,
                                     it will be suggested from the trial (if available).

        Returns:
            dimen_red_algorithm (object or str): PCA algorithm or 'passthrough'.
        """
        if dim_red is None and self.trial:
            dim_red = self.trial.suggest_categorical("dim_red", ["PCA", None])

        if dim_red == "PCA":
            if self.trial:
                pca_n_components = self.trial.suggest_int("pca_n_components", 2, 30)
            else:
                pca_n_components = 5  # Default value if trial is not provided
            dimen_red_algorithm = PCA(n_components=pca_n_components)
        else:
            dimen_red_algorithm = 'passthrough'

        return dimen_red_algorithm