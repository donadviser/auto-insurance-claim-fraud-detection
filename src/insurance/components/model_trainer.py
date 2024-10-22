import os
import sys
import pandas as pd
import numpy as np

import optuna
from typing import Union
from optuna.samplers import TPESampler

from insurance import logging
from insurance import CustomException
from insurance.utils.common import save_bin, load_bin
from insurance.constants import MODEL_CONFIG_FILE
from insurance.entity import ModelTrainerConfig
from insurance.entity import (
    DataIngestionArtefacts,
    DataTransformationArtefacts,
    ModelTrainerArtefacts,
)

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder, StandardScaler, OrdinalEncoder, PowerTransformer, RobustScaler, MinMaxScaler,
    FunctionTransformer, MinMaxScaler)
from sklearn.model_selection import  RandomizedSearchCV, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

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

from insurance.utils.custom_transformers import (
    DropRedundantColumns,
    CreateNewFeature,
    LogTransforms,
    ReplaceValueTransformer
)


class CostModel:
    def __init__(
            self, preprocessing_object: object, trained_model_object: object):
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object


    def predict(self, X) -> float:
        """
        This method predicts the data

        Args:
            X (pd.DataFrame): The data to be predicted.

        Returns:
            float: The predicted data.
        """
        try:
            # Predict the data
            transformed_feature = self.preprocessing_object.transform(X)
            logging.info("Used the trained model to get predictions")
             
            return self.trained_model_object.predict(transformed_feature)
        except Exception as e:
            raise CustomException(e, sys)
        
    def __repr__(self) -> str:
        return f"{type(self.trained_model_object).__name__}()"
    
    def __str__(self) -> str:
        return f"{type(self.trained_model_object).__name__}()"

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
            # Basic hyperparameters
            params = {
                "solver": trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'sag', 'saga']),
                "max_iter": trial.suggest_int('max_iter', 10000, 50000),  # Increased max_iter to allow for better convergence
            }

            # Adjust penalties based on solver
            if params['solver'] == 'lbfgs':
                params['penalty'] = trial.suggest_categorical('penalty_lbfgs', ['l2', None])
            elif params['solver'] == 'liblinear':
                params['penalty'] = trial.suggest_categorical('penalty_liblinear', ['l1', 'l2'])
            elif params['solver'] == 'sag':
                params['penalty'] = trial.suggest_categorical('penalty_sag', ['l2', None])
            else:
                # For 'saga', which supports 'elasticnet'
                params['penalty'] = trial.suggest_categorical('penalty_saga', ['elasticnet', 'l1', 'l2', None])

            # Only suggest C and l1_ratio if penalty is not None
            if params['penalty'] is not None:
                params["C"] = trial.suggest_float('C', 1e-10, 1000, log=True)
            
            # Only suggest l1_ratio if penalty is 'elasticnet'
            if params['penalty'] == 'elasticnet':
                params['l1_ratio'] = trial.suggest_float('l1_ratio', 0, 1)

            return params
        elif classifier_name == "GradientBoosting":
            return {
                "learning_rate" : trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                "n_estimators" : trial.suggest_int('n_estimators', 100, 1000),
                "max_depth" : trial.suggest_int('max_depth', 3, 10),
                "min_samples_split" : trial.suggest_int('min_samples_split', 2, 20),
                "min_samples_leaf" : trial.suggest_int('min_samples_leaf', 1, 20),
                "max_features" : trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    
            }
        elif classifier_name == "KNeighbors":
            params = {
                "n_neighbors": trial.suggest_int('n_neighbors', 1, 50),
                "weights": trial.suggest_categorical('weights', ['uniform', 'distance']),
                "p": trial.suggest_int('p', 1, 2),  # 1: Manhattan, 2: Euclidean
                "leaf_size": trial.suggest_int('leaf_size', 10, 100),
                "metric": trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'minkowski', 'chebyshev'])
            }
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
            "SVC": SVC,
            "GradientBoosting": GradientBoostingClassifier,
            "KNeighbors": KNeighborsClassifier
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
        elif self.model_name == "KNeighbors":
            return model_dict[self.model_name](**self.best_params)
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
                 onehot_features, ordinal_features, transform_features, trial: optuna.Trial=None):
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
        
    def instantiate_numerical_simple_imputer(self, trial: optuna.Trial=None, strategy: str='mean', fill_value: int=-1) -> SimpleImputer:
        if strategy is None and trial:
            strategy = trial.suggest_categorical(
                'numerical_strategy', ['mean', 'median', 'most_frequent']
            )
        #print(f"instantiate_numerical_simple_imputer: strategy= {strategy}")
        return SimpleImputer(strategy=strategy, fill_value=fill_value)

    def instantiate_categorical_simple_imputer(self, trial: optuna.Trial=None, strategy: str='most_frequent', fill_value: str='missing') -> SimpleImputer:
        if strategy is None and trial:
            strategy = trial.suggest_categorical('categorical_strategy', ['most_frequent', 'constant'])
        return SimpleImputer(strategy=strategy, fill_value=fill_value)
    
    def instantiate_outliers(self, trial: optuna.Trial=None, strategy: str='power_transform') -> Union[PowerTransformer, LogTransforms, str]:
        if strategy is None and trial:
            strategy = trial.suggest_categorical(
                'outlier_strategy', ['power_transform', 'log_transform']
            )
        #print(f"instantiate_outliers: strategy= {strategy}")
        if strategy == 'power_transform':
            #print("instantiate_outliers: Entered PowerTransformer() ")
            return PowerTransformer(method='yeo-johnson')
        elif strategy == 'log_transform':
            #print("instantiate_outliers: Entered FunctionTransformer()")
            return FunctionTransformer(np.log1p, validate=False)
        else:
            #print("instantiate_outliers: Entered 'passthrough'")
            return "passthrough"
         
    def build(self, step_name=None, trial: optuna.Trial=None):
        """
        Build the preprocessing pipeline with feature creation, transformation, 
        imputation, scaling, and encoding steps.
        
        Returns:
            Transformer: The appropriate transformer for the given step.
        """
        
        if step_name == "create_new_features":
            return CreateNewFeature(bins_hour=self.bins_hour, names_period=self.names_period)
        
        if step_name == "replace_class":
            return ReplaceValueTransformer(old_value="?", new_value=np.nan)
        
        if step_name == "drop_cols":
            return DropRedundantColumns(redundant_cols=self.drop_columns)
        
        if step_name == 'column_transformer':
            return ColumnTransformer(
                transformers=[
                    ('numerical', Pipeline([
                        ('imputer', self.instantiate_numerical_simple_imputer(trial=trial, strategy='mean')),
                        #('scaler', StandardScaler())  # Add scaler if needed
                    ]), self.numerical_features),
                    
                    ('categorical', Pipeline([
                        ('imputer', self.instantiate_categorical_simple_imputer(trial=trial, strategy='most_frequent', fill_value='missing')),   
                        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))
                    ]), self.onehot_features),
                    
                    ('ordinal', Pipeline([
                        ('imputer', self.instantiate_categorical_simple_imputer(trial=trial, strategy='most_frequent', fill_value='missing')),
                        ('ordinal', OrdinalEncoder())
                    ]), self.ordinal_features),
                    
                    ('outlier_transform', Pipeline([
                        ('imputer', self.instantiate_numerical_simple_imputer(trial=trial)),
                        ('outlier_transformer', self.instantiate_outliers(trial=trial, strategy="power_transformer"))
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

      
             
class ModelTrainer:
    def __init__(
            self,
            data_ingestion_artefacts: DataIngestionArtefacts,
            data_transformation_artefact: DataTransformationArtefacts,
            model_trainer_config: ModelTrainerConfig,
    ):
        self.data_ingestion_artefacts = data_ingestion_artefacts
        self.data_transformation_artefact = data_transformation_artefact
        self.model_trainer_config = model_trainer_config
        
         # Get the model parameters from model config file
        self.model_config = self.model_trainer_config.UTILS.read_yaml_file(filename=MODEL_CONFIG_FILE)
        
        # Reading the Train and Test data from Data Ingestion Artefacts folder
        self.train_set = pd.read_csv(self.data_ingestion_artefacts.train_data_file_path)
        self.test_set = pd.read_csv(self.data_ingestion_artefacts.test_data_file_path)
        logging.info(f"Loaded train_set dataset from the path: {self.data_ingestion_artefacts.train_data_file_path}")
        logging.info(f"Loaded test_set dataset from the path: {self.data_ingestion_artefacts.test_data_file_path}")
        logging.info(f"Shape of train_set: {self.train_set.shape}")
        logging.info(f"Shape of test_set: {self.test_set.shape}")
        
        # Getting neccessary column names from config file
        self.numerical_features = self.model_trainer_config.SCHEMA_CONFIG['numerical_features']
        self.onehot_features = self.model_trainer_config.SCHEMA_CONFIG['onehot_features']
        self.ordinal_features = self.model_trainer_config.SCHEMA_CONFIG['ordinal_features']
        self.transform_features = self.model_trainer_config.SCHEMA_CONFIG['transform_features']
        logging.info("Obtained the NUMERICAL COLS, ONE HOT COLS, ORDINAL COLS and TRANFORMER COLS from schema config")
        
        # Data Cleaning and Feature Engineering for Training dataset
        # Getting the bins and namess from config file
        self.bins_hour = self.model_trainer_config.SCHEMA_CONFIG['incident_hour_time_bins']['bins_hour']
        self.names_period = self.model_trainer_config.SCHEMA_CONFIG['incident_hour_time_bins']['names_period']
        self.drop_columns = self.model_trainer_config.SCHEMA_CONFIG['drop_columns']
        
        self.yes_no_map = self.model_trainer_config.SCHEMA_CONFIG['yes_no_map']
        self.target_columns = self.model_trainer_config.SCHEMA_CONFIG['target_column']
        logging.info("Completed reading the schema config file")
        
        X_train, y_train = self.train_set.drop(columns=[self.target_columns]), self.train_set[self.target_columns]
        X_test, y_test = self.test_set.drop(columns=[self.target_columns]), self.test_set[self.target_columns]

        self.X_train = X_train.copy()
        # target label to 1 and 0
        self.y_train = y_train.replace(self.yes_no_map)  # Map target labels
        
        self.X_test = X_test.copy()
        # target label to 1 and 0
        self.y_test = y_test.replace(self.yes_no_map)  # Map target labels
        logging.info("Completed setting the Train and Test X and y")
        
    # Define objective function for Optuna
    def objective(self, trial: optuna.Trial, classifier_name: str, scoring='f1') -> float:
        """
        Objective function to optimize classifiers dynamically using Optuna.

        Args:
            trial (optuna.Trial): Optuna trial object for suggesting hyperparameters.
            classifier_name (str): Classifier to optimize.
            scoring (str): Scoring metric for cross-validation.

        Returns:
            float: The mean score from cross-validation.
        """
        
        # Get hyperparameters for the classifier from HyperparameterTuner
        hyperparameter_tuner = HyperparameterTuner()
        params = hyperparameter_tuner.get_params(trial, classifier_name)
        #print("hyperparameter parameters obtained from HyperparameterTuner class")
        
        # Got the Preprocessed Pipeline containting Data Cleaning and Column Transformation
        preprocessing_pipeline = PreprocessingPipeline(
            bins_hour=self.bins_hour,
            names_period=self.names_period,
            drop_columns=self.drop_columns,
            numerical_features=self.numerical_features,
            onehot_features=self.onehot_features,
            ordinal_features=self.ordinal_features,
            transform_features=self.transform_features
        )
        
        # Initialize the manager with the preferred pipeline type ('ImbPipeline' or 'Pipeline')
        pipeline_manager = PipelineManager(pipeline_type='ImbPipeline')
        
        # Add transformation steps: Option 1 = Does not work, see the error message    
        pipeline_manager.add_step('create_new_features', preprocessing_pipeline.build(step_name='create_new_features', trial=None), position=0)
        pipeline_manager.add_step('replace_class', preprocessing_pipeline.build(step_name='replace_class', trial=None), position=1)
        pipeline_manager.add_step('drop_cols', preprocessing_pipeline.build(step_name='drop_cols', trial=None), position=2)
        pipeline_manager.add_step('column_transformer', preprocessing_pipeline.build(step_name='column_transformer', trial=trial), position=3)
        
        # Add the resampler step based on the provided resample name or trial suggestion
        resample_selector = ResamplerSelector(trial=trial)    
        resampler_obj = resample_selector.get_resampler()
        pipeline_manager.add_step('resampler', resampler_obj, position=4)
        
        
        # Add the scaler step based on the provided resample name or trial suggestion
        scaler_selector = ScalerSelector(trial=trial)    
        scaler_obj = scaler_selector.get_scaler()
        pipeline_manager.add_step('scaler', scaler_obj, position=5)
        
        
        # Add the Dimensional Reduction step based on the provided parameter or trial suggestion
        #dim_red_selector = DimensionalityReductionSelector(trial=trial)
        #dim_red_obj = dim_red_selector.get_dimensionality_reduction(dim_red=None)
        #pipeline_manager.add_step('dim_reduction', dim_red_obj, position=6)

        # Create an instance of the ModelFactory class with best_model and best_params
        model_factory = ModelFactory(classifier_name, params)
        model_obj = model_factory.get_model_instance()
        pipeline_manager.add_step('model', model_obj, position=7)
        
        pipeline = pipeline_manager.get_pipeline()
        #print(f"pipeline: {pipeline.steps[2:4]}")

        # Cross-validation
        kfold = StratifiedKFold(n_splits=10)
        score = cross_val_score(pipeline, self.X_train, self.y_train, scoring=scoring, n_jobs=-1, cv=kfold, verbose=0, error_score='raise')
        result = score.mean()
        
        """pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        result = accuracy"""

        return result
    
    
    # Run the Optuna study
    def run_optimization(self, config_path: str, n_trials: int = 100, scoring: str = 'f1') -> None:
        """
        Run Optuna study for hyperparameter tuning and model selection.
        
        Args:
            config_path (str): Path to the YAML configuration file.
            n_trials (int): Number of trials for optimization. Defaults to 100.
            scoring (str): Scoring metric for optimization. Defaults to 'f1'.
        """
        
        best_model_score = -1
        best_model = None
        best_params = None
        results = []

        all_models = ["RandomForest", "DecisionTree", 
                    "XGBoost", "LGBM", "GradientBoosting", 
                    "LogisticRegression", "KNeighbors", "CatBoost"]
        
        #all_models = ['DecisionTree']
        
        for model_name in all_models:
            logging.info(f"\n\nOptimizing model: {model_name}")
            study = optuna.create_study(direction="maximize", sampler=TPESampler())
            study.optimize(lambda trial: self.objective(trial, model_name, scoring), n_trials=n_trials)
            
            best_trial_obj = study.best_trial
            
            current_score = best_trial_obj.value            
            
            if current_score and current_score > best_model_score:
                best_model_score = current_score
                best_model = model_name
                best_params = best_trial_obj.params
            
            results.append({
                "model": model_name,
                "params": best_trial_obj.params,
                "model_score_params": best_trial_obj.params,
                "model_score_trial_number": best_trial_obj.number,                
                "model_score_duration": best_trial_obj.duration,
                "model_score_status": best_trial_obj.state,
                "model_score_key": scoring,
                "model_score_value": best_trial_obj.value                
            })
            
        return best_model, best_params, current_score
        
        """
            best_trial = study.best_trial
            results.append({
                "model": model_name,
                "params": best_trial.params,
                "model_score_params": best_trial.params,
                "model_score_trial_number": best_trial.number,
                "model_score_datetime": best_trial.datetime_start,
                "model_score_duration": best_trial.duration,
                "model_score_status": best_trial.state,
                "model_score_key": scoring,
                "model_score_value": best_trial.value                
            })

            current_score = best_trial.value            
            
            if current_score and current_score > best_model_score:
                best_model_score = current_score
                best_model = model_name
                best_params = best_trial.params
                
        logging.info(f"Model: {model_name}, Current Score: {current_score} | Best Model: {best_model}, Best Score: {best_model_score}")

        # Display all results and the best model
        for result in results:
            logging.info(result)
        
        logging.info(f"Best model: {best_model}")
        logging.info(f"Best parameters: {best_params}")
        
        
        # Save the variables to a file
        save_bin(data=(best_model, best_params, best_model_score), path="best_model_and_params.pkl")
        

        # Use list comprehension to gather keys that start with "pipe"
        keys_to_remove = ["resampler", "scaler", "dim_red", "pca_n_components"]

        # Pop those keys from the dictionary
        for key in keys_to_remove:
            best_params.pop(key, None)

        # Resulting dictionary after removal
        logging.info(f'cleaned best params: {best_params}')
         
        return best_model, best_params, best_model_score"""
    
        


        
    # This method is used to initialise model training
    def initiate_model_trainer(self) -> ModelTrainerArtefacts:
        """
        Initiate model training.

        Returns:
            ModelTrainerArtefacts: The model training artefacts.
        """
        
        
        logging.info("Entered the initiate_model_trainer method of ModelTrainer class.")
        
        try:
            # Creating Model Trainer artefacts directory
            os.makedirs(self.model_trainer_config.MODEL_TRAINER_ARTEFACTS_DIR, exist_ok=True)
            logging.info(f"Created the model trainer artefacts directory for {os.path.basename(self.model_trainer_config.MODEL_TRAINER_ARTEFACTS_DIR)}")

            
            # Create and run the study
            best_model, best_params, best_model_score = self.run_optimization(config_path=[], n_trials=30, scoring='f1')
            logging.info(f"Completed the training process")
            logging.info(f"Best params: {best_model}")
            logging.info(f"Best params: {best_params}")
            logging.info(f"best_model_score: {best_model_score}")
             
            
            # Reading model config file for getting the best model score
            model_config = self.model_trainer_config.UTILS.read_yaml_file(
                filename=MODEL_CONFIG_FILE
            )
            base_model_score = float(model_config["base_model_score"])
            logging.info(f"Got the best model score from model config file: {base_model_score}")
            logging.info(f"The best model score from the model training: {best_model_score}")

            # Updating the best model score to model config file if the model score is greather than the base model score
            if best_model_score >= base_model_score:
                self.model_trainer_config.UTILS.update_model_score(best_model_score)
                logging.info("Updated the best model score to model config file")

                """# Loading cost model object with preprocessor and model
                cost_model = CostModel(preprocessing_obj, best_model)
                logging.info("Created the cost model object with preprocessor and model")
                
                trained_model_path = self.model_trainer_config.TRAINED_MODEL_FILE_PATH
                logging.info("Created best model file path")

                # Saving the cost model in model artefacts directory
                model_file_path = self.model_trainer_config.UTILS.save_object(
                    trained_model_path, cost_model
                )"""
                trained_model_path = self.model_trainer_config.TRAINED_MODEL_FILE_PATH
                logging.info(f"Created best model file path: {trained_model_path}")
                
                model_file_path = self.model_trainer_config.UTILS.save_object(
                    trained_model_path, best_params)
                logging.info("Saved the best model object path")
            else:
                logging.info("No best mode found: The best model score is less than the base model score")
                raise Exception("No best model found with score more than base score")
                 
            
            # Savind the Model trainer artefacts
            model_trainer_artefacts = ModelTrainerArtefacts(
                trained_model_file_path=model_file_path
            )
            logging.info("Created the model trainer artefacts")
            logging.info("Exited the initiate_model_trainer method of ModelTrainer class.")
            return model_trainer_artefacts
        except Exception as e:
            raise CustomException(e, sys)
