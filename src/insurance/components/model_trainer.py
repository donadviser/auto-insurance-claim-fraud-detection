import os
import sys
import pandas as pd
import numpy as np

import optuna
import joblib
from typing import Union, Dict, Tuple
from optuna.samplers import TPESampler

from insurance import logging
from insurance import CustomException
from insurance.constants import MODEL_CONFIG_FILE, MODEL_SAVE_FORMAT
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

from sklearn.metrics import  accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

from sklearn.decomposition import PCA
from imblearn.over_sampling import (
    RandomOverSampler,
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
    LogTransformer,
    OutlierDetector,
    ReplaceValueTransformer,
    OutlierHandler,
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
                "solver": trial.suggest_categorical('solver', ['newton-cholesky', 'lbfgs', 'liblinear', 'sag', 'saga']),
                "max_iter": trial.suggest_int('max_iter', 10000, 50000),  # Increased max_iter to allow for better convergence
            }

            # Suggest penalty from a unified set
            all_penalties = ['l1', 'l2', 'elasticnet', None]  # Unified penalties
            params['penalty'] = trial.suggest_categorical('penalty', all_penalties)

            # Only suggest C if penalty is not None
            if params['penalty'] is not None:
                params["C"] = trial.suggest_float('C', 1e-10, 1000, log=True)
            
            # Only suggest l1_ratio if penalty is 'elasticnet'
            if params['penalty'] == 'elasticnet':
                params['l1_ratio'] = trial.suggest_float('l1_ratio', 0, 1)

            # Prune invalid combinations:
            if (
                (params['solver'] == 'lbfgs' and params['penalty'] not in ['l2', None]) or
                (params['solver'] == 'liblinear' and params['penalty'] not in ['l1', 'l2']) or
                (params['solver'] == 'sag' and params['penalty'] not in ['l2', None]) or
                (params['solver'] == 'newton-cholesky' and params['penalty'] not in ['l2', None]) or
                (params['solver'] == 'saga' and params['penalty'] not in ['elasticnet', 'l1', 'l2', None])
            ):
                raise optuna.TrialPruned()  # Invalid combination of solver and penalty

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
        model_hyperparams (dict): The best hyperparameters for the model.
    """

    def __init__(self, model_name: str, model_hyperparams: dict):
        """
        Initialize the ModelFactory with a model name and parameters.
        
        Args:
            model_name (str): The name of the model.
            model_hyperparams (dict): Hyperparameters for the model.
        """
        self.model_name = model_name
        self.model_hyperparams = model_hyperparams

    def get_model_instance(self):
        """
        Creates a model instance based on the model name with additional classifier-specific parameters.

        Returns:
            A model instance with the appropriate parameters.
        """
         
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

         
        if self.model_name not in model_dict:
            raise ValueError(f"Model {self.model_name} is not supported.")

        # Create a model instance with specific parameters
        if self.model_name == "LGBM":
            return model_dict[self.model_name](**self.model_hyperparams, random_state=42, verbose=-1)   
        elif self.model_name == "RandomForest":
            return model_dict[self.model_name](**self.model_hyperparams, random_state=42, n_jobs=-1)  
        elif self.model_name == "SVC":
            return model_dict[self.model_name](**self.model_hyperparams, random_state=42, probability=True)   
        elif self.model_name == "CatBoost":
            return model_dict[self.model_name](**self.model_hyperparams, random_state=42, verbose=0)   
        elif self.model_name == "KNeighbors":
            return model_dict[self.model_name](**self.model_hyperparams)   
        else:
            return model_dict[self.model_name](**self.model_hyperparams, random_state=42)   



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
        self.trial = trial
        
    def instantiate_numerical_simple_imputer(self, strategy: str=None, fill_value: int=-1) -> SimpleImputer:
        if strategy is None and self.trial:
            strategy = self.trial.suggest_categorical('numerical_strategy', ['mean', 'median', 'most_frequent'])
        return SimpleImputer(strategy=strategy, fill_value=fill_value)

    def instantiate_categorical_simple_imputer(self, strategy: str=None, fill_value: str='missing') -> SimpleImputer:
        if strategy is None and self.trial:
            strategy = self.trial.suggest_categorical('categorical_strategy', ['most_frequent', 'constant'])
        return SimpleImputer(strategy=strategy, fill_value=fill_value)
    
    def instantiate_outliers(self, strategy: str=None) -> Union[PowerTransformer, FunctionTransformer, OutlierDetector]:
        """
        Instantiate outlier handling method: PowerTransformer, LogTransformer, or OutlierDetector.

        Args:
            trial (optuna.Trial, optional): The trial object for hyperparameter optimization.

        Returns:
            Union[PowerTransformer, FunctionTransformer, OutlierDetector]: The selected outlier handling method.
        """
        # Suggest from available options
        options = ['power_transform', 'log_transform', 'iqr_clip', 'iqr_median', 'iqr_mean']
        if self.trial:
            strategy = self.trial.suggest_categorical('outlier_strategy', options)
        else:
            strategy = strategy  # Default to first option if no trial is provided

        if strategy == 'power_transform':
            return PowerTransformer(method='yeo-johnson')
        elif strategy == 'log_transform':
            return LogTransformer()
            #return FunctionTransformer(np.log1p)  # Log transformation
        elif strategy in ['iqr_clip', 'iqr_median', 'iqr_mean']:
            return OutlierHandler(strategy=strategy)  # Instantiate OutlierDetector
        else:
            raise ValueError(f"Unknown strategy for outlier handling: {strategy}")

         
    def build(self, step_name=None, **column_transformer_strategy):
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
            
            numerical_strategy = column_transformer_strategy.get('numerical_strategy', None)
            categorical_strategy = column_transformer_strategy.get('categorical_strategy',None)
            outlier_strategy = column_transformer_strategy.get('outlier_strategy', None)        
            
            return ColumnTransformer(
                transformers=[
                    ('categorical', Pipeline([
                        ('imputer', self.instantiate_categorical_simple_imputer(strategy=categorical_strategy)),   
                        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))
                    ]), self.onehot_features),
                    
                    ('numerical', Pipeline([
                        ('imputer', self.instantiate_numerical_simple_imputer(strategy=numerical_strategy)),
                        #('scaler', StandardScaler())  # Add scaler if needed
                    ]), self.numerical_features),
                    
                    
                    
                    ('ordinal', Pipeline([
                        ('imputer', self.instantiate_categorical_simple_imputer(strategy=categorical_strategy)),
                        ('ordinal', OrdinalEncoder())
                    ]), self.ordinal_features),
                    
                    ('outlier_transform', Pipeline([
                        ('imputer', self.instantiate_numerical_simple_imputer(strategy=numerical_strategy)),
                        ('outlier_transformer', self.instantiate_outliers(strategy=outlier_strategy))  # Update this line
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

    def get_dimensionality_reduction(self, dim_red=None, pca_n_components=5):
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
                pca_n_components = pca_n_components  # Default value if trial is not provided
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
        
        # Get the model artefact directory path
        self.model_trainer_artefacts_dir = self.model_trainer_config.MODEL_TRAINER_ARTEFACTS_DIR
        
        self.best_model_artefacts_dir = self.model_trainer_config.BEST_MODEL_ARTEFACTS_DIR
        os.makedirs(self.best_model_artefacts_dir, exist_ok=True)
        logging.info(f"Created Best Model  Artefacts Directory: {self.best_model_artefacts_dir}")
    
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
        self.y_train = y_train.map(self.yes_no_map)  # Map target labels
        
        self.X_test = X_test.copy()
        # target label to 1 and 0
        self.y_test = y_test.map(self.yes_no_map)  # Map target labels
        logging.info("Completed setting the Train and Test X and y")
        
        
    def get_pipeline_model_and_params(self, classifier_name, trial, model_hyperparams=None):
         
        # Got the Preprocessed Pipeline containting Data Cleaning and Column Transformation
    
        preprocessing_pipeline = PreprocessingPipeline(
            bins_hour=self.bins_hour,
            names_period=self.names_period,
            drop_columns=self.drop_columns,
            numerical_features=self.numerical_features,
            onehot_features=self.onehot_features,
            ordinal_features=self.ordinal_features,
            transform_features=self.transform_features,
            trial=trial
        )
        
        # Initialize the manager with the preferred pipeline type ('ImbPipeline' or 'Pipeline')
        pipeline_manager = PipelineManager(pipeline_type='ImbPipeline')
        
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
        dim_red_selector = DimensionalityReductionSelector(trial=trial) 
        dim_red_obj = dim_red_selector.get_dimensionality_reduction()
        pipeline_manager.add_step('dim_reduction', dim_red_obj, position=6)

        # Create an instance of the ModelFactory class with best_model and best_params
        model_factory = ModelFactory(classifier_name, model_hyperparams)
        model_obj = model_factory.get_model_instance()
        pipeline_manager.add_step('model', model_obj, position=7)
        
        pipeline = pipeline_manager.get_pipeline()
        
        return pipeline
    
    
    def get_classification_metrics(self, y_true, y_pred, y_pred_proba=None):
        accuracy = accuracy_score(y_true, y_pred) # Calculate Accuracy
        f1 = f1_score(y_true, y_pred) # Calculate F1-score
        precision = precision_score(y_true, y_pred) # Calculate Precision
        recall = recall_score(y_true, y_pred)  # Calculate Recall
        if y_pred_proba is not None:
            roc_auc = roc_auc_score(y_true, y_pred_proba)
        else:
            roc_auc = roc_auc_score(y_true, y_pred) #Calculate Roc
        detailed_report = classification_report(y_true, y_pred, output_dict=True)  # Detailed report
            
        return {
            'f1': f1,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc,
            "classification_report":  detailed_report
        }
            
         

        
        
    # Define objective function for Optuna
    def objective(self,  classifier_name: str, trial: optuna.Trial=None, scoring='f1') -> float:
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
        model_hyperparams = hyperparameter_tuner.get_params(trial, classifier_name)
            
        
        pipeline = self.get_pipeline_model_and_params(classifier_name=classifier_name, trial=trial, model_hyperparams=model_hyperparams)
        

        # Cross-validation
        kfold = StratifiedKFold(n_splits=10)
        score = cross_val_score(pipeline, self.X_train, self.y_train, scoring=scoring, n_jobs=-1, cv=kfold, verbose=0, error_score='raise')
        score_training = score.mean()
        
        pipeline.fit(self.X_train, self.y_train)
        
        y_pred = pipeline.predict(self.X_test)
        y_pred_proba = pipeline.predict_proba(self.X_test)[:, 1]
        
        classification_metrics = self.get_classification_metrics(self.y_test, y_pred, y_pred_proba)
        
        classification_metrics['classifier_name'] = classifier_name
        classification_metrics['training_score'] = score_training
        
        # Return the metric used in the scoring the training        
        return classification_metrics[scoring]
    
    
    # Define objective function for Optuna
    def detailed_objective(self, classifier_name: str, trial: optuna.Trial=None, scoring='f1') -> ImbPipeline:
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
        model_hyperparams = hyperparameter_tuner.get_params(trial, classifier_name)
         
        
        # Got the Preprocessed Pipeline containting Data Cleaning and Column Transformation        
        pipeline = self.get_pipeline_model_and_params(classifier_name, trial, model_hyperparams=model_hyperparams)
        logging.info(f"Fitted pipeline with the best parameters: {classifier_name}")
        
        # Cross-validation
        #kfold = StratifiedKFold(n_splits=10)
        #score = cross_val_score(pipeline, self.X_train, self.y_train, scoring=scoring, n_jobs=-1, cv=kfold, verbose=0, error_score='raise')
        #score_training = score.mean()
        
        return  pipeline
    
    
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
        best_of_models = []
        best_pipeline = {}

        all_models = ["RandomForest", "DecisionTree", 
                    "XGBoost", "LGBM", "GradientBoosting", 
                    "LogisticRegression", "KNeighbors", "CatBoost"]
        
        all_models = ["RandomForest", "DecisionTree",  "XGBoost", "LGBM", "GradientBoosting", ]
        
        for classifier_name in all_models:
            logging.info(f"Optimizing model: {classifier_name}")
            study = optuna.create_study(direction="maximize", sampler=TPESampler())
            study.optimize(lambda trial: self.objective(classifier_name, trial,  scoring), n_trials=n_trials)
            
            best_trial_obj = study.best_trial
            
            score_training = best_trial_obj.value 
            
            
            # patch the pipeline with tbe frozen best model trials
            pipeline = self.detailed_objective(classifier_name=classifier_name, trial=study.best_trial,  scoring=scoring)
        
            pipeline.fit(self.X_train, self.y_train)
        
            y_pred = pipeline.predict(self.X_test)
            y_pred_proba = pipeline.predict_proba(self.X_test)[:, 1]
            
            classification_metrics = self.get_classification_metrics(self.y_test, y_pred, y_pred_proba)
            classification_metrics['training_score'] = score_training
            
            current_test_score =  classification_metrics[scoring]
             
            
            
            # Serialise the trained pipeline
            trained_model_filename = f'{classifier_name}_pipeline{MODEL_SAVE_FORMAT}'
            trained_model_saved_path = os.path.join(self.model_trainer_artefacts_dir, trained_model_filename)
            joblib.dump((pipeline, classification_metrics), trained_model_saved_path)
            print(f'Serialized {classifier_name} pipeline and test metrics to {trained_model_saved_path}')         
            
            logging.info(f"Model: {classifier_name}, Current Score: {current_test_score} | Best Model: {best_model}, Best Score: {best_model_score}")
        
            if current_test_score and current_test_score > best_model_score:
                best_model_score = current_test_score
                best_model = classifier_name
                best_params = best_trial_obj.params    
            
                best_of_models.append({
                    "model": classifier_name,
                    "params": best_trial_obj.params,
                    "model_score_params": best_trial_obj.params,
                    "model_score_trial_number": best_trial_obj.number,                
                    "model_score_duration": best_trial_obj.duration,
                    "model_score_status": best_trial_obj.state,
                    "model_score_key": scoring,
                    "model_score_value": best_trial_obj.value,
                })
                
                best_pipeline[best_model] = pipeline
                
                
            
            
        # Display all results and the best model
        for result in best_of_models:
            logging.info(result) 
            
        # Save the Overal Best Model Pipeline for Predictiona in Inference
        best_trained_model_filename = f'best_model_pipeline{MODEL_SAVE_FORMAT}'
        best_trained_model_saved_path = os.path.join(self.best_model_artefacts_dir, best_trained_model_filename)
        joblib.dump(best_pipeline[best_model], best_trained_model_saved_path)
        print(f'Serialized best model pipeline {classifier_name} to {best_trained_model_saved_path}')              
            
        return best_model, best_params, current_test_score

        
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
            best_model_name, best_model_params, best_model_score = self.run_optimization(config_path=[], n_trials=30, scoring='f1')
            logging.info("Completed the training process")
            logging.info(f"Best params: {best_model_params}")
            logging.info(f"Best params: {best_model_params}")
            logging.info(f"The best model score from the model training: {best_model_score}")
            
                       
            # Reading model config file for getting the best model score
            model_config = self.model_trainer_config.UTILS.read_yaml_file(
                filename=MODEL_CONFIG_FILE
            )
            base_model_score = float(model_config["base_model_score"])
            logging.info(f"Got the best model score from model config file: {base_model_score}")
             

            # Updating the best model score to model config file if the model score is greather than the base model score
            if best_model_score >= base_model_score:
                best_model_info  = {'best_model_score': best_model_score, 'best_model_name': best_model_name}
                self.model_trainer_config.UTILS.update_model_score(best_model_info)
                
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
                    trained_model_path, (best_model_name, best_model_params, best_model_score))
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
