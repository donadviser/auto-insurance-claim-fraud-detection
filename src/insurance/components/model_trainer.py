import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import optuna
from typing import Union, Dict, Tuple, List
from typing_extensions import Annotated
from optuna.samplers import TPESampler
from dataclasses import dataclass

from insurance import logging
from insurance import CustomException
from insurance.constants import MODEL_CONFIG_FILE, MODEL_SAVE_FORMAT, PARAM_FILE_PATH
from insurance.entity import ModelTrainerConfig
from insurance.entity import (
    DataIngestionArtefacts,
    DataTransformationArtefacts,
    ModelTrainerArtefacts,
)

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder, StandardScaler,
    OrdinalEncoder, PowerTransformer,
    RobustScaler, MinMaxScaler,
    FunctionTransformer)
from sklearn.model_selection import   cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier,
                              AdaBoostClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

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

import mlflow
import mlflow.sklearn




@dataclass
class TrainModelMetrics:
    """Dataclass to encapsulate model evaluation metrics and comparison results."""
    accuracy_score: float
    f1_score: float
    precision_score: float
    recall_score: float
    roc_auc_score: float


class CostModel:
    def __init__(self,
                 pipeline_model: object,
                 X_train: pd.DataFrame = None,
                 y_train: pd.DataFrame = None,
                 preprocess_pipeline: object=None,
                ):
        """
        Initialize the CostModel class

        Args:
            pipeline_model (object): model or model in the pipeline
            X_train (pd.DataFrame, optional): data (features). Needed if training is required
            y_train (pd.DataFrame, optional): labels. Needed if training is required
            preprocess_pipeline (object, optional): _description_. Defaults to None.
        """
        self.preprocess_pipeline = preprocess_pipeline
        self.pipeline_model = pipeline_model
        self.X_train = X_train
        self.y_train = y_train

    @staticmethod
    def _handle_exception(e: Exception) -> None:
        raise CustomException(e, sys)


    def train(self) -> object:
        """Train the model with the provided data."""
        try:
            if self.preprocess_pipeline:
                self.X_train = self.preprocess_pipeline.fit_transform(self.X_train)
            self.pipeline_model.fit(self.X_train, self.y_train)
            return self.pipeline_model
        except Exception as e:
            self._handle_exception(e)


    def predict(self, X_test) -> Tuple[
        Annotated[float, "y_pred"],
        Annotated[float, "y_pred_proba"]
        ]:
        """
        This method predicts the data

        Args:
            X (pd.DataFrame): The data to be predicted.

        Returns:
            float: The predicted data.
        """
        try:
            if self.preprocess_pipeline:
                X_test = self.preprocess_pipeline.transform(X_test)
                logging.info("Transformed the X_test using preprocess_pipeline to get predictions")

            y_pred = self.pipeline_model.predict(X_test)
            y_pred_proba = self.pipeline_model.predict_proba(X_test)[:, 1]

            return y_pred, y_pred_proba
        except Exception as e:
            self._handle_exception(e)


    def evaluate(self, y_test, y_pred, y_pred_proba=None)-> Dict[str, float]:
        try:
            accuracy = accuracy_score(y_test, y_pred)  # Calculate Accuracy
            f1 = f1_score(y_test, y_pred, average='weighted')  # Calculate F1-score
            precision = precision_score(y_test, y_pred, average='weighted')  # Calculate Precision
            recall = recall_score(y_test, y_pred, average='weighted')  # Calculate Recall
            roc_auc = roc_auc_score(y_test, y_pred_proba, average='weighted') if y_pred_proba is not None else roc_auc_score(y_test, y_pred, average='weighted')  # Calculate Roc
            #detailed_report = classification_report(y_test, y_pred, output_dict=True)  # Detailed report

            return {
                'f1': f1,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'roc_auc': roc_auc,
                #"classification_report": detailed_report
            }
        except Exception as e:
            self._handle_exception(e)



    def __repr__(self) -> str:
        return f"{type(self.pipeline_model).__name__}()"

    def __str__(self) -> str:
        return f"{type(self.pipeline_model).__name__}()"


class HyperparameterTuner:
    """
    HyperparameterTuner to return hyperparameters for each classifier.
    """

    def __init__(self):
        pass

    def get_params(self, trial: optuna.trial.Trial, model_name: str, classifier_params):
        """
        Get hyperparameters for a specified classifier.

        Args:
            trial (optuna.Trial): Optuna trial instance for hyperparameter suggestions.
            model_name (str): Name of the classifier.

        Returns:
            dict: A dictionary of hyperparameters.
        """
        logging.info(f"Entered get_params, model_name: {model_name}")

        # Fetch classifier-specific parameters
        model_params = classifier_params.get(model_name, {})
        params = {}


        # Fetch classifier-specific parameters
        for param, settings in model_params.items():
            if isinstance(settings, dict):
                param_type = settings.get("type")
                min_val = settings.get("min")
                max_val = settings.get("max")

                # Generate parameter suggestions based on type
                if param_type == "int":
                    params[param] = trial.suggest_int(param, min_val, max_val)
                elif param_type == "float":
                    params[param] = trial.suggest_float(param, min_val, max_val, log=settings.get("log", False))
                elif param_type == "categorical":
                    params[param] = trial.suggest_categorical(param, settings["choices"])
            else:
                logging.info(f"model_name: {model_name} | param: {param} | settings: {settings}")
                params[param] = settings  # Fixed parameters
        logging.info(f"Exiting get_params, model_name: {model_name} | params: {params}")
        return params


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
            "LGBMClassifier": LGBMClassifier,
            "XGBClassifier": XGBClassifier,
            "CatBoostClassifier": CatBoostClassifier,
            "RandomForestClassifier": RandomForestClassifier,
            "DecisionTreeClassifier": DecisionTreeClassifier,
            "LogisticRegression": LogisticRegression,
            "SVC": SVC,
            "GradientBoostingClassifier": GradientBoostingClassifier,
            "KNeighborsClassifier": KNeighborsClassifier,
            "AdaBoostClassifier": AdaBoostClassifier
        }


        if self.model_name not in model_dict:
            raise ValueError(f"Model {self.model_name} is not supported.")

        # Create a model instance with specific parameters
        if self.model_name == "KNeighborsClassifier":
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
        if self.trial and strategy is None:
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
            return ReplaceValueTransformer(old_value="?", new_value='missing')

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

            """resampler = self.trial.suggest_categorical(
                'resampler', ['SMOTEENN',]
            )"""

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

        # -- Instantiate scaler (skip scaler for CatBoostClassifier as it handles categorical features internally)
        if scaler_name is None and self.trial:
            scaler_name = self.trial.suggest_categorical("scaler", ['minmax', 'standard', 'robust'])
            #scaler_name = self.trial.suggest_categorical("scaler", ['robust'])

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


        #mlflow.set_tracking_uri("http://localhost:5000")

         # Get the model parameters from model config file
        self.model_config = self.model_trainer_config.UTILS.read_yaml_file(filename=MODEL_CONFIG_FILE)

        self.classifiers = [list(classifier.keys())[0] for classifier in self.model_config.get('classifiers', [])]

        self.classifier_params = {classifier: params
                                  for clf_config in self.model_config.get('classifiers', [])
                                  for classifier, params in clf_config.items()
                                  }
        logging.info(f"self.classifiers: {self.classifiers}")
        logging.info(f"self.classifier_params: {self.classifier_params}")

        # Get the params from the params.yaml file
        self.param_constants = self.model_trainer_config.UTILS.read_yaml_file(filename=PARAM_FILE_PATH)
        logging.info(f"self.param_constants: {self.param_constants}")


        # Get the model artefact directory path
        self.model_trainer_artefacts_dir = self.model_trainer_config.MODEL_TRAINER_ARTEFACTS_DIR

        # Get the train test artefact directory path
        self.metric_artefacts_dir = self.model_trainer_config.METRIC_ARTEFACTS_DIR
        os.makedirs(self.metric_artefacts_dir, exist_ok=True)



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


        self.X_train, self.y_train = self.model_trainer_config.UTILS.separate_data(
            self.train_set,
            self.target_columns,
            self.yes_no_map,
            )

        self.X_test, self.y_test = self.model_trainer_config.UTILS.separate_data(
            self.test_set,
            self.target_columns,
            self.yes_no_map,
            )
        logging.info("Completed separating X and y into the X_train, y_train, X_test and y_test")

    @staticmethod
    def _handle_exception(e: Exception) -> None:
        raise CustomException(e, sys)

    def get_pipeline_model_and_params(self, trial, model_name,  model_hyperparams=None):

        # Got the Preprocessed Pipeline containting Data Cleaning and Column Transformation
        try:
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
            #dim_red_selector = DimensionalityReductionSelector(trial=trial)
            #dim_red_obj = dim_red_selector.get_dimensionality_reduction()
            #pipeline_manager.add_step('dim_reduction', dim_red_obj, position=6)

            # Create an instance of the ModelFactory class with best_model and best_params
            model_factory = ModelFactory(model_name, model_hyperparams)
            model_obj = model_factory.get_model_instance()
            pipeline_manager.add_step('model', model_obj, position=7)

            pipeline = pipeline_manager.get_pipeline()

            return pipeline

        except Exception as e:
            self._handle_exception(e)



    # Run the Optuna study
    def run_optimization(self) -> None:
        """
        Run Optuna study for hyperparameter tuning and model selection.
        """
        try:

            classifier_params =self.classifier_params

            logging.info(f"classifier_params: {classifier_params}")
            hyperparameter_tuner = HyperparameterTuner()

            all_trained_models = {}
            kfold = StratifiedKFold(n_splits=self.param_constants['N_SPLITS'])



            classifier_short_names = {
            "KNeighborsClassifier": "KNeighbors",
            "RandomForestClassifier": "RandomForest",
            "GradientBoostingClassifier": "GradientBoosting",
            "LogisticRegression": "LogisticRegression",
            "SVC": "SVC",
            "DecisionTreeClassifier": "DecisionTree",
            "LGBMClassifier": "LightGBM",
            "XGBClassifier": "XGB",
            "CatBoostClassifier": "CatBoost",
            "AdaBoostClassifier": "AdaBoost",
            }

            scores_dict = {}
            best_training_score = 0
            best_trained_model = None

            if self.param_constants["CLASSIFIER"] == "ALL":
                self.model_available = self.classifiers
            else:
                self.model_available = self.param_constants["CLASSIFIER"]

            for model_name in self.model_available:
                logging.info(f"Starting tuning and training for {model_name}")
                # Initialize scores list for the current model
                model_short_name = classifier_short_names.get(model_name, model_name)
                scores_dict[model_short_name] = []
                # Define Optuna objective
                def objective(trial):
                    model_hyperparams = hyperparameter_tuner.get_params(
                        trial=trial,
                        model_name=model_name,
                        classifier_params=classifier_params
                        )
                    logging.info(f"model_hyperparams: {model_hyperparams}")
                    pipeline = self.get_pipeline_model_and_params(trial=trial, model_name=model_name, model_hyperparams=model_hyperparams)
                    # Cross-validation
                    scores = cross_val_score(pipeline, self.X_train, self.y_train,
                                            scoring=self.param_constants['SCORING'],
                                            n_jobs=self.param_constants['N_JOBS'],
                                            cv=kfold,
                                            verbose=self.param_constants['VERBOSE'],
                                            error_score='raise')
                    mean_score = scores.mean()
                    scores_dict[model_short_name].extend(scores)
                    return mean_score
                logging.info(f'completed cross-validation for the model_name: {model_name}')
                study = optuna.create_study(direction="maximize", sampler=TPESampler())
                study.optimize(objective, n_trials=self.param_constants['N_TRIALS'])

                # Train final pipeline with best parameters from Optuna
                # Get hyperparameters for the classifier from HyperparameterTuner
                logging.info(f'Training {model_name} with the best parameters obtained from Optuna tunning')
                model_hyperparams = hyperparameter_tuner.get_params(trial=study.best_trial, model_name=model_name, classifier_params=classifier_params)
                pipeline = self.get_pipeline_model_and_params(trial=study.best_trial, model_name=model_name, model_hyperparams=model_hyperparams)

                trainer = CostModel(pipeline, self.X_train, self.y_train)
                trained_pipeline = trainer.train()
                y_pred, y_pred_proba = trainer.predict(self.X_train)
                evaluation_scores = trainer.evaluate(self.y_train, y_pred, y_pred_proba)
                model_score = evaluation_scores[self.param_constants['SCORING']]
                logging.info(f"Current Model: {model_name}, Best Current Model Trained Score: {model_score}")
                logging.info(f"Best Current Model Params: {study.best_params}")

                # Serialise the trained pipeline
                all_trained_models[model_name] = trained_pipeline
                if model_score >= best_training_score:
                    best_training_score = model_score
                    best_trained_model = model_name

            logging.info(f"Overall Best Trained Model: {best_trained_model}, Best Trained Model Score: {best_training_score}")

            # Plotting boxplot for the training scores of each classifier
            sns.set(style="darkgrid")
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=[scores for scores in scores_dict.values()],
                        orient="v",
                        palette="Set3")
            plt.xticks(ticks=range(len(scores_dict)), labels=scores_dict.keys())
            plt.title("Comparison of Training Scores for Each Classifier")
            plt.xlabel("Classifier")
            plt.ylabel("Optuna Hyperparameter Tunning Cross-validation F1 Score ")


            # Superimposing mean scores as scatter points with higher zorder
            mean_scores = [np.mean(scores) for scores in scores_dict.values()]
            for i, mean_score in enumerate(mean_scores):
                plt.scatter(i, mean_score, color='red', marker='o', s=100, label='Mean Score' if i == 0 else "", zorder=10)
            plt.xticks(rotation=45, ha="right")
            plt.legend()

            file_path = os.path.join(self.metric_artefacts_dir, "Boxplot_training_score.png")
            plt.savefig(file_path, bbox_inches='tight')
            plt.close()
            logging.info("Boxplot for training score saved")

            return  all_trained_models
        except Exception as e:
            self._handle_exception(e)


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
            all_trained_models = self.run_optimization()


            for model_name in self.model_available:
                trained_model_filename = f'{model_name}_pipeline{MODEL_SAVE_FORMAT}'
                trained_model_saved_path = os.path.join(self.model_trainer_artefacts_dir, trained_model_filename)
                trained_pipeline = all_trained_models[model_name]
                self.model_trainer_config.UTILS.save_object(trained_model_saved_path, trained_pipeline)
                logging.info(f'Serialized {model_name} trained pipeline to {trained_model_saved_path}')

            # Savind the Model trainer artefacts
            metric_artefacts_dir = self.metric_artefacts_dir
            model_file_path = self.model_trainer_artefacts_dir
            model_trainer_artefacts = ModelTrainerArtefacts(
                trained_model_file_path=model_file_path,
                metric_artefacts_dir=metric_artefacts_dir,
                trained_model_names=self.model_available
            )


            logging.info(f"Returned the model trainer artefacts directory: {model_file_path}")
            logging.info(f"Returned the metric artefacts directory: {metric_artefacts_dir}")
            logging.info("Exited the initiate_model_trainer method of ModelTrainer class.")
            return model_trainer_artefacts
        except Exception as e:
            self._handle_exception(e)
