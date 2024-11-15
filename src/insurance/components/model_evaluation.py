import sys
import os
import pandas as pd
from dataclasses import dataclass
from insurance import logging
from insurance import CustomException
from insurance.constants import (
    MODEL_BUCKET_NAME, MODEL_CONFIG_FILE,
    S3_MODEL_NAME, MODEL_SAVE_FORMAT, PARAM_FILE_PATH
    )
from insurance.entity.config_entity import ModelEvaluationConfig
from insurance.entity.artefacts_entity import (
    DataIngestionArtefacts,
    ModelTrainerArtefacts,
    ModelEvaluationArtefacts,
)

from insurance.components.model_trainer import CostModel
from insurance.utils.evaluation_artefacts import ModelDiagnosticsLogger
from insurance.utils.shap_visualization_logger import SHAPLogger

import mlflow
import mlflow.sklearn
#from mlflow.models import infer_signature


@dataclass
class EvaluateModelResponse:
    """Dataclass to encapsulate model evaluation metrics and comparison results."""
    trained_model_f1_score: float
    s3_model_f1_score: float
    is_model_accepted: bool
    difference: float


class ModelEvaluation:
    """Class for evaluating the trained model against a production model stored in S3."""

    def __init__(
        self,
        model_trainer_artefact: ModelTrainerArtefacts,
        model_evaluation_config: ModelEvaluationConfig,
        data_ingestion_artefact: DataIngestionArtefacts,
    ):
        self.model_trainer_artefact = model_trainer_artefact
        self.model_evaluation_config = model_evaluation_config
        self.data_ingestion_artefact = data_ingestion_artefact

        # Get the model parameters from model config file
        self.model_config = self.model_evaluation_config.UTILS.read_yaml_file(filename=MODEL_CONFIG_FILE)

        #self.classifiers = self.model_config['classifiers']
        logging.info("ModelEvaluation initialised with configuration and artefacts.")
        # Get the params from the params.yaml file
        self.param_constants = self.model_evaluation_config.UTILS.read_yaml_file(filename=PARAM_FILE_PATH)
        logging.info(f"self.param_constants: {self.param_constants}")

        # Directory to save the best model pipeline
        self.best_model_artefacts_dir = self.model_evaluation_config.BEST_MODEL_ARTEFACTS_DIR
        os.makedirs(self.best_model_artefacts_dir, exist_ok=True)

        #mlflow.set_tracking_uri("http://127.0.0.1:5001")
        #logging.info(f"mflow: {mlflow.get_tracking_uri()} set")

    @staticmethod
    def _handle_exception(e: Exception) -> None:
        raise CustomException(e, sys)

    def get_s3_model(self) -> object:
        """Fetch the production model from S3 if available.

        Returns:
            object: Loaded model object if present in S3; otherwise, None.
        """
        try:
            logging.info("Fetching the S3 model from the specified bucket.")
            status = self.model_evaluation_config.S3_OPERATIONS.is_model_present(MODEL_BUCKET_NAME, S3_MODEL_NAME)
            if status:
                model = self.model_evaluation_config.S3_OPERATIONS.load_model(S3_MODEL_NAME, MODEL_BUCKET_NAME)
                logging.info("S3 model successfully loaded.")
                return model
            logging.info("No model found in S3 bucket.")
            return None
        except Exception as e:
            self._handle_exception(e)

    def evaluate_model(self) -> EvaluateModelResponse:
        """Evaluate the trained model and compare it with the production model from S3.

        Returns:
            EvaluateModelResponse: Encapsulated response with f1 scores and acceptance status.
        """
        try:
            # Load test data
            test_data_file_path = self.data_ingestion_artefact.test_data_file_path
            train_data_file_path = self.data_ingestion_artefact.train_data_file_path

            test_df = pd.read_csv(test_data_file_path)
            logging.info(f"Test data loaded from the path: {test_data_file_path}")
            train_df = pd.read_csv(train_data_file_path)
            logging.info(f"Train data loaded from the path: {train_data_file_path}")

            self.yes_no_map = self.model_evaluation_config.SCHEMA_CONFIG['yes_no_map']
            self.target_columns = self.model_evaluation_config.SCHEMA_CONFIG['target_column']
            logging.info("Data loaded, test features and target extracted.")

            logging.info("Separate the test data and map the target label")
            self.X_test, self.y_test = self.model_evaluation_config.UTILS.separate_data(
            test_df,
            self.target_columns,
            self.yes_no_map,
            )

            logging.info("Separate the train data and map the target label")
            self.X_train, self.y_train = self.model_evaluation_config.UTILS.separate_data(
            train_df,
            self.target_columns,
            self.yes_no_map,
            )
            # Evaluate trained model
            evaluation_score_param = self.param_constants['SCORING']

            trained_model_dir_path = self.model_trainer_artefact.trained_model_file_path
            metric_artefacts_dir = self.model_trainer_artefact.metric_artefacts_dir
            trained_model_names = self.model_trainer_artefact.trained_model_names

            best_model_pipeline = None
            best_model_name = None
            best_model_score = 0


            #logging.info(f"Started MLflow experiment: insurance")

            logging.info("Find the best trained model based on the scoring metric")
            for model_name in trained_model_names:
                logging.info(f"Starting evaluating model: {model_name}")

                trained_model_filename = f'{model_name}_pipeline{MODEL_SAVE_FORMAT}'
                trained_model_saved_path = os.path.join(trained_model_dir_path, trained_model_filename)
                logging.info(f"trained model path: {trained_model_saved_path}")

                trained_pipeline = self.model_evaluation_config.UTILS.load_object(trained_model_saved_path)
                logging.info(f'Deserialized {model_name} trained pipeline from {trained_model_saved_path}')

                # Evaluate the trained model with test data
                logging.info(f"Start prededicting with {model_name} pipeline")
                pipeline = CostModel(trained_pipeline)
                y_pred, y_pred_proba = pipeline.predict(self.X_test)
                evaluation_scores = pipeline.evaluate(self.y_test, y_pred, y_pred_proba)

                # Infer the model signature
                #signature = infer_signature(self.X_test, pipeline.predict(self.X_test))

                model_score = evaluation_scores[evaluation_score_param]
                logging.info(f"Model: {model_name}, Model Score ({evaluation_score_param}): {model_score}")

                if model_score > best_model_score:
                        best_model_score = model_score
                        best_model_name = model_name
                        best_model_pipeline=pipeline


                # Evaluation Artefacts
                logging.info(f"Entres ModelDiagonisticsLogger for evaluation metrics plots for {model_name}")
                logging.info(f"metric_artefacts_dir: {metric_artefacts_dir}")
                evaluator = ModelDiagnosticsLogger(trained_pipeline,
                                                    self.X_test, self.y_test,
                                                    model_name,
                                                    artefact_path=metric_artefacts_dir,
                                                    mlflow_tracking=False)
                evaluator.log_model_diagnostics()

                shap_logger = SHAPLogger(pipeline=trained_pipeline,
                                         X_train=self.X_train,
                                         X_test=self.X_test,
                                         model_name=model_name,
                                         artefact_dir_path=metric_artefacts_dir,
                                         mlflow_tracking=False
                                         )
                shap_logger.log_all()



                """#Ensure any active run is ended
                if mlflow.active_run():
                    mlflow.end_run()
                # Start an MLflow run for each model
                mlflow.set_experiment("auto_insurance")
                with mlflow.start_run(run_name=model_name):

                    mlflow.set_tag("version","1.0.0")
                    model_params = trained_pipeline.named_steps['model'].get_params()
                    mlflow.log_params(model_params)
                    logging.info(f"evaluation scores: {evaluation_scores}")
                    mlflow.log_metrics(evaluation_scores)


                    if model_score > best_model_score:
                        best_model_score = model_score
                        best_model_name = model_name
                        best_model_pipeline=pipeline


                    #model_only = trained_pipeline.named_steps['model']
                    if model_name.lower().startswith("xgb") is True:
                        mlflow.sklearn.log_model(trained_model_saved_path, artifact_path="model")
                    elif model_name.lower().startswith("lgb") is True:
                        mlflow.sklearn.log_model(trained_model_saved_path, artifact_path="model")
                    elif model_name.lower().startswith("cat") is True:
                        mlflow.sklearn.log_model(trained_model_saved_path, artifact_path="model")
                    else:
                        mlflow.sklearn.log_model(trained_model_saved_path, artifact_path="model")

                    # Log additional metrics
                    mlflow.log_metric("best_score", model_score)
                    mlflow.log_artifact(trained_model_saved_path)

                    # Evaluation Artefacts
                    evaluator = ModelDiagnosticsLogger(trained_pipeline,
                                                       self.X_test, self.y_test,
                                                       model_name,
                                                       artefact_path=metric_artefacts_dir,
                                                       mlflow_tracking=True)
                    evaluator.log_model_diagnostics()

                    shap_logger = SHAPLogger(pipeline=trained_pipeline,
                                             X_train=self.X_train,
                                             X_test=self.X_test,
                                             model_name=model_name,
                                             artefact_dir_path=metric_artefacts_dir,
                                             mlflow_tracking=True
                                            )
                    shap_logger.log_all()"""



            trained_model_f1_score = best_model_score
            logging.info(f"Best Trained Model -  Name: {best_model_name} with F1 score: {trained_model_f1_score}")

            # Save the best model pipeline
            self.best_trained_model_path = self.model_evaluation_config.BEST_MODEL_PATH
            logging.info(f"Created best model file path: {self.best_trained_model_path}")
            self.model_evaluation_config.UTILS.save_object(
                    self.best_trained_model_path, best_model_pipeline)
            logging.info("Saved the best model object path")

            # Reading model config file for getting the base model score
            base_model_score = float(self.model_config['base_model_score'])
            base_model_name = self.model_config['base_model_name']
            logging.info(f"Base model -  Name: {base_model_name} with F1 score: {base_model_score}")

            # If base model score is not available, set it to 0
            if base_model_score is None:
                base_model_score = 0.0


            s3_model_f1_score = 0.0

            # If trained model score is less than the base model score, set it to the base model score
            if trained_model_f1_score >= base_model_score:
                best_model_info  = {'best_model_score': trained_model_f1_score, 'best_model_name': best_model_name}
                self.model_evaluation_config.UTILS.update_model_score(best_model_info)
                logging.info("Updated the best model score to model config file")



                # Load cost model object with preprocessor and model
                #cost_model = CostModel(preprocessing_obj, best_model_pipeline)

                # Evaluate S3 model if available
                s3_model = self.get_s3_model()
                #s3_model = None # Uncommet to prevent aws S3 access
                if s3_model:
                    y_pred, y_pred_proba = s3_model.predict(self.X_test)
                    evaluation_scores = pipeline.evaluate(self.y_test, y_pred, y_pred_proba)

                    s3_model_f1_score = evaluation_scores[evaluation_score_param]
                    logging.info(f"S3 model F1 score: {s3_model_f1_score}")

                # Decision making
                is_model_accepted = trained_model_f1_score > s3_model_f1_score
                difference = trained_model_f1_score - s3_model_f1_score

            else:
                is_model_accepted = False
                difference = trained_model_f1_score

            result = EvaluateModelResponse(
                    trained_model_f1_score=trained_model_f1_score,
                    s3_model_f1_score=s3_model_f1_score,
                    is_model_accepted=is_model_accepted,
                    difference=difference
                )

            logging.info(f"Model evaluation result: {result}")
            return result
        except Exception as e:
            logging.error("Error during model evaluation.", exc_info=True)
            self._handle_exception(e)

    def initiate_model_evaluation(self) -> ModelEvaluationArtefacts:
        """Initiate the model evaluation and create artefacts for the result.

        Returns:
            ModelEvaluationArtefacts: Object with evaluation results to be stored.
        """
        try:
            evaluate_model_response = self.evaluate_model()


            model_evaluation_artefacts = ModelEvaluationArtefacts(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                trained_model_path=self.best_trained_model_path,
                changed_accuracy=evaluate_model_response.difference,
            )
            logging.info("Model evaluation artefacts created.")
            return model_evaluation_artefacts
        except Exception as e:
            logging.error("Error initiating model evaluation.", exc_info=True)
            self._handle_exception(e)
