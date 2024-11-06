import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.metrics import f1_score
from insurance import logging
from insurance import CustomException
from insurance.constants import TARGET_COLUMN, MODEL_BUCKET_NAME, S3_MODEL_NAME
from insurance.entity.config_entity import ModelEvaluationConfig
from insurance.entity.artefacts_entity import (
    DataIngestionArtefacts,
    ModelTrainerArtefacts,
    ModelEvaluationArtefacts,
)

import mlflow
import mlflow.sklearn


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
        logging.info("ModelEvaluation initialised with configuration and artefacts.")
        
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
            
            test_df = pd.read_csv(test_data_file_path)
            logging.info(f"Test data loaded from the path: {test_data_file_path}")
             
            
            
            self.yes_no_map = self.model_evaluation_config.SCHEMA_CONFIG['yes_no_map']
            self.target_columns = self.model_evaluation_config.SCHEMA_CONFIG['target_column']
            logging.info("Data loaded, test features and target extracted.")
            X_test, y_test = test_df.drop(columns=[self.target_columns]), test_df[self.target_columns]
            logging.info("Test data loaded and split into features (X) and target (y).")
            y_test = y_test.map(self.yes_no_map)  # 
            logging.info("Mapped target labels to 'Y' = 1 and 'N' = 0")
            logging.info(f"Sample of test features X_test.head(): {X_test.head()}.")
            logging.info(f"Sample of test target y_test[:5]: {y_test[:5]}.")
            

            # Evaluate trained model
            trained_model = self.model_evaluation_config.UTILS.load_object(
            self.model_trainer_artefact.trained_model_file_path)
            
            
            y_hat_trained_model = trained_model.predict(X_test)
            trained_model_f1_score = f1_score(y_test, y_hat_trained_model, average='weighted')
            logging.info(f"Trained model F1 score: {trained_model_f1_score}")

            # Evaluate S3 model if available
            s3_model_f1_score = 0.0
            s3_model = self.get_s3_model()
            #s3_model = None
            if s3_model:
                y_hat_s3_model = s3_model.predict(X_test)
                s3_model_f1_score = f1_score(y_test, y_hat_s3_model, average='weighted')
                logging.info(f"S3 model F1 score: {s3_model_f1_score}")

            # Decision making
            is_model_accepted = trained_model_f1_score > s3_model_f1_score
            difference = trained_model_f1_score - s3_model_f1_score
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
                trained_model_path=self.model_trainer_artefact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference,
            )
            logging.info("Model evaluation artefacts created.")
            return model_evaluation_artefacts
        except Exception as e:
            logging.error("Error initiating model evaluation.", exc_info=True)
            self._handle_exception(e)
