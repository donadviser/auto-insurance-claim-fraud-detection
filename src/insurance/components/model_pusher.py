import sys
from insurance  import logging
from insurance  import CustomException

from insurance.configuration.s3_operations import S3Operations
from insurance.entity.artefacts_entity import (
    DataTransformationArtefacts,
    ModelTrainerArtefacts,
    ModelPusherArtefacts,
)
from insurance.entity.config_entity import ModelPusherConfig



class ModelPusher:
    def __init__(
            self,
            model_pusher_config: ModelPusherConfig,
            model_trainer_artefacts: ModelTrainerArtefacts,
            data_transformation_artefacts: DataTransformationArtefacts,
            s3: S3Operations,
    ):
        self.model_pusher_config = model_pusher_config
        self.model_trainer_artefacts = model_trainer_artefacts
        self.data_transformation_artefacts = data_transformation_artefacts
        self.s3 = s3

    # This method is used to push the model to s3
    def initiate_model_pusher(self) -> ModelPusherArtefacts:
        """
        Initiate model pusher.

        Returns:
            ModelPusherArtefacts: The model pusher artefacts.
        """
        logging.info("Entered the initiate_model_pusher method of ModelPusher class.")
        try:
            # Uploading the best model to S3 bucket
            self.s3.upload_file(
                self.model_trainer_artefacts.trained_model_file_path,
                self.model_pusher_config.S3_MODEL_KEY_PATH,
                self.model_pusher_config.BUCKET_NAME,
                remove=False,
            )
            logging.info("Uploaded best model to s3 bucket")
            logging.info("Exited initiate_model_pusher method of ModelPusher class")

            # Saving the model pusher artefacts
            model_pusher_artefact = ModelPusherArtefacts(
                model_bucket_name=self.model_pusher_config.BUCKET_NAME,
                s3_model_path=self.model_pusher_config.S3_MODEL_KEY_PATH
            )
            return model_pusher_artefact
        except Exception as e:
            raise CustomException(e, sys)