from dataclasses import dataclass
from from_root import from_root
import os

from insurance.utils.main_utils import MainUtils
from insurance.constants import *
from insurance import CustomException, logging



@dataclass
class DataIngestionConfig:
    def __init__(self):
        self.UTILS = MainUtils()
        self.SCHEMA_CONFIG = self.UTILS.read_yaml_file(filename=SCHEMA_FILE_PATH)
        self.ROOT_DIR = ARTEFACTS_DIR
        self.RAW_DATA_DIR = RAW_DATA_DIR
        self.SOURCE_URL = SOURCE_URL
        self.LOCAL_FILE_NAME = LOCAL_FILE_NAME
        self.TARGET_COLUMN = TARGET_COLUMN

        self.DOWNLOADED_DATA_FILE_PATH: str = os.path.join(
            self.RAW_DATA_DIR, self.LOCAL_FILE_NAME
            )

        self.DROP_COLS = list(self.SCHEMA_CONFIG["drop_columns"])
        self.DATA_INGESTION_ARTEFACTS_DIR: str = os.path.join(
            from_root(), ARTEFACTS_DIR, DATA_INGESTION_ARTEFACTS_DIR
            )
        self.TRAIN_DATA_ARTEFACT_FILE_DIR: str = os.path.join(
            self.DATA_INGESTION_ARTEFACTS_DIR, DATA_INGESTION_TRAIN_DIR
            )
        self.TEST_DATA_ARTEFACT_FILE_DIR: str = os.path.join(
            self.DATA_INGESTION_ARTEFACTS_DIR, DATA_INGESTION_TEST_DIR
            )
        self.TRAIN_DATA_FILE_PATH: str = os.path.join(
            self.TRAIN_DATA_ARTEFACT_FILE_DIR, DATA_INGESTION_TRAIN_FILE_NAME
            )
        self.TEST_DATA_FILE_PATH: str = os.path.join(
            self.TEST_DATA_ARTEFACT_FILE_DIR, DATA_INGESTION_TEST_FILE_NAME
            )
        
@dataclass
class DataValidationConfig:
    def __init__(self):
        self.UTILS = MainUtils()
        self.SCHEMA_CONFIG = self.UTILS.read_yaml_file(filename=SCHEMA_FILE_PATH)

        self.DATA_INGESTION_ARTEFACTS_DIR: str = os.path.join(
            from_root(), ARTEFACTS_DIR, DATA_INGESTION_ARTEFACTS_DIR
        )
        self.DATA_VALIDATION_ARTEFACTS_DIR: str = os.path.join(
            from_root(), ARTEFACTS_DIR, DATA_VALIDATION_ARTEFACT_DIR
            )
        self.DATA_DRIFT_FILE_PATH: str = os.path.join(
            self.DATA_VALIDATION_ARTEFACTS_DIR, DATA_DRIFT_FILE_NAME
            )
        


@dataclass
class DataTransformationConfig:
    def __init__(self):
        self.UTILS = MainUtils()
        self.SCHEMA_CONFIG = self.UTILS.read_yaml_file(filename=SCHEMA_FILE_PATH)

        self.DATA_INGESTION_ARTEFACTS_DIR: str = os.path.join(
            from_root(), ARTEFACTS_DIR, DATA_INGESTION_ARTEFACTS_DIR
        )
        self.DATA_TRANSFORMATION_ARTEFACTS_DIR: str = os.path.join(
            from_root(), ARTEFACTS_DIR, DATA_TRANSFORMATION_ARTEFACTS_DIR
            )
        self.TRANSFORMED_TRAIN_DATA_DIR: str = os.path.join(
            self.DATA_TRANSFORMATION_ARTEFACTS_DIR, TRANSFORMED_TRAIN_DATA_DIR
            )
        self.TRANSFORMED_TEST_DATA_DIR: str = os.path.join(
            self.DATA_TRANSFORMATION_ARTEFACTS_DIR, TRANSFORMED_TEST_DATA_DIR
            )
        self.TRANSFORMED_TRAIN_FILE_PATH: str = os.path.join(
            self.TRANSFORMED_TRAIN_DATA_DIR, TRANSFORMED_TRAIN_DATA_FILE_NAME
            )
        self.TRANSFORMED_TEST_FILE_PATH: str = os.path.join(
            self.TRANSFORMED_TEST_DATA_DIR, TRANSFORMED_TEST_DATA_FILE_NAME
            )
        self.PREPROCESSOR_FILE_PATH: str = os.path.join(
            from_root(), ARTEFACTS_DIR, DATA_TRANSFORMATION_ARTEFACTS_DIR, PREPROCESSOR_OBJECT_FILE_NAME
            )


# Model Evaluation Configurations
@dataclass
class ModelTrainerConfig:
    def __init__(self):
        self.UTILS = MainUtils()
        
        self.DATA_TRANSFORMATION_ARTEFACTS_DIR: str = os.path.join(
            from_root(), ARTEFACTS_DIR, DATA_TRANSFORMATION_ARTEFACTS_DIR
        )
        self.MODEL_TRAINER_ARTEFACTS_DIR: str = os.path.join(
            from_root(), ARTEFACTS_DIR, MODEL_TRAINER_ARTEFACTS_DIR
            )
        self.PREPROCESSOR_OBJECT_FILE_PATH: str = os.path.join(
            self.DATA_TRANSFORMATION_ARTEFACTS_DIR, PREPROCESSOR_OBJECT_FILE_NAME
    
        )
        self.TRAINED_MODEL_FILE_PATH: str = os.path.join(
            from_root(), ARTEFACTS_DIR, MODEL_TRAINER_ARTEFACTS_DIR, MODEL_FILE_NAME
            )