import os
from os import environ
from datetime import datetime
from from_root import from_root

TIMESTAMP: str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
CURRENT_YEAR =  datetime.now().year

# Configuration file
MODEL_CONFIG_FILE = "config/model.yaml"
#CONFIG_FILE_PATH = "config/config.yaml"
SCHEMA_FILE_PATH = "config/schema.yaml"

SOURCE_URL = "https://github.com/donadviser/datasets/raw/master/data-don/insurance_fraud_claims.csv"
LOCAL_FILE_NAME = "insurance_fraud_claims.csv"

TARGET_COLUMN = "fraud_reported"

TEST_SIZE = 0.2

ARTEFACTS_DIR = os.path.join(from_root(), "artefacts", TIMESTAMP)

RAW_DATA_DIR = os.path.join(ARTEFACTS_DIR, "RawData")

DATA_INGESTION_ARTEFACTS_DIR = "DataIngestionArtefacts"
DATA_INGESTION_TRAIN_DIR = "Train"
DATA_INGESTION_TEST_DIR = "Test"
DATA_INGESTION_TRAIN_FILE_NAME = "train.csv"
DATA_INGESTION_TEST_FILE_NAME = "test.csv"

DATA_VALIDATION_ARTEFACT_DIR = "DataValidationArtefacts"
DATA_DRIFT_FILE_NAME = "DataDriftReport.yaml"

DATA_TRANSFORMATION_ARTEFACTS_DIR = "DataTransformationArtefacts"
TRANSFORMED_TRAIN_DATA_DIR = "TransformedTrain"
TRANSFORMED_TEST_DATA_DIR = "TransformedTest"
TRANSFORMED_TRAIN_DATA_FILE_NAME = "transformed_train_data.npz"
TRANSFORMED_TEST_DATA_FILE_NAME = "transformed_test_data.npz"
PREPROCESSOR_OBJECT_FILE_NAME = "insurance_claim_fraud_preprocessor.pkl"

MODEL_TRAINER_ARTEFACTS_DIR = "ModelTrainerArtefacts"
MODEL_FILE_NAME = "insurance_claim_fraud_model.pkl"
MODEL_SAVE_FORMAT = ".pkl"

#S3 BUCKET
BUCKET_NAME = "hexa-shipment-model-io-files"
S3_MODEL_NAME = "insurance_claim_fraud_model.pkl"


APP_HOST = "0.0.0.0"
APP_PORT = 8080