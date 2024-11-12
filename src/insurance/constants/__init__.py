import os
from os import environ
from datetime import datetime
from from_root import from_root

TIMESTAMP: str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
CURRENT_YEAR =  datetime.now().year

# Configuration file
MODEL_CONFIG_FILE = "config/model.yaml"
PARAM_FILE_PATH = "params.yaml"
SCHEMA_FILE_PATH = "config/schema.yaml"

SOURCE_URL = "https://github.com/donadviser/datasets/raw/master/data-don/auto_insurance_claim_fraud.csv"
LOCAL_FILE_NAME = "insurance_fraud_claims.csv"

TARGET_COLUMN = "fraud_reported"

TEST_SIZE = 0.2

ARTEFACTS_ROOT_DIR = os.path.join(from_root(), "artefacts")
ARTEFACTS_DIR = os.path.join(ARTEFACTS_ROOT_DIR, TIMESTAMP)

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
TRANSFORM_FEATURES_DICT_FILE_NAME = "transformed_features.pkl"

MODEL_TRAINER_ARTEFACTS_DIR = "ModelTrainerArtefacts"
MODEL_FILE_NAME = "best_model_insurance_claim_fraud.pkl"
MODEL_SAVE_FORMAT = ".pkl"
BEST_MODEL_ARTEFACTS_DIR = "BestModelArtefacts"
"""
MODEL EVALUATION related constant #S3 BUCKET
"""
METRIC_ARTEFACTS_DIR = "MetricArtefacts"
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02
MODEL_BUCKET_NAME:str = "insurance-claim-fraud-model"
MODEL_PUSHER_S3_KEY = "model-registry"
S3_MODEL_NAME = "best_model_insurance_claim_fraud.pkl"


APP_HOST = "0.0.0.0"
APP_PORT = 8080

# MlFlow
TRACKING_URI = "http://127.0.0.1:5000"