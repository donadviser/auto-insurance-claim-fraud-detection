from dataclasses import dataclass

# Data Ingestion Artefacts

@dataclass
class DataIngestionArtefacts:
    train_data_file_path: str
    test_data_file_path: str


@dataclass
class DataValidationArtefacts:
    data_drift_file_path: str
    validation_status: bool


@dataclass
class DataTransformationArtefacts:
    transformed_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str

@dataclass
class ModelTrainerArtefacts:
    trained_model_file_path: str