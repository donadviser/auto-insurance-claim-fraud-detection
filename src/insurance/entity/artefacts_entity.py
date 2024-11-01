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
    validation_message: str


@dataclass
class DataTransformationArtefacts:
    transform_features_file_path: str
    transform_features_dict: dict
    #transformed_test_file_path: str

@dataclass
class ModelTrainerArtefacts:
    trained_model_file_path: str
    

@dataclass
class ModelEvaluationArtefacts:
    is_model_accepted:bool
    changed_accuracy:float
    trained_model_path:str
    

@dataclass
class ModelPusherArtefacts:
    model_bucket_name: str
    s3_model_path: str