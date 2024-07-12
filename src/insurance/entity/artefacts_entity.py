from dataclasses import dataclass

# Data Ingestion Artefacts

@dataclass
class DataIngestionArtefacts:
    train_data_file_path: str
    test_data_file_path: str