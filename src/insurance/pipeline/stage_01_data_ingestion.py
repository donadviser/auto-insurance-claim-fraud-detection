import sys
from insurance import logging, CustomException

from insurance.entity import (
    DataIngestionConfig,
)
from insurance.entity import (
    DataIngestionArtefacts,
)

from insurance.components import (
    DataIngestion,
)
                              


class DataIngestionTrainingPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def main(self):
        logging.info("Entered the start_data_ingestion method of TrainPipeline class.")
        try:            
            data_ingestion = DataIngestion(self.data_ingestion_config)
            data_ingestion_artefacts = data_ingestion.initiate_data_ingestion()
            logging.info("Exited the start_data_ingestion method of TrainPipeline class.")
            return data_ingestion_artefacts
        except Exception as e:
            raise CustomException(e, sys)






        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()

        df = data_ingestion.get_data_from_local_data_file()
        print(df.head())

