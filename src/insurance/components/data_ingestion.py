import os
import sys
from typing import Tuple
import pandas as pd
import urllib.request as request
from zipfile import ZipFile
from tqdm import tqdm
from pathlib import Path
from insurance.entity import DataIngestionConfig
from insurance.entity import DataIngestionArtefacts
from insurance import logging, CustomException
from insurance.utils import get_size
from sklearn.model_selection import train_test_split
from insurance.constants import TEST_SIZE


class DataIngestion:
    def __init__(
            self,
            data_ingestion_config: DataIngestionConfig,
            ):
        self.data_ingestion_config = data_ingestion_config
        

    def get_data_from_data_source(self):
        """

        """
        logging.info("Entered the get_data_from_data_source method of DataIngestion class")
        logging.info("Trying to download file...")
        os.makedirs(self.data_ingestion_config.ROOT_DIR, exist_ok=True)
        os.makedirs(self.data_ingestion_config.RAW_DATA_DIR, exist_ok=True)

         

        if not os.path.exists(self.data_ingestion_config.DOWNLOADED_DATA_FILE_PATH):
            logging.info("Download started...")
            filename, headers = request.urlretrieve(
                url=self.data_ingestion_config.SOURCE_URL,
                filename=self.data_ingestion_config.DOWNLOADED_DATA_FILE_PATH
            )
            logging.info(f"{filename} download! with following info: \n{headers}")

        else:
            logging.info(f"File already exists of size: {get_size(Path(self.data_ingestion_config.DOWNLOADED_DATA_FILE_PATH))}")  

    # This method will fetch data from mongoDB
    def get_data_from_local_data_file(self) -> pd.DataFrame:
        """
        Get the data from csv file.

        Returns:
            pd.DataFrame: The data from csv file.
        """
        logging.info("Entered the get_data_from_file method of DataIngestion class")
        try:

            filename = self.data_ingestion_config.DOWNLOADED_DATA_FILE_PATH
            df = pd.read_csv(filename)
            logging.info(f"Obtaied the dataframe from local data file:  {filename}")
            logging.info("Exited the get_data_from_local_data_file method of DataIngestion class")
            return df
        except Exception as e:
            raise CustomException(e, sys) 


    # This method will split the data into train and test
    def split_data_as_train_test(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the data into train and test.

        Args:
            df (pd.DataFrame): The dataframe to be split.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: The train and test dataframe.
        """
        logging.info("Entered the split_data_as_train_test method of DataIngestion class")
        try:
            # Creating Data Ingestion Artefacts directory inside Artefacts folder            
            os.makedirs(self.data_ingestion_config.DATA_INGESTION_ARTEFACTS_DIR, exist_ok=True)
            
            # Splitting the data into train an test
            train_set, test_set = train_test_split(df, test_size=TEST_SIZE)
            logging.info("Splitted the data into train and test")

            # Creating Train directory inside DataIngestionArtefacts directory
            os.makedirs(self.data_ingestion_config.TRAIN_DATA_ARTEFACT_FILE_DIR, exist_ok=True)
            logging.info(f"Created {os.path.basename(self.data_ingestion_config.TRAIN_DATA_ARTEFACT_FILE_DIR)} directory")

            # Creating Test directory inside DataIngestionArtefacts directory
            os.makedirs(self.data_ingestion_config.TEST_DATA_ARTEFACT_FILE_DIR, exist_ok=True)
            logging.info(f"Created {os.path.basename(self.data_ingestion_config.TEST_DATA_ARTEFACT_FILE_DIR)} directory")

            # Saving the train and test data as csv files
            train_data_file_path = self.data_ingestion_config.TRAIN_DATA_FILE_PATH
            train_set.to_csv(train_data_file_path, index=False, header=True)

            test_data_file_path = self.data_ingestion_config.TEST_DATA_FILE_PATH

            test_set.to_csv(test_data_file_path, index=False, header=True)
            logging.info("Saved the train and test data as csv files")
            logging.info(f"Saved {os.path.basename(self.data_ingestion_config.TRAIN_DATA_FILE_PATH)}, \
                          {os.path.basename(self.data_ingestion_config.TEST_DATA_FILE_PATH)} in \
                            {os.path.basename(self.data_ingestion_config.DATA_INGESTION_ARTEFACTS_DIR)} directory")
            logging.info("Exited the split_data_as_train_test method of DataIngestion class")
            return train_set, test_set
        except Exception as e:
            raise CustomException(e, sys) 


    # This method initiates data ingestion
    def initiate_data_ingestion(self) -> DataIngestionArtefacts:
        """
        Initiate the data ingestion.

        Returns:
            DataIngestionArtefacts: The data ingestion artefacts.

            config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()

        df = data_ingestion.get_data_from_local_data_file()
        print(df.head())

        """
        logging.info("Entered the initiate_data_ingestion method of DataIngestion class")
        try:
            # Getting the data from the data source
            self.get_data_from_data_source()
            df = self.get_data_from_local_data_file()
            print(df.head())

            # Dropping the unnecessary columns from the dataframe
            df1 = df.drop(columns=self.data_ingestion_config.DROP_COLS, axis=1)
            df1 = df1.dropna()
            logging.info("Obtained the data from mongodb and dropped the unnecessary columns from the dataframe")

            # Splitting the data as train set and test set
            train_set, test_set = self.split_data_as_train_test(df1)
            logging.info("Initiated the data ingestion")
            logging.info("Exited the initiate_data_ingestion method of DataIngestion class")

            # Saving data validation artfefacts
            data_ingestion_artefacts = DataIngestionArtefacts(
                train_data_file_path=self.data_ingestion_config.TRAIN_DATA_FILE_PATH,
                test_data_file_path=self.data_ingestion_config.TEST_DATA_FILE_PATH,
            )

            return data_ingestion_artefacts 
        except Exception as e:
            raise CustomException(e, sys)    