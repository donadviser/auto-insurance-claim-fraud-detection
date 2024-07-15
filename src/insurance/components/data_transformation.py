import os
import sys
import pandas as pd
import numpy as np

from insurance import logging
from insurance import CustomException

from category_encoders.binary import BinaryEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from insurance.entity import DataTransformationConfig

from insurance.entity import (
    DataIngestionArtefacts,
    DataTransformationArtefacts,
    )

class DataTransformation:
    def __init__(
            self,
            data_ingestion_artefacts: DataIngestionArtefacts,
            data_transformation_config: DataTransformationConfig,
    ):
        
        self.data_ingestion_artefacts = data_ingestion_artefacts
        self.data_transformation_config = data_transformation_config

        # Reading the Train and Test data from Data Ingestion Artefacts folder
        self.train_set = pd.read_csv(self.data_ingestion_artefacts.train_data_file_path)
        self.test_set = pd.read_csv(self.data_ingestion_artefacts.test_data_file_path)
        logging.info("Initiated data transformation for the dataset")

    
    # This method is used to get the transformer object
    def get_data_transformer_object(self) -> object:
        """
        Get the data transformer object. This method gives preprocessor object

        Returns:
            object: The data transformer object.
        """
        logging.info("Entered the get_data_transformer_object method of DataTransformation class.")

        try:
            # Getting neccessary column names from config file
            numerical_columns = self.data_transformation_config.SCHEMA_CONFIG['numerical_columns']
            onehot_columns = self.data_transformation_config.SCHEMA_CONFIG['onehot_columns']
            binary_columns = self.data_transformation_config.SCHEMA_CONFIG['binary_columns']
            logging.info(f"Obtained the NUMERICAL COLS, ONE HOT COLS, BINARY COLS from schema config")

            # Creating the transformer object
            numerical_transformer  = StandardScaler()
            oh_transformer = OneHotEncoder(handle_unknown='ignore')
            binary_transformer = BinaryEncoder()
            logging.info(f"Initialised the StandardScaler, OneHotEncoder and BinaryEncoder")

            # Using transformer objects in column transformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('OneHotEncoder', oh_transformer, onehot_columns),
                    ('BinaryEncoder', binary_transformer, binary_columns),
                    ('StandardScaler', numerical_transformer, numerical_columns),                   
                ],
            )
            logging.info("Created preprocessor object from ColumnTransformer.")
            logging.info("Exited the get_data_transformer_object method of DataTransformation class.")
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        
    # This is a static method for capping the outlier
    @staticmethod
    def _outlier_capping(col: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Capping the outlier.

        Args:
            col (str): The column name.
            df (pd.DataFrame): The dataframe to be capped.

        Returns:
            pd.DataFrame: The capped dataframe.
        """
        logging.info("Entered the _outlier_capping method of DataTransformation class.")
        try:
            logging.info(f"Performing _outlier_capping for columns in the dataframe")
            # Calculating the 1st and 3rd quartile
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            logging.info("Calculated the 1st and 3rd quartile")

            # Calculating the lower and upper limit
            lower_limit = q1 - 1.5 * iqr
            upper_limit = q3 + 1.5 * iqr
            logging.info("Calculated the lower and upper limit")

            # Capping the outlier
            df[col] = np.where(
                df[col] > upper_limit,
                upper_limit,
                np.where(df[col] < lower_limit, lower_limit, df[col]),
            )
            logging.info("Performed the _outlier_capping for columns in the dataframe")
            logging.info("Exited _outlier_capping method of Data_Transformation class")
            return df
        except Exception as e:
            raise CustomException(e, sys)
        
    # This method is used to initialise data transformation
    def initiate_data_transformation(self) -> DataTransformationArtefacts:
        """
        Initiate data transformation.

        Returns:
            DataTransformationArtefacts: The data transformation artefacts.
        """
        logging.info("Entered the initiate_data_transformation method of DataTransformation class.")
        try:
            # Creating directory for data transformation artefacts             
            os.makedirs(self.data_transformation_config.DATA_TRANSFORMATION_ARTEFACTS_DIR, exist_ok=True)
            logging.info(f"Created the data transformation artefacts directory for {os.path.basename(self.data_transformation_config.DATA_TRANSFORMATION_ARTEFACTS_DIR)}")

            # Getting the preprocessor object
            preprocessor = self.get_data_transformer_object()
            logging.info("Obtained the transformer object")

            target_column_name = self.data_transformation_config.SCHEMA_CONFIG['target_column']
            numerical_columns = self.data_transformation_config.SCHEMA_CONFIG['numerical_columns']

            # #Outlier Capping
            # continuous_columns = [
            #     feature
            #     for feature in numerical_columns
            #     if len(self.train_set[feature].unique())>= 25
            # ]
            # [self._outlier_capping(col, self.train_set) for col in continuous_columns]
            # [self._outlier_capping(col, self.test_set) for col in continuous_columns]

            # Gettting the input features and target feature for Training dataset
            input_feature_train_df = self.train_set.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = self.train_set[target_column_name]
            logging.info("Obtained the input features and target feature for Training dataset")

            # Gettting the input features and target feature for Test dataset
            input_feature_test_df = self.test_set.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = self.test_set[target_column_name]
            logging.info("Obtained the input features and target feature for Test dataset")

            # Applying preprocessing object on training dataframe and testing dataframe
            input_feature_train_array = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_array = preprocessor.transform(input_feature_test_df)
            logging.info("Applied the preprocessor object on training and testing dataframe")

            # Concatenating input features and target features array for Train dataset and Test dataset
            train_array = np.c_[
                input_feature_train_array,
                target_feature_train_df.values.reshape(-1, 1),
            ]

            test_array = np.c_[
                input_feature_test_array,
                target_feature_test_df.values.reshape(-1, 1),
            ]
            logging.info("Concatenated the input features and target features array for Train and Test dataset")

            # Creating directory for transformed train dataset array and saving the array
            os.makedirs(
                self.data_transformation_config.TRANSFORMED_TRAIN_DATA_DIR,
                exist_ok=True,
            )

            transformed_train_file = self.data_transformation_config.UTILS.save_numpy_array_data(
                self.data_transformation_config.TRANSFORMED_TRAIN_FILE_PATH,
                train_array,
            )

            # Creating directory for transformed test dataset array and saving the array
            os.makedirs(
                self.data_transformation_config.TRANSFORMED_TEST_DATA_DIR,
                exist_ok=True,
            )

            transformed_test_file = self.data_transformation_config.UTILS.save_numpy_array_data(
                self.data_transformation_config.TRANSFORMED_TEST_FILE_PATH,
                test_array,
            )

            preprocessor_obj_file = self.data_transformation_config.UTILS.save_object(
                self.data_transformation_config.PREPROCESSOR_FILE_PATH,
                preprocessor,
            )
            logging.info("Created the preprocessor object and saving the object")
            logging.info("Created the transformed train dataset array and saving the array")
            logging.info("Created the transformed test dataset array and saving the array")
            logging.info("Exited the initiate_data_transformation method of DataTransformation class.")
            data_transformation_artefacts = DataTransformationArtefacts(
                transformed_object_file_path = preprocessor_obj_file,
                transformed_train_file_path=transformed_train_file,
                transformed_test_file_path=transformed_test_file,
            )

            return data_transformation_artefacts
        except Exception as e:
            raise CustomException(e, sys)


            