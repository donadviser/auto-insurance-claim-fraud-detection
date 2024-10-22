import os
import sys
import pandas as pd
import numpy as np
from typing import List
import copy

from insurance import logging
from insurance import CustomException
from insurance.constants import CURRENT_YEAR

from imblearn.combine import SMOTEENN
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer, OrdinalEncoder

from insurance.entity import DataTransformationConfig

from insurance.entity import (
    DataIngestionArtefacts,
    DataValidationArtefacts,
    DataTransformationArtefacts,
    )

class DataTransformation:
    def __init__(
            self,
            data_ingestion_artefacts: DataIngestionArtefacts,
            data_transformation_config: DataTransformationConfig,
            data_validation_artefacts: DataValidationArtefacts
            ):
        
        self.data_ingestion_artefacts = data_ingestion_artefacts
        self.data_transformation_config = data_transformation_config
        self.data_validation_artefacts = data_validation_artefacts
        self.ALL_FEATURES = []
        
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
            numerical_features = self.data_transformation_config.SCHEMA_CONFIG['numerical_features']
            onehot_features = self.data_transformation_config.SCHEMA_CONFIG['onehot_features']
            ordinal_features = self.data_transformation_config.SCHEMA_CONFIG['ordinal_features']
            transform_features = self.data_transformation_config.SCHEMA_CONFIG['transform_features']
            logging.info("Obtained the NUMERICAL COLS, ONE HOT COLS, ORDINAL COLS and TRANFORMER COLS from schema config")
            
            # Data Cleaning and Feature Engineering for Training dataset
            # Getting the bins and namess from config file
            bins_hour = self.data_transformation_config.SCHEMA_CONFIG['incident_hour_time_bins']['bins_hour']
            names_period = self.data_transformation_config.SCHEMA_CONFIG['incident_hour_time_bins']['names_period']
            drop_columns = self.data_transformation_config.SCHEMA_CONFIG['drop_columns']
            
            yes_no_map = self.data_transformation_config.SCHEMA_CONFIG['yes_no_map']
            
            
    
            #pipeline_preprocessor = copy.deepcopy(pipeline_imbalance)
        
            logging.info("Initialised the CustomTransformers, StandardScaler, OneHotEncoder, OrdinalEncoder and PowerTransformer and Resampling")
            logging.info("Created preprocessor object from ColumnTransformer.")
            logging.info("Exited the get_data_transformer_object method of DataTransformation class.")
            return yes_no_map
        except Exception as e:
            raise CustomException(e, sys)
        

    
    def initiate_data_transformation(self) -> DataTransformationArtefacts:
        """
        Initiate data transformation.
        This method performs data transformation, imputation, outlier capping, and saves the transformed data.

        Returns:
            DataTransformationArtefacts: The data transformation artefacts.
        """
        

        logging.info("Entered the initiate_data_transformation method of DataTransformation class.")
        try:
            if self.data_validation_artefacts.validation_status:      
                os.makedirs(self.data_transformation_config.DATA_TRANSFORMATION_ARTEFACTS_DIR, exist_ok=True)
                logging.info(f"Created the data transformation artefacts directory for {os.path.basename(self.data_transformation_config.DATA_TRANSFORMATION_ARTEFACTS_DIR)}")

                # Getting the preprocessor object
                pipeline_preprocessor = self.get_data_transformer_object()
                logging.info("Obtained the transformer object")

                #target_column_name = self.data_transformation_config.SCHEMA_CONFIG['target_column']
                


                # Gettting the input features and target feature for Training dataset
                #input_feature_train_df = self.train_set.drop(columns=[target_column_name])
                #target_feature_train_df = self.train_set[target_column_name]
                logging.info("Obtained the input features and target feature for Training dataset")

                # Gettting the input features and target feature for Test dataset
                #input_feature_test_df = self.test_set.drop(columns=[target_column_name], axis=1)
                #target_feature_test_df = self.test_set[target_column_name]
                logging.info("Obtained the input features and target feature for Test dataset")
                
                
               #yes_no_map = self.data_transformation_config.SCHEMA_CONFIG['yes_no_map']
                
                
                # Update the dataframe with the yes_no_map
                #target_processed_train_data = (target_feature_train_df.map(yes_no_map))                                    
                #target_processed_test_data = (target_feature_test_df.map(yes_no_map))
                logging.info("Updated the y_train and y_test dataset with the yes_no_map")
               

                preprocessor_obj_file = self.data_transformation_config.UTILS.save_object(
                    self.data_transformation_config.PREPROCESSOR_FILE_PATH,
                    pipeline_preprocessor,
                )
                logging.info("Created the preprocessor object and saving the object")
                logging.info("Created the transformed train dataset array and saving the array")
                logging.info("Created the transformed test dataset array and saving the array")
                logging.info("Exited the initiate_data_transformation method of DataTransformation class.")
                data_transformation_artefacts = DataTransformationArtefacts(
                    transformed_object_file_path = preprocessor_obj_file,
                )
            else:
                logging.info("Data validation failed. Skipping data transformation")
                data_transformation_artefacts = None  # Returning None if data validation fails.
                logging.info("Exited the initiate_data_transformation method of DataTransformation class.")
                raise Exception(self.data_validation_artefacts.validation_message)    

            return data_transformation_artefacts
        except Exception as e:
            raise CustomException(e, sys)
        


            