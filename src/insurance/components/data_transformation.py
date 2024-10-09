import os
import sys
import pandas as pd
import numpy as np
from typing import List

from insurance import logging
from insurance import CustomException
from insurance.constants import CURRENT_YEAR

from imblearn.combine import SMOTEENN
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
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
            self.ALL_FEATURES = numerical_features+onehot_features+ordinal_features+transform_features
            logging.info(f"self.ALL_FEATURES: {self.ALL_FEATURES}")
            logging.info(f"Length of ALL FEATURES: {len(self.ALL_FEATURES)}")

            # Creating the transformer object
            onehot_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ]
                                               )

            ordinal_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
                ]
                                           )

            numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
                ]
                                             )
            
            power_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('transformer', PowerTransformer(method='yeo-johnson'))
                ]
                                         )    
            logging.info("Initialised the StandardScaler, OneHotEncoder, OrdinalEncoder and PowerTransformer")

            # Using transformer objects in column transformer
            preprocessor = ColumnTransformer(
                transformers = [
                    ('OneHotEncoder', onehot_transformer, onehot_features),
                    ("Ordinal_Encoder", ordinal_transformer, ordinal_features),
                    ("PowerTransformer", power_transformer, transform_features), 
                    ('StandardScaler', numerical_transformer, numerical_features)
                    ]
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
            logging.info("Performing _outlier_capping for columns in the dataframe")
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
        
    @staticmethod    
    def preprocess_data(df: pd.DataFrame, current_year: int, 
                        drop_cols: List, bins_hour: List, names_period: List
                        ) -> pd.DataFrame:
        """
        Preprocess the dataset by applying various transformations including
        handling missing values, calculating vehicle age, and creating new features.

        Parameters:
        - df: pandas DataFrame
        - current_year: int, the current year for calculating vehicle age
        - drop_cols: list, columns to drop after feature engineering
        
        Returns:
        - processed DataFrame
        """
        try:
            processed_data = (df
                            .replace('?', 'unknown')  # Replace '?' with 'unknown' or np.nan
                            .assign(vehicle_age=current_year - df['auto_year'],  # Calculate vehicle age
                                    incident_period_of_day=pd.cut(pd.to_numeric(df['incident_hour_of_the_day'], errors='coerce'),  # Handle 'unknown' as NaN in numeric columns
                                                                        bins=bins_hour, 
                                                                        labels=names_period)  # Create period of day feature
                                    )
                            .drop(columns=drop_cols, errors='ignore')
                            
                            )
            return processed_data 
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
                input_feature_train_df = self.train_set.drop(columns=[target_column_name])
                target_feature_train_df = self.train_set[target_column_name]
                logging.info("Obtained the input features and target feature for Training dataset")

                # Gettting the input features and target feature for Test dataset
                input_feature_test_df = self.test_set.drop(columns=[target_column_name], axis=1)
                target_feature_test_df = self.test_set[target_column_name]
                logging.info("Obtained the input features and target feature for Test dataset")
                
                
                # Data Cleaning and Feature Engineering for Training dataset
                # Getting the bins and namess from config file
                bins_hour = self.data_transformation_config.SCHEMA_CONFIG['incident_hour_time_bins']['bins_hour']
                names_period = self.data_transformation_config.SCHEMA_CONFIG['incident_hour_time_bins']['names_period']
                drop_columns = self.data_transformation_config.SCHEMA_CONFIG['drop_columns']
                yes_no_map = self.data_transformation_config.SCHEMA_CONFIG['yes_no_map']
                
                # Apply the preprocess_data function to both train and test datasets
                input_processed_train_data = self.preprocess_data(df=input_feature_train_df, current_year=CURRENT_YEAR,
                                                                drop_cols=drop_columns, bins_hour=bins_hour, names_period=names_period)
                input_processed_test_data = self.preprocess_data(df=input_feature_test_df, current_year=CURRENT_YEAR,
                                                                drop_cols=drop_columns, bins_hour=bins_hour, names_period=names_period)
                logging.info("Applied the preprocess_data function to both train and test datasets")
                
                # Update the dataframe with the yes_no_map
                target_processed_train_data = (target_feature_train_df.map(yes_no_map))                                    
                target_processed_test_data = (target_feature_test_df.map(yes_no_map))
                logging.info("Updated the y_train and y_test dataset with the yes_no_map")
                
                # Applying preprocessing object on training dataframe and testing dataframe
                input_feature_train_array = preprocessor.fit_transform(input_processed_train_data)
                input_feature_test_array = preprocessor.transform(input_processed_test_data)
                logging.info("Applied the preprocessor object on training and testing dataframe")
                
                # Applying SMOTEENN sampling strategy for imbalance datasets
                logging.info("Applying SMOTEENN on Training dataset")

                smt = SMOTEENN(sampling_strategy="minority")

                input_feature_train_final, target_feature_train_final = smt.fit_resample(
                    input_feature_train_array, target_processed_train_data
                )
                logging.info("Applied SMOTEENN on training dataset")

                logging.info("Applying SMOTEENN on testing dataset")
                input_feature_test_final, target_feature_test_final = smt.fit_resample(
                    input_feature_test_array, target_processed_test_data
                )
                logging.info("Applied SMOTEENN on testing dataset")
                
                
                
                # Concatenating input features and target features array for Train dataset and Test dataset
                train_array = np.c_[
                    input_feature_train_final,
                    target_feature_train_final.values.reshape(-1, 1),
                ]

                test_array = np.c_[
                    input_feature_test_final,
                    target_feature_test_final.values.reshape(-1, 1),
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
            else:
                logging.info("Data validation failed. Skipping data transformation")
                data_transformation_artefacts = None  # Returning None if data validation fails.
                logging.info("Exited the initiate_data_transformation method of DataTransformation class.")
                raise Exception(self.data_validation_artefacts.validation_message)    

            return data_transformation_artefacts
        except Exception as e:
            raise CustomException(e, sys)


            