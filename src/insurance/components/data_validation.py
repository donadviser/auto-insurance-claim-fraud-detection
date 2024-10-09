import json
import os
import sys
import pandas as pd
from typing import Tuple, Union

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


from insurance import logging
from insurance import CustomException
from insurance.entity.config_entity import DataValidationConfig
from insurance.entity.artefacts_entity import (
    DataIngestionArtefacts,
    DataValidationArtefacts,
)

class DataValidation:
    def __init__(
            self,
            data_ingestion_artefacts: DataIngestionArtefacts,
            data_validation_config: DataValidationConfig,
            
    ):
        self.data_ingestion_artefacts = data_ingestion_artefacts
        self.data_validation_config = data_validation_config
        

    # This method is used to validate schema columns
    def validate_schema_columns(self, df:pd.DataFrame) -> bool:
        """
        Validate schema columns.

        Args:
            df (pd.DataFrame): The dataframe to be validated.

        Returns:
            bool: The validation status.
        """
        logging.info("Entered the validate_schema_columns method of DataValidation class.")
        try:
            logging.info(f"length of columns - df.columns: {len(df.columns)}")
            logging.info(f'length of columns from schema: {len(self.data_validation_config.SCHEMA_CONFIG["columns"])}')
            # Checking the len of the dataframe columns and schema file columns
            if len(df.columns) == len(self.data_validation_config.SCHEMA_CONFIG["columns"]):
                validation_status = True
            else:
                validation_status = False
            return validation_status
            
        except Exception as e:
            raise CustomException(e, sys)

    # This method is used to validate the numerical columns
    def is_numerical_column_exists(self, df:pd.DataFrame) -> bool:
        """
        Validate numerical columns.

        Args:
            df (pd.DataFrame): The dataframe to be validated.

        Returns:
            bool: The validation status.
        """
        logging.info("Entered the validate_numerical_columns method of DataValidation class.")
        try:
            validation_status = False

            # Checking numerical schema columns with dataframe numerical columns
            for column in self.data_validation_config.SCHEMA_CONFIG["numerical_columns"]:
                if column not in df.columns:
                    validation_status = False
                    logging.info(f"Numerical column - {column} not found in dataframe")
                else:
                    validation_status = True

            return validation_status
        except Exception as e:
            raise CustomException(e, sys)
        
    # This method is used to validate the categorical columns
    def is_categorical_column_exists(self, df:pd.DataFrame) -> bool:
        """
        Validate categorical columns.

        Args:
            df (pd.DataFrame): The dataframe to be validated.

        Returns:
            bool: The validation status.
        """
        logging.info("Entered the validate_categorical_columns method of DataValidation class.")
        try:
            validation_status = False

            # Checking categorical schema columns with dataframe categorical columns
            for column in self.data_validation_config.SCHEMA_CONFIG["categorical_columns"]:
                if column not in df.columns:
                    validation_status = False
                    logging.info(f"Categorical column - {column} not found in dataframe")
                else:
                    validation_status = True

            return validation_status
        except Exception as e:
            raise CustomException(e, sys)
        
    # This method is used to validate dataset schema columns
    def validate_dataset_schema_columns(self) -> Tuple[bool, bool]:
        """
        Validate dataset schema.

        Args:
            df (pd.DataFrame): The dataframe to be validated.

        Returns:
            bool: The validation status.
        """
        logging.info("Entered the validate_dataset_schema method of DataValidation class.")
        try:
            logging.info("Validating dataset schema columns")
            
            # Validating schema columns for Train dataframe
            train_schema_status = self.validate_schema_columns(self.train_set)
            logging.info("Validated dataset schema columns on the train set")

            # Validating schema columns for Test dataframe
            test_schema_status = self.validate_schema_columns(self.test_set)
            logging.info("Validated dataset schema columns on the test set")

            logging.info("Validated dataset schema columns")

            return train_schema_status, test_schema_status
        except Exception as e:
            raise CustomException(e, sys)
        

    # This method is used to validate dataset numerical columns
    def validate_is_numerical_column_exists(self) -> Tuple[bool, bool]:
        """
        Validate dataset numerical columns.

        Args:
            df (pd.DataFrame): The dataframe to be validated.

        Returns:
            bool: The validation status.
        """
        logging.info("Entered the validate_is_numerical_column_exists method of DataValidation class.")
        try:
            logging.info("Validating dataset numerical columns")
            
            # Validating numerical columns for Train dataframe
            train_num_datatype_status = self.is_numerical_column_exists(self.train_set)
            logging.info("Validated dataset numerical columns on the train set")

            # Validating numerical columns for Test dataframe
            test_num_datatype_status = self.is_numerical_column_exists(self.test_set)
            logging.info("Validated dataset numerical columns on the test set")

            logging.info("Validated dataset numerical columns")

            return train_num_datatype_status, test_num_datatype_status
        except Exception as e:
            raise CustomException(e, sys)
        
    # This method is used to validate the categorical columns
    def validate_is_categorical_column_exists(self) -> Tuple[bool, bool]:
        """
        Validate dataset categorical columns.

        Args:
            df (pd.DataFrame): The dataframe to be validated.

        Returns:
            bool: The validation status.
        """
        logging.info("Entered the validate_is_categorical_column_exists method of DataValidation class.")
        try:
            logging.info("Validating dataset categorical columns")
            
            # Validating categorical columns for Train dataframe
            train_cat_datatype_status = self.is_categorical_column_exists(self.train_set)
            logging.info("Validated dataset categorical columns on the train set")

            # Validating categorical columns for Test dataframe
            test_cat_datatype_status = self.is_categorical_column_exists(self.test_set)
            logging.info("Validated dataset categorical columns on the test set")

            logging.info("Validated dataset categorical columns")

            return train_cat_datatype_status, test_cat_datatype_status
        except Exception as e:
            raise CustomException(e, sys)
            
    def detect_dataset_drift(self, reference: pd.DataFrame, production: pd.DataFrame, get_ratio: bool = False) -> Union[bool, float]:
        """
        Detect data drift between reference and production dataframes.

        Args:
            reference (pd.DataFrame): The reference dataframe.
            production (pd.DataFrame): The production dataframe.
            get_ratio (bool, optional): Whether to return the drift ratio. Defaults to False.

        Returns:
            Union[bool, float]: True if data drift detected, False otherwise. If get_ratio is True, returns the drift ratio as a float.
        """
        logging.info("Detecting data drift between reference and production dataframes.")

        try:
            # Create a report object with data drift preset
            report = Report(metrics=[DataDriftPreset()])

            # Run the report calculation
            report.run(reference_data=reference, current_data=production)

            # Getting data drift report in json format
            report_json = report.json()
            json_report = json.loads(report_json)

            # Saving the json report in artefacts directory
            data_drift_file_path = self.data_validation_config.DATA_DRIFT_FILE_PATH
            self.data_validation_config.UTILS.write_json_to_yaml(
                json_report, data_drift_file_path
            )

            n_features = json_report["metrics"][0]["result"]["number_of_columns"] 
            n_drifted_features = json_report["metrics"][0]["result"]["number_of_drifted_columns"]
            
            if get_ratio:
                return n_drifted_features / n_features  # Calculating the drift ratio
            else:
                return json_report["metrics"][0]["result"]["dataset_drift"]
            
            
        except Exception as e:
            raise CustomException(e, sys)
        

        
    # This method is used to initiate the data validation.
    def initiate_data_validation(self) -> DataValidationArtefacts:
        """
        Initiate data validation.

        Returns:
            DataValidationArtefacts: The data validation artefacts.
        """
        logging.info("Entered the initiate_data_validation method of DataValidation class.")
        try:
            # Reading the Train and Test data from Data Ingestion Artefacts folder

            self.train_set = pd.read_csv(
                self.data_ingestion_artefacts.train_data_file_path
            )
            self.test_set = pd.read_csv(
                self.data_ingestion_artefacts.test_data_file_path
            )
            
            logging.info(f"The length of train_set columns before validataion is {len(self.train_set.columns)}")
            logging.info(f"The length of test_set columns before validataion is {len(self.test_set.columns)}")
            
            logging.info("Initiated data validation for the dataset")

            # Creating the Data Validation Artefacts directory
            os.makedirs(
                self.data_validation_config.DATA_VALIDATION_ARTEFACTS_DIR, exist_ok=True
            )
            logging.info(f"Created the Data Validation Artefacts for {os.path.basename(self.data_validation_config.DATA_VALIDATION_ARTEFACTS_DIR)}")  

            # Checking the dataset drift
            drift = self.detect_dataset_drift(self.train_set, self.test_set)
            (
                schema_train_col_status,
                schema_test_col_status,
            ) = self.validate_dataset_schema_columns()
            logging.info(f"Schema train cols status is {schema_train_col_status} and schema test cols status is {schema_test_col_status}")

            (
                schema_train_cat_cols_status,
                schema_test_cat_cols_status,
            ) = self.validate_is_categorical_column_exists()
            logging.info(f"Schema train cat cols status is {schema_train_cat_cols_status} and schema test cat cols status is {schema_test_cat_cols_status}")

            (
                schema_train_num_cols_status,
                schema_test_num_cols_status,
            ) = self.validate_is_numerical_column_exists()
            logging.info(f"Schema train num cols status is {schema_train_num_cols_status} and schema test num cols status is {schema_test_num_cols_status}")

            logging.info("Validated dataset schema for numerical datatype")

            # Checking drift status, initially the status is None
            drift_status = None
            if (
                schema_train_cat_cols_status is True and
                schema_test_cat_cols_status is True and
                schema_train_num_cols_status is True and
                schema_test_num_cols_status is True and
                schema_train_col_status is True and
                schema_test_col_status is True and
                drift is False
            ):
                logging.info(f"Dataset schema validation completed")
                drift_status = True
            else:
                drift_status = False

            # Saving data validation artfefacts
            data_validation_artefacts = DataValidationArtefacts(
                data_drift_file_path=self.data_validation_config.DATA_DRIFT_FILE_PATH,
                validation_status=drift_status,
            )

            return data_validation_artefacts
    
        except Exception as e:
            raise CustomException(e, sys)