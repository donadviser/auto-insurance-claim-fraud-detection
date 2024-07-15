import os
import sys
import pandas as pd
from typing import List, Tuple

from insurance import logging
from insurance import CustomException
from insurance.constants import MODEL_CONFIG_FILE
from insurance.entity import ModelTrainerConfig
from insurance.entity import (
    DataTransformationArtefacts,
    ModelTrainerArtefacts,
)

class CostModel:
    def __init__(
            self, preprocessing_object: object, trained_model_object: object):
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object


    def predict(self, X) -> float:
        """
        This method predicts the data

        Args:
            X (pd.DataFrame): The data to be predicted.

        Returns:
            float: The predicted data.
        """
        try:
            # Predict the data
            transformed_feature = self.preprocessing_object.transform(X)
            logging.info("Used the trained model to get predictions")
             
            return self.trained_model_object.predict(transformed_feature)
        except Exception as e:
            raise CustomException(e, sys)
        
    def __repr__(self) -> str:
        return f"{type(self.trained_model_object).__name__}()"
    
    def __str__(self) -> str:
        return f"{type(self.trained_model_object).__name__}()"



class ModelTrainer:
    def __init__(
            self,
            data_transformation_artefact: DataTransformationArtefacts,
            model_trainer_config: ModelTrainerConfig,
    ):
        self.data_transformation_artefact = data_transformation_artefact
        self.model_trainer_config = model_trainer_config


    # This method is used to get the trained models
    def get_trained_models(
            self, X_data: pd.DataFrame, y_data: pd.DataFrame
            ) -> List[Tuple[float, object, str]]:
        """
        Get the trained models.

        Args:
            X_data (pd.DataFrame): The X data.
        
        Returns:
            List[Tuple[float, object, str]]: The trained models.
        """
        logging.info("Entered the get_trained_models method of ModelTrainer class.")
        try:
            # Get the model lists from model config file
            model_config = self.model_trainer_config.UTILS.read_yaml_file(
                filename=MODEL_CONFIG_FILE
            )
            models_list = list(model_config["train_model"].keys())
            logging.info("Got model list from tthe config file")

            # Splitting the data in X_train, y_train  X_test, and y_test
            X_train, y_train, X_test, y_test = (
                X_data.drop(X_data.columns[len(X_data.columns) - 1], axis=1),
                X_data.iloc[:, -1],
                y_data.drop(y_data.columns[len(y_data.columns) - 1], axis=1),
                y_data.iloc[:, -1],
            )

            # Getting the trained model list
            tuned_model_list = [
                (
                    self.model_trainer_config.UTILS.get_tuned_model(
                        model_name, X_train, y_train, X_test, y_test
                    )
                )
                for model_name in models_list
            ]
            logging.info("Got the trained model list")
            logging.info("Exited the get_trained_models method of ModelTrainer class.")
            return tuned_model_list
        except Exception as e:
            raise CustomException(e, sys)
        
    # This method is used to initialise model training
    def initiate_model_trainer(self) -> ModelTrainerArtefacts:
        """
        Initiate model training.

        Returns:
            ModelTrainerArtefacts: The model training artefacts.
        """
        logging.info("Entered the initiate_model_trainer method of ModelTrainer class.")
        try:
            # Creating Model Trainer artefacts directory
            os.makedirs(self.model_trainer_config.MODEL_TRAINER_ARTEFACTS_DIR, exist_ok=True)
            logging.info(f"Created the model trainer artefacts directory for {os.path.basename(self.model_trainer_config.MODEL_TRAINER_ARTEFACTS_DIR)}")

            # Loading the train array data and reading it into a DataFrame
            train_array = self.model_trainer_config.UTILS.load_numpy_array_data(
                self.data_transformation_artefact.transformed_train_file_path
            )
            train_df = pd.DataFrame(train_array)
            logging.info("Loaded train array from DataTransformationArtefacts directory and converted into DataFrame")

            #Loading the test array data and reading it into a DataFrame
            test_array = self.model_trainer_config.UTILS.load_numpy_array_data(
                self.data_transformation_artefact.transformed_test_file_path
            )
        
            test_df = pd.DataFrame(test_array)
            logging.info("Loaded test array from DataTransformationArtefacts directory and converted into DataFrame")

            # Getting the models list and finding the best model with score
            list_of_trained_models = self.get_trained_models(train_df, test_df)
            logging.info("Got the list of tuple of model score, model and model_name")

            (
                best_model,
                best_model_score,
            ) = self.model_trainer_config.UTILS.get_best_model_with_name_and_score(
                list_of_trained_models
            )
            logging.info("Got the best model and its score")

            # Loading the preprocessor object
            preprocessor_obj_file_path =  (
                self.data_transformation_artefact.transformed_object_file_path
            )
            print(f"preprocessor_obj_file_path\n {preprocessor_obj_file_path}")
            preprocessing_obj = self.model_trainer_config.UTILS.load_object(
                preprocessor_obj_file_path
            )
            logging.info("Loaded the preprocessor object from DataTransformationArtefacts directory")

            # Reading model config file for getting the best model score
            model_config = self.model_trainer_config.UTILS.read_yaml_file(
                filename=MODEL_CONFIG_FILE
            )
            base_model_score = float(model_config["base_model_score"])
            logging.info("Got the best model score from model config file")

            # Updating the best model score to model config file if the model score is greather than the base model score
            if best_model_score >= base_model_score:
                self.model_trainer_config.UTILS.update_model_score(best_model_score)
                logging.info("Updated the best model score to model config file")

                # Loading cost model object with preprocessor and model
                cost_model = CostModel(preprocessing_obj, best_model)
                logging.info("Created the cost model object with preprocessor and model")
                
                trained_model_path = self.model_trainer_config.TRAINED_MODEL_FILE_PATH
                logging.info("Created best model file path")

                # Saving the cost model in model artefacts directory
                model_file_path = self.model_trainer_config.UTILS.save_object(
                    trained_model_path, cost_model
                )
                logging.info("Saved the best model object path")
            else:
                logging.info("No best mode found: The best model score is less than the base model score")
                #raise "No best model found with score more than base score"
            
            # Savind the Model trainer artefacts
            model_trainer_artefacts = ModelTrainerArtefacts(
                trained_model_file_path=model_file_path
            )
            logging.info("Created the model trainer artefacts")
            logging.info("Exited the initiate_model_trainer method of ModelTrainer class.")
            return model_trainer_artefacts
        except Exception as e:
            raise CustomException(e, sys)
