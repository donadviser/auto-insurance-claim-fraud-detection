import shutil
import re
import sys
from typing import Dict, Tuple
from typing_extensions import Annotated
import dill
import pandas as pd
import numpy as np
import yaml
from yaml import safe_dump


import xgboost
import lightgbm
import catboost
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.utils import all_estimators

from insurance import logging, CustomException
from insurance.constants import *

import os
import joblib
from pathlib import Path
from typing import Any


class MainUtils:
    def read_yaml_file(self, filename: str) -> Dict:
        logging.info("Entered the read_yaml_file method of MainUtils class.")
        try:
            with open(filename, "rb") as yaml_file:
                data = yaml.safe_load(yaml_file)
            logging.info(f"Successfully read the yaml data from {filename}")
            return data
        except Exception as e:
            raise CustomException(e, sys)

    def write_json_to_yaml(self, json_file: Dict,  yaml_file_path: str) -> yaml:
        logging.info("Entered the write_json_to_yaml method of MainUtils class.")
        try:
            with open(yaml_file_path, "w") as yaml_file:
                safe_dump(json_file, yaml_file, default_flow_style=False)
            logging.info(f"Successfully saved the json data to {yaml_file_path}")
        except Exception as e:
            raise CustomException(e, sys)

    def save_numpy_array_data(self, file_path: str, array: np.array) -> None:
        logging.info("Entered the save_numpy_array_data method of MainUtils class.")
        try:
            with open(file_path, "wb") as file_obj:
                np.save(file_obj, array)
            logging.info(f"Successfully saved the numpy array data to {file_path}")
            return file_path
        except Exception as e:
            raise CustomException(e, sys)

    def load_numpy_array_data(self, file_path: str) -> np.array:
        logging.info("Entered the load_numpy_array_data method of MainUtils class.")
        try:
            with open(file_path, "rb") as file_obj:
                array = np.load(file_obj, allow_pickle=True)
            logging.info(f"Successfully loaded the numpy array data from {file_path}")
            return array
        except Exception as e:
            raise CustomException(e, sys)

    # Separate X and y
    @staticmethod
    def separate_data(data: pd.DataFrame, target_col: str, yes_no_map=None) -> Tuple[
        Annotated[pd.DataFrame, "Features"],
        Annotated[pd.Series, "Target"]
        ]:
        try:
            X = data.drop(columns=[target_col])
            y = data[target_col]
            if yes_no_map is not None:
                y = y.map(yes_no_map)
            return X, y
        except Exception as e:
                raise CustomException(e, sys)

    def get_tuned_model(
            self,
            model_name: str,
            train_x: pd.DataFrame,
            train_y: pd.DataFrame,
            test_x: pd.DataFrame,
            test_y: pd.DataFrame,
    ) -> Tuple[float, object, str]:
        logging.info("Entered the get_tuned_model method of MainUtils class")
        try:
            model = self.get_base_model(model_name)
            model_best_params = self.get_model_params(model, train_x, train_y)
            model.set_params(**model_best_params)
            model.fit(train_x, train_y)
            preds = model.predict(test_x)
            model_score = self.get_model_score(test_y, preds)
            logging.info(f"Successfully tuned the {model_name} model.")
            return model_score, model, model.__class__.__name__
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def get_model_score(test_y: pd.DataFrame, preds: pd.DataFrame) -> float:
        logging.info("Entered the get_model_score method of MainUtils class")
        try:
            model_score = accuracy_score(test_y, preds)
            logging.info(f"Model score: accuracy_score is {model_score}")
            logging.info("Exited the get_model_score method of MainUntils class")
            return model_score
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def get_base_model(model_name: str) -> object:
        logging.info("Entered the get_base_model method of MainUtils class")
        try:
            if model_name.lower().startswith("xgb") is True:
                model = xgboost.__dict__[model_name]()
            elif model_name.lower().startswith("lgb") is True:
                model = lightgbm.__dict__[model_name]()
            elif model_name.lower().startswith("cat") is True:
                model = catboost.__dict__[model_name]()
            else:
                model_idx = [model[0] for model in all_estimators()].index(model_name)
                model = all_estimators().__getitem__(model_idx)[1]()
            logging.info(f"Successfully loaded the {model_name} model.")
            logging.info("Exited the get_base_model method of MainUtils class")
            return model
        except Exception as e:
            raise CustomException(e, sys)


    def get_model_params(self, model: object, X_train: pd.DataFrame, y_train: pd.DataFrame) -> Dict:
        logging.info("Entered the get_model_params method of MainUtils class")
        try:
            VERBOSE = 3
            CV = 5
            N_JOBD = -1

            model_name = model.__class__.__name__
            model_config = self.read_yaml_file(filename=MODEL_CONFIG_FILE)
            model_param_grid = model_config["train_model"][model_name]
            model_grid = GridSearchCV(
                model,
                model_param_grid,
                verbose=VERBOSE,
                cv=CV,
                n_jobs=N_JOBD,
            )
            model_grid.fit(X_train, y_train)
            logging.info(f"Successfully tuned the {model.__class__.__name__} model.")
            logging.info("Exited the get_model_params method of MainUtils class")
            return model_grid.best_params_
        except Exception as e:
            raise CustomException(e, sys)


    @staticmethod
    def save_object(file_path: str, obj: object) -> None:
        logging.info("Entered the save_object method of MainUtils class")
        try:
            with open(file_path, "wb") as file_obj:
                dill.dump(obj, file_obj)
            logging.info(f"Successfully saved the object to {file_path}")
            logging.info("Exited the save_object method of MainUtils class")
            return file_path
        except Exception as e:
            raise CustomException(e, sys)


    @staticmethod
    def load_object(file_path: str) -> object:
        logging.info(f"Entered the load_object method of MainUtils class from: {file_path}")
        try:
            with open(file_path, "rb") as file_obj:
                obj = dill.load(file_obj)
            logging.info(f"Successfully loaded the object from {file_path}")
            logging.info("Exited the load_object method of MainUtils class")
            return obj
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def get_best_model_with_name_and_score(model_list: list) -> Tuple[object, float]:
        logging.info("Entered the get_best_model_with_name_and_score method of MainUtils class")
        try:
            best_score = max(model_list)[0]
            best_model = max(model_list)[1]
            logging.info("Exited the get_best_model_with_name_and_score method of MainUtils class")
            return best_model, best_score
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def create_artefacts_zip(file_name: str, folder_name: str) -> None:
        logging.info("Entered the create_artefacts_zip method of MainUtils class")
        try:
            shutil.make_archive(file_name, "zip", folder_name)
            logging.info("Exited the create_artefacts_zip method of MainUtils class")
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def unzip_file(file_name: str, folder_name: str) -> None:
        logging.info("Entered the unzip_file method of MainUtils class")
        try:
            shutil.unpack_archive(file_name, folder_name)
            logging.info("Exited the unzip_file method of MainUtils class")
        except Exception as e:
            raise CustomException(e, sys)

    def update_model_score(self, best_model_info: Dict) -> None:
        logging.info("Entered the update_model_score method of MainUtils class")
        try:
            model_config = self.read_yaml_file(filename=MODEL_CONFIG_FILE)
            best_model_score = best_model_info['best_model_score']
            best_model_name = best_model_info['best_model_name']

            model_config["base_model_score"] = str(best_model_score)
            model_config["base_model_name"] = str(best_model_name)
            with open(MODEL_CONFIG_FILE, "w+") as file_obj:
                safe_dump(model_config, file_obj, sort_keys=False)
            logging.info("Exited the update_model_score method of MainUtils class")
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def rename_columns_to_snake_case(df: pd.DataFrame) -> pd.DataFrame:
        """
        Renames the columns of a DataFrame to snake_case, handling camel case, acronyms, Pascal case, hyphens,
        multiple spaces, and already snake_case columns.

        Args:
            df: The DataFrame to rename.

        Returns:
            A new DataFrame with the columns renamed to snake_case.
        """
        try:
            def to_snake_case(col_name):
                # Replace hyphens or multiple spaces with an underscore
                col_name = re.sub(r'[-\s]+', '_', col_name)
                # Handle acronyms and split camel case / Pascal case
                col_name = re.sub(r'(?<!^)(?=[A-Z])', '_', col_name).lower()
                # Replace multiple underscores with a single underscore
                return re.sub(r'_+', '_', col_name)

            # Apply the snake_case function to all column names
            return df.rename(columns=lambda col: to_snake_case(col))
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def save_bin(data: Any, path: Path):
        """save binary file

        Args:
            data (Any): data to be saved as binary
            path (Path): path to binary file
        """
        joblib.dump(value=data, filename=path)
        logging.info(f"binary file saved at: {path}")


    @staticmethod
    def load_bin(path: Path) -> Any:
        """load binary data

        Args:
            path (Path): path to binary file

        Returns:
            Any: object stored in the file
        """
        data = joblib.load(path)
        logging.info(f"binary file loaded from: {path}")
        return data



    @staticmethod
    def get_size(path: Path) -> str:
        """get size in KB

        Args:
            path (Path): path of the file

        Returns:
            str: size in KB
        """
        size_in_kb = round(os.path.getsize(path)/1024)
        return f"~ {size_in_kb} KB"