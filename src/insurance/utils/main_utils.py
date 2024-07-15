import shutil
import sys
from typing import Dict, Tuple, List
import dill
import pandas as pd
import numpy as np
import yaml
from yaml import safe_dump


#import xgboost
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.utils import all_estimators

from insurance import logging, CustomException
from insurance.constants import *
 

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