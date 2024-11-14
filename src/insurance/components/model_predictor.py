import sys
import pandas as pd
from typing import Dict
from insurance import logging
from insurance import CustomException
from insurance.constants import *
from insurance.configuration.s3_operations import S3Operations
import joblib


class InsuranceData:
    def __init__(self,
                policy_state,
                collision_type,
                property_damage,
                police_report_available,
                insured_sex,
                insured_education_level,
                insured_relationship,
                incident_type,
                incident_severity,
                authorities_contacted,
                incident_state,
                incident_city,
                policy_deductable,
                number_of_vehicles_involved,
                bodily_injuries,
                witnesses,
                incident_hour_of_the_day,
                months_as_customer,
                age,
                policy_annual_premium,
                injury_claim,
                property_claim,
                vehicle_claim,
                insured_occupation,
                insured_hobbies,
                auto_make,
                umbrella_limit,
                capital_gains,
                capital_loss,
                auto_year,
                total_claim_amount,
                ):
        self.policy_state = policy_state
        self.collision_type = collision_type
        self.property_damage = property_damage
        self.police_report_available = police_report_available
        self.insured_sex = insured_sex
        self.insured_education_level = insured_education_level
        self.insured_relationship = insured_relationship
        self.incident_type = incident_type
        self.incident_severity = incident_severity
        self.authorities_contacted = authorities_contacted
        self.incident_state = incident_state
        self.incident_city = incident_city
        self.policy_deductable = policy_deductable
        self.number_of_vehicles_involved = number_of_vehicles_involved
        self.bodily_injuries = bodily_injuries
        self.witnesses = witnesses
        self.incident_hour_of_the_day = incident_hour_of_the_day
        self.months_as_customer = months_as_customer
        self.age = age
        self.policy_annual_premium = policy_annual_premium
        self.injury_claim = injury_claim
        self.property_claim = property_claim
        self.vehicle_claim = vehicle_claim
        self.insured_occupation = insured_occupation
        self.insured_hobbies = insured_hobbies
        self.auto_make = auto_make
        self.umbrella_limit = umbrella_limit
        self.capital_gains = capital_gains
        self.capital_loss = capital_loss
        self.auto_year = auto_year
        self.total_claim_amount = total_claim_amount


    def get_data(self) -> Dict:
        """Get the data from the form in the frontend

        Returns:
            Dict: The data
        """

        logging.info("Entered the get_data method of the ShippingData class")

        try:
            input_data = {
                "policy_state": [self.policy_state],
                "collision_type": [self.collision_type],
                "property_damage": [self.property_damage],
                "police_report_available": [self.police_report_available],
                "insured_sex": [self.insured_sex],
                "insured_education_level": [self.insured_education_level],
                "insured_relationship": [self.insured_relationship],
                "incident_type": [self.incident_type],
                "incident_severity": [self.incident_severity],
                "authorities_contacted": [self.authorities_contacted],
                "incident_state": [self.incident_state],
                "incident_city": [self.incident_city],
                "policy_deductable": int([self.policy_deductable][0]),
                "number_of_vehicles_involved": int([self.number_of_vehicles_involved][0]),
                "bodily_injuries": int([self.bodily_injuries][0]),
                "witnesses": int([self.witnesses][0]),
                "incident_hour_of_the_day": int([self.incident_hour_of_the_day][0]),
                "months_as_customer": int([self.months_as_customer][0]),
                "age": int([self.age][0]),
                "policy_annual_premium": float([self.policy_annual_premium][0]),
                "injury_claim": float([self.injury_claim][0]),
                "property_claim": float([self.property_claim][0]),
                "vehicle_claim": float([self.vehicle_claim][0]),
                "insured_occupation": [self.insured_occupation],
                "insured_hobbies": [self.insured_hobbies],
                "auto_make": [self.auto_make],
                "umbrella_limit": float([self.umbrella_limit][0]),
                "capital_gains": float([self.capital_gains][0]),
                "capital_loss": float([self.capital_loss][0]),
                "auto_year": int([self.auto_year][0]),
                "total_claim_amount": float([self.total_claim_amount][0])
            }
            logging.info("Exited the get_data method of the ShippingData class")
            return input_data
        except Exception as e:
            raise CustomException(e, sys)


    def get_input_data_frame(self) -> pd.DataFrame:
        """Get the input data as a pandas DataFrame

        Returns:
            pd.DataFrame: The data
        """

        logging.info("Entered the get_input_data_frame method of the InsuranceData class")

        try:
            input_data = self.get_data()
            data_frame = pd.DataFrame(input_data)
            logging.info("Obtained the input data in Python dictionary format")
            logging.info("Exited the get_input_data_frame method of the InsuranceData class")
            return data_frame
        except Exception as e:
            raise CustomException(e, sys)



class CostPredictor:
    def __init__(self):
        self.s3_operations = S3Operations()
        self.bucket_name = MODEL_BUCKET_NAME
        #self.model_path = "/Users/donadviser/github-projects/auto-insurance-claim-fraud-detection/artefacts/BestModelArtefacts/best_model_insurance_claim_fraud.pkl"

    def predict(self, X) -> float:
        """Predict the cost using the trained model

        Args:
            X (pd.DataFrame): The input data

        Returns:
            float: The predicted cost
        """

        logging.info("Entered the predict method of the CostPredictor class")

        try:
            # Load the trained model from S3 Bucket
            best_model = self.s3_operations.load_model(S3_MODEL_NAME, self.bucket_name)
            #best_model = joblib.load(self.model_path)

            # Check if the model is loaded successfully
            if not best_model:
                raise ValueError("Failed to load the best model")
            logging.info(f"Loaded best mode from s3 bucket")

            #logging.info(f"X.columns: {X.columns}")

            # Make predictions using the loaded model
            prediction = best_model.predict(X)

            logging.info("Exited the predict method of the CostPredictor class")
            return prediction
        except Exception as e:
            raise CustomException(e, sys)

