import sys
import pandas as pd
from typing import Dict
from insurance import logging
from insurance import CustomException
from insurance.constants import *
from insurance.configuration.s3_operations import S3Operations


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
        self.total_claim_amount = total_claim_amount


    def get_data(self) -> Dict:
        """Get the data from the form in the frontend

        Returns:
            Dict: The data
        """

        logging.info("Entered the get_data method of the ShippingData class")

        try:
            input_data = {
                "Policy State": [self.policy_state],
                "Collision Type": [self.collision_type],
                "Property Damage": [self.property_damage],
                "Police Report Available": [self.police_report_available],
                "Insured Sex": [self.insured_sex],
                "Insured Education Level": [self.insured_education_level],
                "Insured Relationship": [self.insured_relationship],
                "Incident Type": [self.incident_type],
                "Incident Severity": [self.incident_severity],
                "Authorities Contacted": [self.authorities_contacted],
                "Incident State": [self.incident_state],
                "Incident City": [self.incident_city],
                "Policy Deductable": [self.policy_deductable],
                "Number of Vehicles Involved": [self.number_of_vehicles_involved],
                "Bodily Injuries": [self.bodily_injuries],
                "Witnesses": [self.witnesses],
                "Incident Hour of the Day": [self.incident_hour_of_the_day],
                "Months as Customer": [self.months_as_customer],
                "Age": [self.age],
                "Policy Annual Premium": [self.policy_annual_premium],
                "Injury Claim": [self.injury_claim],
                "Property Claim": [self.property_claim],
                "Vehicle Claim": [self.vehicle_claim],
                "Insured Occupation": [self.insured_occupation],
                "Insured Hobbies": [self.insured_hobbies],
                "Auto Make": [self.auto_make],
                "Umbrella Limit": [self.umbrella_limit],
                "Capital Gains": [self.capital_gains],
                "Capital Loss": [self.capital_loss],
                "Total Claim Amount": [self.total_claim_amount]
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

        logging.info("Entered the get_input_data_frame method of the ShippingData class")

        try:
            input_data = self.get_data()
            data_frame = pd.DataFrame(input_data)
            logging.info("Obtained the input data in Python dictionary format")
            logging.info("Exited the get_input_data_frame method of the ShippingData class")
            return data_frame
        except Exception as e:
            raise CustomException(e, sys)



    class CostPredictor:
        def __init__(self):
            self.s3_operations = S3Operations()
            self.bucket_name = MODEL_BUCKET_NAME

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
                logging.info(f"Loaded best mode from s3 bucket: {best_model}")

                # Make predictions using the loaded model
                prediction = best_model.predict(X)

                logging.info("Exited the predict method of the CostPredictor class")
                return prediction
            except Exception as e:
                raise CustomException(e, sys)

