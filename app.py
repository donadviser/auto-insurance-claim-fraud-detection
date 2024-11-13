from fastapi import FastAPI, Request
from typing import Optional
from uvicorn import run as app_run
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from insurance.utils.main_utils import MainUtils
from insurance import logging
from insurance.pipeline.training_pipeling import TrainPipeline
from insurance.components.model_predictor import CostPredictor, InsuranceData
from insurance.constants import APP_HOST, APP_PORT

# Create FastAPI app instance
app = FastAPI()

# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Create Jinja2 templates instance
templates = Jinja2Templates(directory="templates")

# Enable CORS for all origins
origins = ["*"]

# Add CORS middleware to the FastAPI app
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Class to handle form data
class DataForm:
    def __init__(self, request: Request):
        self.request: Request = request
        self.policy_state: Optional[str] = None
        self.collision_type: Optional[str] = None
        self.property_damage: Optional[str] = None
        self.police_report_available: Optional[str] = None
        self.insured_sex: Optional[str] = None
        self.insured_education_level: Optional[str] = None
        self.insured_relationship: Optional[str] = None
        self.incident_type: Optional[str] = None
        self.incident_severity: Optional[str] = None
        self.authorities_contacted: Optional[str] = None
        self.incident_state: Optional[str] = None
        self.incident_city: Optional[str] = None
        self.policy_deductable: Optional[int] = None
        self.number_of_vehicles_involved: Optional[int] = None
        self.bodily_injuries: Optional[int] = None
        self.witnesses: Optional[int] = None
        self.incident_hour_of_the_day: Optional[int] = None
        self.months_as_customer: Optional[int] = None
        self.age: Optional[int] = None
        self.policy_annual_premium: Optional[float] = None
        self.injury_claim: Optional[float] = None
        self.property_claim: Optional[float] = None
        self.vehicle_claim: Optional[float] = None
        self.insured_occupation: Optional[str] = None
        self.insured_hobbies: Optional[str] = None
        self.auto_make: Optional[str] = None
        self.umbrella_limit: Optional[float] = None
        self.capital_gains: Optional[float] = None
        self.capital_loss: Optional[float] = None
        self.auto_year: Optional[int] = None
        self.total_claim_amount: Optional[float] = None


    async def get_insurance_data(self):
        form = await self.request.form()
        self.policy_state = form.get("policy_state")
        self.collision_type = form.get("collision_type")
        self.property_damage = form.get("property_damage")
        self.police_report_available = form.get("police_report_available")
        self.insured_sex = form.get("insured_sex")
        self.insured_education_level = form.get("insured_education_level")
        self.insured_relationship = form.get("insured_relationship")
        self.incident_type = form.get("incident_type")
        self.incident_severity = form.get("incident_severity")
        self.authorities_contacted = form.get("authorities_contacted")
        self.incident_state = form.get("incident_state")
        self.incident_city = form.get("incident_city")
        self.policy_deductable = form.get("policy_deductable")
        self.number_of_vehicles_involved = form.get("number_of_vehicles_involved")
        self.bodily_injuries = form.get("bodily_injuries")
        self.witnesses = form.get("witnesses")
        self.incident_hour_of_the_day = form.get("incident_hour_of_the_day")
        self.months_as_customer = form.get("months_as_customer")
        self.age = form.get("age")
        self.policy_annual_premium = form.get("policy_annual_premium")
        self.injury_claim = form.get("injury_claim")
        self.property_claim = form.get("property_claim")
        self.vehicle_claim = form.get("vehicle_claim")
        self.insured_occupation = form.get("insured_occupation")
        self.insured_hobbies = form.get("insured_hobbies")
        self.auto_make = form.get("auto_make")
        self.umbrella_limit = form.get("umbrella_limit")
        self.capital_gains = form.get("capital_gains")
        self.capital_loss = form.get("capital_loss")
        self.auto_year = form.get("auto_year")
        self.total_claim_amount = form.get("total_claim_amount")


# Route to trigger the training pipeline
@app.get("/train")
async def trainRouteClient():
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()

        return Response(status_code=200)
    except Exception as e:
        logging.error(f"Error training the pipeline: {str(e)}")
        return Response(f"status_code=500: Error occured {e}")


# Route to render the prediction form
@app.get("/predict")
async def predictGetRouteClient(request: Request):
    try:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "context": "Rendering"}
        )
    except Exception as e:
        logging.error(f"Error rendering the prediction form: {str(e)}")
        return Response(f"status_code=500: Error occured: {e}")


@app.post("/predict")
async def predictRouteClient(request: Request):
    try:
        form = DataForm(request)
        await form.get_insurance_data()

        insurance_data = InsuranceData(
            policy_state=form.policy_state,
            collision_type=form.collision_type,
            property_damage=form.property_damage,
            police_report_available=form.police_report_available,
            insured_sex=form.insured_sex,
            insured_education_level=form.insured_education_level,
            insured_relationship=form.insured_relationship,
            incident_type=form.incident_type,
            incident_severity=form.incident_severity,
            authorities_contacted=form.authorities_contacted,
            incident_state=form.incident_state,
            incident_city=form.incident_city,
            policy_deductable=form.policy_deductable,
            number_of_vehicles_involved=form.number_of_vehicles_involved,
            bodily_injuries=form.bodily_injuries,
            witnesses=form.witnesses,
            incident_hour_of_the_day=form.incident_hour_of_the_day,
            months_as_customer=form.months_as_customer,
            age=form.age,
            policy_annual_premium=form.policy_annual_premium,
            injury_claim=form.injury_claim,
            property_claim=form.property_claim,
            vehicle_claim=form.vehicle_claim,
            insured_occupation=form.insured_occupation,
            insured_hobbies=form.insured_hobbies,
            auto_make=form.auto_make,
            umbrella_limit=form.umbrella_limit,
            capital_gains=form.capital_gains,
            capital_loss=form.capital_loss,
            auto_year=form.auto_year,
            total_claim_amount=form.total_claim_amount
        )

        logging.info("To get the input data")
        cost_df = insurance_data.get_input_data_frame()
        #logging.info(f"Obtained the input data: {cost_df.head()}")
        cost_predictor = CostPredictor()
        cost_value = cost_predictor.predict(X=cost_df)
        cost_value = int(cost_value[0][0])
        predicted_result = {0: "NO", 1: "YES"}.get(cost_value, "Unknown")
        logging.info(f"cost_value: {cost_value} | predicted_result: {predicted_result}")

        return templates.TemplateResponse(
            "index.html",
            {"request": request, "context":  predicted_result}
        )
    except Exception as e:
        logging.error(f"Error predicting the cost: {str(e)}")
        return {"status": False, "error": f"{e}"}


if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)


