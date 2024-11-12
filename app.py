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
from insurance.components.model_predictor import CostPredictor, ShippingData
from insurance.constants import APP_HOST, APP_PORT