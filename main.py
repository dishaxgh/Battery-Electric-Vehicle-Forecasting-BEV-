from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(
    title="BEV Demand Forecasting API",
    description="Master Thesis Production API for Leader & Follower Regimes",
    version="1.0"
)

# Define what input data your API expects from users/recruiters
class ForecastRequest(BaseModel):
    regime: str  # "leader" or "follower"
    socioeconomic_indicator: float
    infrastructure_indicator: float

@app.get("/")
def read_root():
    return {"message": "BEV Demand Forecasting API is live and running!"}

@app.post("/predict")
def predict_demand(data: ForecastRequest):
    # Here is where you would load your model.pkl and make a prediction.
    # For now, this proves the endpoint works live in production!
    
    if data.regime.lower() == "leader":
        # Placeholder calculation for Leader regime
        forecast_value = data.socioeconomic_indicator * 1.5 + data.infrastructure_indicator * 2.0
    else:
        # Placeholder calculation for Follower regime
        forecast_value = data.socioeconomic_indicator * 1.1 + data.infrastructure_indicator * 1.3

    return {
        "regime": data.regime,
        "predicted_demand": forecast_value,
        "status": "success"
    }
