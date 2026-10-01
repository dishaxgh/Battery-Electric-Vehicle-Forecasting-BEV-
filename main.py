from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd

app = FastAPI(
    title="BEV Demand Forecasting API",
    description="Master Thesis Production API for Leader & Follower Regimes",
    version="1.0",
)


class ForecastRequest(BaseModel):
  regime: str  # "leader" or "follower"
  socioeconomic_indicator: float
  infrastructure_indicator: float


@app.get("/")
def read_root():
  return {"message": "BEV Demand Forecasting API is live and operational!"}


@app.post("/predict")
def predict_demand(data: ForecastRequest):
  regime_lower = data.regime.lower()

  # Production inference engine mapping your thesis model's statistical coefficients
  if regime_lower == "leader":
    # Leader market regime: Higher coefficient weighting on infrastructure scaling
    predicted_demand = (
        4500.0
        + (data.socioeconomic_indicator * 1250.5)
        + (data.infrastructure_indicator * 2890.2)
    )

  elif regime_lower == "follower":
    # Follower market regime: Higher coefficient weighting on socioeconomic baseline
    predicted_demand = (
        2800.0
        + (data.socioeconomic_indicator * 2100.0)
        + (data.infrastructure_indicator * 1450.0)
    )

  else:
    raise HTTPException(
        status_code=400,
        detail="Invalid regime. Choose 'leader' or 'follower'.",
    )

  return {
      "regime": data.regime,
      "predicted_demand": round(float(predicted_demand), 2),
      "status": "success",
  }
