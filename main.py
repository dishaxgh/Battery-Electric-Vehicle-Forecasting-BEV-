import sys
import numpy as np

# --- Compatibility patch for NumPy version mismatch ---
try:
  import numpy._core
except ImportError:
  import numpy.core

  sys.modules["numpy._core"] = numpy.core

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(
    title="BEV Demand Forecasting API",
    description="Master Thesis Production API for Leader & Follower Regimes",
    version="1.0",
)

# Global lazy-load variables
_follower_model = None
_leader_model = None


def get_follower_model():
  global _follower_model
  if _follower_model is None:
    print("Lazy-loading follower model...")
    _follower_model = joblib.load("models/follower_hybrid_weighted_model.pkl")
  return _follower_model


def get_leader_model():
  global _leader_model
  if _leader_model is None:
    print("Lazy-loading leader model...")
    _leader_model = joblib.load("models/leader_arimax_models.pkl")
  return _leader_model


class ForecastRequest(BaseModel):
  regime: str  # "leader" or "follower"
  socioeconomic_indicator: float
  infrastructure_indicator: float


@app.get("/")
def read_root():
  return {"message": "BEV Demand Forecasting API with Lazy-Loaded Models is live!"}


@app.post("/predict")
def predict_demand(data: ForecastRequest):
  input_data = pd.DataFrame([{
      "socioeconomic_indicator": data.socioeconomic_indicator,
      "infrastructure_indicator": data.infrastructure_indicator,
  }])

  regime_lower = data.regime.lower()

  try:
    if regime_lower == "leader":
      model = get_leader_model()
      prediction = model.predict(input_data)

    elif regime_lower == "follower":
      model = get_follower_model()
      prediction = model.predict(input_data)

    else:
      raise HTTPException(
          status_code=400,
          detail="Invalid regime. Choose 'leader' or 'follower'.",
      )
  except Exception as e:
    raise HTTPException(
        status_code=500, detail=f"Model prediction failed: {str(e)}"
    )

  return {
      "regime": data.regime,
      "predicted_demand": float(prediction[0]),
      "status": "success",
  }
