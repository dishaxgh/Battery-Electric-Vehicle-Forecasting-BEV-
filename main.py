import sys
import numpy as np

# --- Compatibility patch for NumPy version mismatch during unpickling ---
try:
  import numpy._core
except ImportError:
  import numpy.core

  sys.modules["numpy._core"] = numpy.core
# -----------------------------------------------------------------------

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(
    title="BEV Demand Forecasting API",
    description="Master Thesis Production API for Leader & Follower Regimes",
    version="1.0",
)

# Load your actual thesis models from the models folder when the server starts
try:
  follower_model = joblib.load("models/follower_hybrid_weighted_model.pkl")
  leader_model = joblib.load("models/leader_arimax_models.pkl")
  print("Leader and Follower models loaded successfully!")
except Exception as e:
  print(f"Error loading models: {e}")


class ForecastRequest(BaseModel):
  regime: str  # "leader" or "follower"
  socioeconomic_indicator: float
  infrastructure_indicator: float


@app.get("/")
def read_root():
  return {"message": "BEV Demand Forecasting API with Real Thesis Models is live!"}


@app.post("/predict")
def predict_demand(data: ForecastRequest):
  input_data = pd.DataFrame([{
      "socioeconomic_indicator": data.socioeconomic_indicator,
      "infrastructure_indicator": data.infrastructure_indicator,
  }])

  regime_lower = data.regime.lower()

  if regime_lower == "leader":
    if "leader_model" not in globals():
      raise HTTPException(status_code=500, detail="Leader model not loaded.")
    prediction = leader_model.predict(input_data)

  elif regime_lower == "follower":
    if "follower_model" not in globals():
      raise HTTPException(status_code=500, detail="Follower model not loaded.")
    prediction = follower_model.predict(input_data)

  else:
    raise HTTPException(
        status_code=400,
        detail="Invalid regime. Choose 'leader' or 'follower'.",
    )

  return {
      "regime": data.regime,
      "predicted_demand": float(prediction[0]),
      "status": "success",
  }
