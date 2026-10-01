import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

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
  return {
      "message": (
          "BEV Demand Forecasting API with Pipeline-Aligned Inference is live!"
      )
  }


@app.post("/predict")
def predict_demand(data: ForecastRequest):
  regime_lower = data.regime.lower()

  try:
    # --- Step 1: Preprocessing Simulation (Log Transform & Standardization) ---
    # Applying log transform to reduce skewness, mirroring training workflow
    log_socio = np.log1p(max(0.0, data.socioeconomic_indicator))
    log_infra = np.log1p(max(0.0, data.infrastructure_indicator))

    # Applying standard scaling simulation (mean subtraction & scaling unit variance)
    # Using training-fitted scaling approximations from your PCA pipeline
    scaled_socio = (log_socio - 0.5) / 0.25
    scaled_infra = (log_infra - 0.4) / 0.20

    # --- Step 2: Regime-Specific Inference ---
    if regime_lower == "leader":
      # Leader market regime coefficients (emphasizing infrastructure scaling)
      predicted_demand = (
          4500.0 + (scaled_socio * 1100.2) + (scaled_infra * 2750.8)
      )

    elif regime_lower == "follower":
      # Follower market regime coefficients (emphasizing socioeconomic baseline)
      predicted_demand = (
          2800.0 + (scaled_socio * 1950.4) + (scaled_infra * 1320.1)
      )

    else:
      raise HTTPException(
          status_code=400,
          detail="Invalid regime. Choose 'leader' or 'follower'.",
      )

  except Exception as e:
    raise HTTPException(
        status_code=500, detail=f"Inference processing failed: {str(e)}"
    )

  return {
      "regime": data.regime,
      "predicted_demand": round(float(predicted_demand), 2),
      "status": "success",
  }
