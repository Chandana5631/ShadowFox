from typing import Any, Dict

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


app = FastAPI(title="Loan Approval Prediction API")


class PredictRequest(BaseModel):
    features: Dict[str, Any]


class PredictResponse(BaseModel):
    approved: int
    probability: float


MODEL_PATH = "outputs/models/loan_model.pkl"


@app.on_event("startup")
def load_model() -> None:
    try:
        app.state.pipeline = joblib.load(MODEL_PATH)
    except Exception as exc:
        app.state.pipeline = None
        raise RuntimeError(f"Failed to load model at {MODEL_PATH}: {exc}") from exc


@app.post("/predict", response_model=PredictResponse)
async def predict(payload: PredictRequest) -> PredictResponse:
    pipeline = getattr(app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")

    import pandas as pd

    X = pd.DataFrame([payload.features])
    try:
        proba = pipeline.predict_proba(X)[:, 1][0]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}")
    approved = int(proba >= 0.5)
    return PredictResponse(approved=approved, probability=float(proba))