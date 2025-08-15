from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd


MODEL_PATH = os.environ.get("LOAN_MODEL_PATH", "/workspace/artifacts/loan_model.joblib")

app = FastAPI(title="Loan Approval Prediction API", version="1.0.0")

_model = None


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    return _model


class PredictRequest(BaseModel):
    records: List[Dict[str, Any]]


class PredictResponse(BaseModel):
    predictions: List[int]
    probabilities: Optional[List[float]] = None


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    try:
        model = get_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if len(request.records) == 0:
        raise HTTPException(status_code=400, detail="No records provided")

    df = pd.DataFrame(request.records)
    preds = model.predict(df)
    probs: Optional[List[float]] = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(df)[:, 1].tolist()

    return PredictResponse(
        predictions=[int(p) for p in preds.tolist()],
        probabilities=probs,
    )