from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
	from loan_approval.config import POSITIVE_LABEL
except Exception:
	# Ensure local src is importable when running `uvicorn api.main:app`
	import sys as _sys
	_ROOT = Path(__file__).resolve().parents[1] / "src"
	if str(_ROOT) not in _sys.path:
		_sys.path.append(str(_ROOT))
	from loan_approval.config import POSITIVE_LABEL

APP_DIR = Path(__file__).parent.absolute()
ROOT_DIR = APP_DIR.parent
FRONTEND_DIR = ROOT_DIR / "frontend"
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "loan_approval_pipeline.joblib"

app = FastAPI(title="Loan Approval API", version="0.1.0")

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


class PredictRequest(BaseModel):
	records: List[Dict[str, Any]] = Field(..., description="List of input records")


class PredictResponse(BaseModel):
	predictions: List[str]
	probabilities: Optional[List[float]]


_pipeline = None


def load_pipeline() -> Any:
	global _pipeline
	if _pipeline is not None:
		return _pipeline
	model_path = os.environ.get("MODEL_PATH", str(DEFAULT_MODEL_PATH))
	path = Path(model_path)
	if not path.exists():
		raise FileNotFoundError(f"Model not found at: {path}")
	_pipeline = joblib.load(path)
	return _pipeline


@app.get("/health")
async def health() -> Dict[str, str]:
	try:
		load_pipeline()
		return {"status": "ok"}
	except Exception as e:
		return {"status": f"error: {e}"}


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest) -> PredictResponse:
	try:
		pipeline = load_pipeline()
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))

	if not req.records:
		raise HTTPException(status_code=400, detail="No records provided")

	df = pd.DataFrame.from_records(req.records)
	preds = pipeline.predict(df)
	preds_list = [int(p) if not isinstance(p, (str, bool)) else p for p in preds.tolist()]
	labels = [POSITIVE_LABEL if int(p) == 1 else ("N" if POSITIVE_LABEL == "Y" else f"not_{POSITIVE_LABEL}") for p in preds_list]

	probas_list: Optional[List[float]]
	try:
		probas = pipeline.predict_proba(df)
		if probas.shape[1] == 2:
			probas_list = probas[:, 1].tolist()
		else:
			probas_list = None
	except Exception:
		probas_list = None

	return PredictResponse(predictions=labels, probabilities=probas_list)


if FRONTEND_DIR.exists():
	app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")