# Loan Approval Prediction

A complete, reproducible machine learning project to predict loan approval based on applicant and loan attributes. Includes data preprocessing, modeling (with imbalance handling), evaluation, a saved pipeline, an optional FastAPI service, and a simple frontend.

## Project Structure

- `requirements.txt`: Python dependencies
- `src/loan_approval/`: Package with all ML code
  - `config.py`: Defaults and constants
  - `data_loader.py`: Dataset loading helpers (incl. Google Drive links)
  - `features.py`: Feature engineering utilities
  - `preprocessing.py`: Column detection and preprocessing pipelines
  - `modeling.py`: Model builders (LogReg, RandomForest, optional XGBoost/LightGBM)
  - `train.py`: CLI for training, evaluation, and saving pipeline
  - `infer.py`: CLI for running predictions on JSON input
- `models/`: Saved model artifacts (pipeline `.joblib`)
- `data/`: Place raw datasets here (not committed)
- `api/main.py`: FastAPI app exposing `/predict`
- `frontend/index.html`: Simple UI to call the API
- `notebooks/EDA_and_Modeling.ipynb`: EDA + modeling exploration

## Setup

1) Create virtual environment (optional but recommended)

```bash
python -m venv .venv
source .venv/bin/activate
```

2) Install dependencies

```bash
pip install -r requirements.txt
```

Note: XGBoost/LightGBM are optional; uncomment in `requirements.txt` to install if desired.

3) Make `src` importable for local execution

```bash
export PYTHONPATH="$PYTHONPATH:$(pwd)/src"
```

## Training

```bash
python -m loan_approval.train \
  --data /absolute/path/to/your_dataset.csv \
  --model rf \
  --use-smote \
  --save-path $(pwd)/models/loan_approval_pipeline.joblib
```

Common flags:
- `--model {lr,rf,xgb,lgbm}`: Estimator to use (xgb/lgbm require optional deps)
- `--use-smote`: Apply SMOTE oversampling inside the training pipeline
- `--target Loan_Status`: Target column (default)
- `--test-size 0.15 --val-size 0.17647`: For 70/15/15 split

Artifacts:
- Trained pipeline: `models/loan_approval_pipeline.joblib`
- Metrics JSON: `models/metrics.json`

## FastAPI Service

Start API (ensure `MODEL_PATH` points to your saved pipeline):

```bash
export MODEL_PATH=$(pwd)/models/loan_approval_pipeline.joblib
python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

- Open frontend: `http://localhost:8000` (served by FastAPI)
- Health check: `GET /health`
- Predict: `POST /predict`

Sample request body:

```json
{
  "records": [
    {
      "Gender": "Male",
      "Married": "Yes",
      "Dependents": "0",
      "Education": "Graduate",
      "Self_Employed": "No",
      "ApplicantIncome": 5849,
      "CoapplicantIncome": 0,
      "LoanAmount": 128,
      "Loan_Amount_Term": 360,
      "Credit_History": 1,
      "Property_Area": "Urban"
    }
  ]
}
```

Response:

```json
{
  "predictions": ["Y"],
  "probabilities": [0.83]
}
```

## Notebook

Open `notebooks/EDA_and_Modeling.ipynb` to explore the data (EDA, visualizations, baseline models). It uses the same package code for consistency.

## Notes

- Handles missing data via imputers; categorical encoding via One-Hot; scaling numeric features.
- Optional features: Total income, debt-to-income ratio, log loan amount.
- Class imbalance: class weights and/or SMOTE.
- The saved pipeline can ingest raw JSON records and produce predictions directly.