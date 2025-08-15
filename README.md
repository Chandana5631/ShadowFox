# Loan Approval Prediction

## Overview
Predict whether a loan should be approved based on applicant profile and loan attributes. Includes EDA, training with imbalance handling, model persistence, and an optional FastAPI inference service.

## Project Structure
```
/workspace
  ├─ src/loan_approval
  │   ├─ __init__.py
  │   ├─ data.py
  │   ├─ features.py
  │   ├─ preprocess.py
  │   └─ train.py
  ├─ fastapi_app/main.py
  ├─ notebooks/LoanApproval_EDA.ipynb
  ├─ requirements.txt
  ├─ artifacts/
  └─ outputs/
```

## Setup
1) (Optional) Create and activate a virtualenv.
2) Install dependencies:
```bash
pip install -r /workspace/requirements.txt
```

## Training
Provide a CSV path or URL. If it is a Google Drive link, `gdown` will be used automatically.
```bash
PYTHONPATH=/workspace/src \
python -m loan_approval.train \
  --data-path /path/to/loan.csv \
  --target-col Loan_Status \
  --use-smote \
  --artifact-path /workspace/artifacts/loan_model.joblib \
  --output-dir /workspace/outputs
```

Key options:
- `--target-col`: name of the target column (default: `Loan_Status`).
- `--use-smote`: enable SMOTE oversampling inside the pipeline.
- `--models`: comma-separated list in {`log_reg`,`rf`} (default runs both and picks best by ROC-AUC).

Outputs:
- Saved model pipeline: `/workspace/artifacts/loan_model.joblib`
- Metrics, reports and plots under `/workspace/outputs`

## Serve API
Start FastAPI app after training (or with an existing model file):
```bash
export LOAN_MODEL_PATH=/workspace/artifacts/loan_model.joblib
PYTHONPATH=/workspace/src uvicorn fastapi_app.main:app --host 0.0.0.0 --port 8000
```

Example request:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"records": [{"Gender": "Male", "Married": "Yes", "ApplicantIncome": 5000, "CoapplicantIncome": 2000, "LoanAmount": 150, "Loan_Amount_Term": 360, "Credit_History": 1.0, "Property_Area": "Urban"}]}'
```

## Notes
- Feature engineering (total income and debt-to-income ratio) is integrated into the training pipeline and reused at inference.
- The pipeline includes robust preprocessing: imputers, one-hot encoding for categoricals, scaling for numerics, optional SMOTE, and classifier.
- For XGBoost/LightGBM, you can extend the trainer; they are optional and not required to run this project.
